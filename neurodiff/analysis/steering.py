"""
Activation steering and intervention analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from dataclasses import dataclass

from ..core.base import DiffingMethod, DiffResult
from ..core.config import DiffConfig
from ..core.model_loader import DualModelManager

logger = logging.getLogger(__name__)


@dataclass
class SteeringVector:
    """Container for steering vectors."""
    layer: int
    vector: torch.Tensor
    source: str  # "mean_diff", "pca", "targeted", etc.
    metadata: Dict[str, Any] = None


class ActivationSteering(DiffingMethod):
    """Steer models using difference vectors from activation analysis."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.steering_vectors = {}
        self.steering_results = []
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute steering vectors from activation differences."""
        if inputs is None:
            raise ValueError("Inputs required for steering analysis")
            
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Compute steering vectors for each layer
        num_layers = len(manager.base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            # Get mean activation difference as steering vector
            diff_activation = manager.get_activation_diff(inputs, layer_idx)
            
            # Compute mean difference vector
            mean_diff = diff_activation.mean(dim=(0, 1))  # Average over batch and sequence
            
            # Create steering vector
            steering_vec = SteeringVector(
                layer=layer_idx,
                vector=mean_diff,
                source="mean_diff",
                metadata={
                    "norm": mean_diff.norm().item(),
                    "shape": list(mean_diff.shape),
                }
            )
            
            self.steering_vectors[layer_idx] = steering_vec
            
            # Test steering effectiveness
            result = self._test_steering(manager, inputs, steering_vec)
            results.append(result)
            
        return results
    
    def _test_steering(self, manager: DualModelManager,
                      inputs: torch.Tensor,
                      steering_vec: SteeringVector) -> DiffResult:
        """Test the effectiveness of a steering vector."""
        # Get baseline outputs
        with torch.no_grad():
            base_output = manager.base_model.forward(inputs)
            ft_output = manager.ft_model.forward(inputs)
            
        base_logits = base_output.logits
        ft_logits = ft_output.logits
        
        # Apply steering at different intensities
        intensities = [0.1, 0.5, 1.0, 2.0]
        steering_effects = []
        
        for intensity in intensities:
            # Apply steering to base model
            steered_logits = self._apply_steering(
                manager.base_model, inputs, steering_vec, intensity
            )
            
            # Measure effect
            effect = self._measure_steering_effect(
                base_logits, ft_logits, steered_logits
            )
            effect["intensity"] = intensity
            steering_effects.append(effect)
            
        # Find optimal intensity
        similarities = [e["ft_similarity"] for e in steering_effects]
        optimal_idx = np.argmax(similarities)
        optimal_intensity = intensities[optimal_idx]
        
        data = {
            "steering_effects": steering_effects,
            "optimal_intensity": optimal_intensity,
            "max_ft_similarity": max(similarities),
            "baseline_similarity": steering_effects[0]["ft_similarity"] if steering_effects else 0,
            "vector_norm": steering_vec.metadata["norm"],
        }
        
        return DiffResult(
            method="activation_steering",
            layer=steering_vec.layer,
            magnitude=max(similarities),
            direction=steering_vec.vector,
            data=data,
            metadata={
                "source": steering_vec.source,
                "effective": max(similarities) > 0.5,
            }
        )
    
    def _apply_steering(self, model: Any, inputs: torch.Tensor,
                       steering_vec: SteeringVector,
                       intensity: float) -> torch.Tensor:
        """Apply steering vector to model activations."""
        # Create steering hook
        def steering_hook(module, input, output):
            # Add steering vector to activations
            steered_output = output[0] + intensity * steering_vec.vector
            return (steered_output,) + output[1:]
            
        # Apply hook at target layer
        layer_module = model.model.transformer.h[steering_vec.layer]
        handle = layer_module.register_forward_hook(steering_hook)
        
        try:
            with torch.no_grad():
                output = model.forward(inputs)
            return output.logits
        finally:
            handle.remove()
            
    def _measure_steering_effect(self, base_logits: torch.Tensor,
                               ft_logits: torch.Tensor,
                               steered_logits: torch.Tensor) -> Dict[str, float]:
        """Measure the effect of steering."""
        # Convert to probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        ft_probs = F.softmax(ft_logits, dim=-1)
        steered_probs = F.softmax(steered_logits, dim=-1)
        
        # Compute similarities
        kl_base_steered = F.kl_div(
            base_probs.log(), steered_probs, reduction='batchmean'
        ).item()
        
        kl_steered_ft = F.kl_div(
            steered_probs.log(), ft_probs, reduction='batchmean'
        ).item()
        
        # Cosine similarity in logit space
        cos_sim_ft = F.cosine_similarity(
            steered_logits.flatten(), ft_logits.flatten(), dim=0
        ).item()
        
        return {
            "kl_from_base": kl_base_steered,
            "kl_to_ft": kl_steered_ft,
            "ft_similarity": 1.0 / (1.0 + kl_steered_ft),
            "cosine_sim_ft": cos_sim_ft,
            "behavior_shift": kl_base_steered,
        }
    
    def compute_targeted_steering(self, manager: DualModelManager,
                                source_inputs: torch.Tensor,
                                target_inputs: torch.Tensor,
                                layer: int) -> SteeringVector:
        """Compute steering vector for specific behavior change."""
        # Get activations for source and target behaviors
        source_acts = manager.get_base_activations(source_inputs, layer)
        target_acts = manager.get_base_activations(target_inputs, layer)
        
        # Compute difference
        steering_direction = target_acts.mean(dim=(0, 1)) - source_acts.mean(dim=(0, 1))
        
        # Normalize
        steering_direction = steering_direction / (steering_direction.norm() + 1e-8)
        
        return SteeringVector(
            layer=layer,
            vector=steering_direction,
            source="targeted",
            metadata={
                "source_shape": list(source_inputs.shape),
                "target_shape": list(target_inputs.shape),
                "normalized": True,
            }
        )
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize steering effectiveness."""
        figures = {}
        
        # 1. Steering effectiveness by layer
        fig_effectiveness = self._plot_steering_effectiveness(results)
        figures["steering_effectiveness"] = fig_effectiveness
        
        # 2. Intensity response curves
        fig_intensity = self._plot_intensity_response(results)
        figures["intensity_response"] = fig_intensity
        
        # 3. Vector magnitude analysis
        fig_magnitude = self._plot_vector_magnitudes(results)
        figures["vector_magnitudes"] = fig_magnitude
        
        return figures
    
    def _plot_steering_effectiveness(self, results: List[DiffResult]) -> plt.Figure:
        """Plot steering effectiveness across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        max_similarities = [r.data["max_ft_similarity"] for r in results]
        baseline_similarities = [r.data["baseline_similarity"] for r in results]
        
        x = np.array(layers)
        width = 0.35
        
        bars1 = ax.bar(x - width/2, baseline_similarities, width, 
                       label='Baseline', color='lightblue', alpha=0.7)
        bars2 = ax.bar(x + width/2, max_similarities, width,
                       label='With Steering', color='darkblue', alpha=0.7)
        
        # Highlight effective layers
        for i, (layer, result) in enumerate(zip(layers, results)):
            if result.metadata["effective"]:
                ax.plot(layer, max_similarities[i], 'g*', markersize=15)
                
        ax.set_xlabel('Layer')
        ax.set_ylabel('Similarity to Fine-tuned Model')
        ax.set_title('Steering Effectiveness by Layer')
        ax.set_xticks(layers)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_intensity_response(self, results: List[DiffResult]) -> plt.Figure:
        """Plot response to different steering intensities."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Select up to 4 layers to visualize
        selected_results = results[:4]
        
        for idx, result in enumerate(selected_results):
            ax = axes[idx]
            
            effects = result.data["steering_effects"]
            intensities = [e["intensity"] for e in effects]
            ft_similarities = [e["ft_similarity"] for e in effects]
            behavior_shifts = [e["behavior_shift"] for e in effects]
            
            # Plot similarity to FT
            ax.plot(intensities, ft_similarities, 'o-', label='FT Similarity', linewidth=2)
            
            # Plot behavior shift on secondary axis
            ax2 = ax.twinx()
            ax2.plot(intensities, behavior_shifts, 's--', color='red', 
                    label='Behavior Shift', linewidth=2)
            ax2.set_ylabel('Behavior Shift (KL)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            # Mark optimal intensity
            optimal = result.data["optimal_intensity"]
            ax.axvline(optimal, color='green', linestyle=':', alpha=0.5)
            ax.text(optimal, 0.5, f'Optimal: {optimal}', 
                   transform=ax.get_xaxis_transform(), ha='center')
            
            ax.set_xlabel('Steering Intensity')
            ax.set_ylabel('FT Similarity', color='blue')
            ax.set_title(f'Layer {result.layer} Intensity Response')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig
    
    def _plot_vector_magnitudes(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze steering vector properties."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        layers = [r.layer for r in results]
        vector_norms = [r.data["vector_norm"] for r in results]
        effectiveness = [r.data["max_ft_similarity"] for r in results]
        
        # Plot 1: Vector norms by layer
        ax1.plot(layers, vector_norms, 'o-', linewidth=2, markersize=8)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Steering Vector Norm')
        ax1.set_title('Steering Vector Magnitude by Layer')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effectiveness vs norm
        scatter = ax2.scatter(vector_norms, effectiveness, 
                            c=layers, cmap='viridis', s=100, alpha=0.7)
        ax2.set_xlabel('Vector Norm')
        ax2.set_ylabel('Steering Effectiveness')
        ax2.set_title('Vector Magnitude vs Effectiveness')
        
        # Add correlation line
        if len(vector_norms) > 2:
            z = np.polyfit(vector_norms, effectiveness, 1)
            p = np.poly1d(z)
            ax2.plot(sorted(vector_norms), p(sorted(vector_norms)), 
                    "r--", alpha=0.5, label=f'Linear fit')
            ax2.legend()
            
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Layer')
        
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


class ReadableTraceAnalyzer(DiffingMethod):
    """Analyze persistent 'readable traces' in activation differences."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.trace_consistency = {}
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Analyze readable traces across diverse inputs."""
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Collect traces across multiple diverse inputs
        num_layers = len(manager.base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        # Initialize trace storage
        layer_traces = {layer: [] for layer in layers_to_analyze}
        
        # Process batches of inputs
        num_batches = min(10, len(inputs) // self.config.batch_size)
        
        for batch_idx in range(num_batches):
            batch_start = batch_idx * self.config.batch_size
            batch_end = min(batch_start + self.config.batch_size, len(inputs))
            batch_inputs = inputs[batch_start:batch_end]
            
            for layer_idx in layers_to_analyze:
                # Get activation difference
                diff = manager.get_activation_diff(batch_inputs, layer_idx)
                layer_traces[layer_idx].append(diff)
                
        # Analyze trace consistency
        for layer_idx, traces in layer_traces.items():
            if traces:
                result = self._analyze_trace_consistency(layer_idx, traces)
                results.append(result)
                
        return results
    
    def _analyze_trace_consistency(self, layer: int, 
                                  traces: List[torch.Tensor]) -> DiffResult:
        """Analyze consistency of traces across different inputs."""
        # Stack traces
        stacked_traces = torch.stack(traces)
        
        # Compute mean trace
        mean_trace = stacked_traces.mean(dim=0)
        
        # Compute consistency metrics
        std_trace = stacked_traces.std(dim=0)
        consistency_score = 1.0 / (1.0 + std_trace.mean().item())
        
        # Find most consistent dimensions
        dim_consistency = 1.0 / (1.0 + std_trace.flatten())
        top_k = min(100, dim_consistency.shape[0])
        consistent_dims, consistent_indices = torch.topk(dim_consistency, top_k)
        
        # Compute directional consistency (cosine similarity between traces)
        trace_pairs = []
        for i in range(min(5, len(traces))):
            for j in range(i + 1, min(5, len(traces))):
                cos_sim = F.cosine_similarity(
                    traces[i].flatten(), traces[j].flatten(), dim=0
                ).item()
                trace_pairs.append(cos_sim)
                
        avg_cosine_sim = np.mean(trace_pairs) if trace_pairs else 0
        
        # Check if trace encodes semantic direction
        trace_magnitude = mean_trace.norm().item()
        relative_std = (std_trace.mean() / (mean_trace.abs().mean() + 1e-8)).item()
        
        data = {
            "consistency_score": consistency_score,
            "trace_magnitude": trace_magnitude,
            "relative_std": relative_std,
            "avg_cosine_similarity": avg_cosine_sim,
            "num_samples": len(traces),
            "top_consistent_dims": consistent_indices.cpu().numpy().tolist()[:10],
            "semantic_encoding": consistency_score > 0.7 and avg_cosine_sim > 0.5,
        }
        
        # Store for visualization
        self.trace_consistency[layer] = {
            "mean_trace": mean_trace,
            "std_trace": std_trace,
            "consistency_score": consistency_score,
        }
        
        return DiffResult(
            method="readable_trace",
            layer=layer,
            magnitude=trace_magnitude,
            direction=mean_trace.mean(dim=(0, 1)),  # Average direction
            data=data,
            metadata={
                "readable": data["semantic_encoding"],
                "trace_shape": list(mean_trace.shape),
            }
        )
    
    def extract_semantic_directions(self) -> Dict[int, torch.Tensor]:
        """Extract semantic directions from readable traces."""
        semantic_dirs = {}
        
        for layer, trace_data in self.trace_consistency.items():
            if trace_data["consistency_score"] > 0.7:
                # Normalize the mean trace as semantic direction
                mean_trace = trace_data["mean_trace"]
                semantic_dir = mean_trace / (mean_trace.norm() + 1e-8)
                semantic_dirs[layer] = semantic_dir
                
        return semantic_dirs
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize readable trace analysis."""
        figures = {}
        
        # 1. Consistency scores across layers
        fig_consistency = self._plot_consistency_scores(results)
        figures["trace_consistency"] = fig_consistency
        
        # 2. Semantic encoding analysis
        fig_semantic = self._plot_semantic_encoding(results)
        figures["semantic_encoding"] = fig_semantic
        
        # 3. Trace stability
        fig_stability = self._plot_trace_stability()
        figures["trace_stability"] = fig_stability
        
        return figures
    
    def _plot_consistency_scores(self, results: List[DiffResult]) -> plt.Figure:
        """Plot trace consistency across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        consistency = [r.data["consistency_score"] for r in results]
        semantic = [r.metadata["readable"] for r in results]
        
        bars = ax.bar(layers, consistency, color='lightblue', edgecolor='navy', alpha=0.7)
        
        # Highlight semantically readable layers
        for bar, is_semantic in zip(bars, semantic):
            if is_semantic:
                bar.set_color('green')
                
        ax.set_xlabel('Layer')
        ax.set_ylabel('Consistency Score')
        ax.set_title('Readable Trace Consistency Across Layers')
        ax.axhline(0.7, color='red', linestyle='--', alpha=0.5, 
                  label='Semantic threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_semantic_encoding(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze semantic encoding properties."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract data
        layers = [r.layer for r in results]
        magnitudes = [r.data["trace_magnitude"] for r in results]
        cosine_sims = [r.data["avg_cosine_similarity"] for r in results]
        readable = [r.metadata["readable"] for r in results]
        
        # Plot 1: Magnitude vs consistency
        consistency = [r.data["consistency_score"] for r in results]
        scatter1 = ax1.scatter(magnitudes, consistency, 
                             c=['green' if r else 'red' for r in readable],
                             s=100, alpha=0.7, edgecolors='black')
        ax1.set_xlabel('Trace Magnitude')
        ax1.set_ylabel('Consistency Score')
        ax1.set_title('Trace Properties: Magnitude vs Consistency')
        ax1.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Semantically Readable'),
            Patch(facecolor='red', label='Not Readable')
        ]
        ax1.legend(handles=legend_elements)
        
        # Plot 2: Layer vs cosine similarity
        ax2.plot(layers, cosine_sims, 'o-', linewidth=2, markersize=8)
        ax2.axhline(0.5, color='red', linestyle='--', alpha=0.5,
                   label='Readability threshold')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('Average Cosine Similarity')
        ax2.set_title('Directional Consistency Across Layers')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
        
        plt.tight_layout()
        return fig
    
    def _plot_trace_stability(self) -> plt.Figure:
        """Visualize trace stability patterns."""
        if not self.trace_consistency:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'No trace data available', 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create stability matrix
        layers = sorted(self.trace_consistency.keys())
        num_dims = min(50, self.trace_consistency[layers[0]]["mean_trace"].flatten().shape[0])
        
        stability_matrix = np.zeros((len(layers), num_dims))
        
        for i, layer in enumerate(layers):
            mean_trace = self.trace_consistency[layer]["mean_trace"].flatten()[:num_dims]
            std_trace = self.trace_consistency[layer]["std_trace"].flatten()[:num_dims]
            
            # Stability = mean / (std + eps)
            stability = mean_trace.abs() / (std_trace + 1e-8)
            stability_matrix[i, :] = stability.cpu().numpy()
            
        # Plot heatmap
        im = ax.imshow(stability_matrix, aspect='auto', cmap='hot', 
                      interpolation='nearest')
        ax.set_xlabel('Dimension (subset)')
        ax.set_ylabel('Layer')
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels(layers)
        ax.set_title('Trace Stability Heatmap (Mean/Std Ratio)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Stability')
        
        plt.tight_layout()
        return fig