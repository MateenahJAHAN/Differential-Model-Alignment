"""
Tier II Analysis: Activation dynamics and differential analysis.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from tqdm import tqdm
from dataclasses import dataclass

from ..core.base import DiffingMethod, DiffResult, ActivationStore, Hook
from ..core.config import DiffConfig
from ..core.model_loader import DualModelManager, HuggingFaceModelWrapper

logger = logging.getLogger(__name__)


@dataclass
class ActivationTrace:
    """Container for activation traces across layers."""
    layer: int
    base_activation: torch.Tensor
    ft_activation: torch.Tensor
    diff_activation: torch.Tensor
    metadata: Dict[str, Any] = None


class ActivationDifferenceAnalyzer(DiffingMethod):
    """Analyze activation differences between models."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.activation_store = ActivationStore(max_size=10000)
        self.traces = []
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute activation differences across layers."""
        if inputs is None:
            raise ValueError("Inputs required for activation analysis")
            
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Analyze each layer
        num_layers = len(manager.base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            # Get activation differences
            diff_activation = manager.get_activation_diff(inputs, layer_idx)
            base_activation = manager.get_base_activations(inputs, layer_idx)
            ft_activation = manager.get_ft_activations(inputs, layer_idx)
            
            # Store trace
            trace = ActivationTrace(
                layer=layer_idx,
                base_activation=base_activation,
                ft_activation=ft_activation,
                diff_activation=diff_activation,
                metadata={"input_shape": list(inputs.shape)}
            )
            self.traces.append(trace)
            
            # Analyze the difference
            result = self._analyze_activation_diff(trace)
            results.append(result)
            
        return results
    
    def _analyze_activation_diff(self, trace: ActivationTrace) -> DiffResult:
        """Analyze a single activation difference."""
        diff = trace.diff_activation
        
        # Compute metrics
        l2_norm = torch.norm(diff, p=2).item()
        cosine_sim = F.cosine_similarity(
            trace.base_activation.flatten(),
            trace.ft_activation.flatten(),
            dim=0
        ).item()
        
        # Find most changed positions
        diff_flat = diff.abs().flatten()
        top_k = min(100, diff_flat.numel())
        top_values, top_indices = torch.topk(diff_flat, top_k)
        
        # Directional analysis
        mean_direction = diff.mean(dim=0)  # Average across batch/sequence
        
        data = {
            "l2_norm": l2_norm,
            "cosine_similarity": cosine_sim,
            "mean_abs_diff": diff.abs().mean().item(),
            "max_abs_diff": diff.abs().max().item(),
            "std_diff": diff.std().item(),
            "top_changed_positions": top_indices.cpu().numpy().tolist(),
            "top_change_magnitudes": top_values.cpu().numpy().tolist(),
            "activation_magnitude_ratio": (
                trace.ft_activation.norm() / trace.base_activation.norm()
            ).item(),
        }
        
        return DiffResult(
            method="activation_diff",
            layer=trace.layer,
            magnitude=l2_norm,
            direction=mean_direction,
            data=data,
            metadata={
                "shape": list(diff.shape),
                "device": str(diff.device),
            }
        )
    
    def get_readable_traces(self, num_samples: int = 10) -> Dict[int, torch.Tensor]:
        """Extract 'readable traces' - consistent activation differences."""
        readable_traces = {}
        
        for layer_idx in set(t.layer for t in self.traces):
            layer_traces = [t for t in self.traces if t.layer == layer_idx]
            
            if len(layer_traces) >= num_samples:
                # Stack differences and compute mean
                diffs = torch.stack([t.diff_activation for t in layer_traces[:num_samples]])
                mean_diff = diffs.mean(dim=0)
                
                # Check consistency (low variance indicates readable trace)
                std_diff = diffs.std(dim=0)
                consistency_score = 1.0 / (1.0 + std_diff.mean().item())
                
                readable_traces[layer_idx] = {
                    "mean_trace": mean_diff,
                    "consistency": consistency_score,
                    "magnitude": mean_diff.norm().item(),
                }
                
        return readable_traces
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize activation differences."""
        figures = {}
        
        # 1. Layer-wise activation difference norms
        fig_norms = self._plot_layer_norms(results)
        figures["activation_norms"] = fig_norms
        
        # 2. Cosine similarity progression
        fig_cosine = self._plot_cosine_similarity(results)
        figures["cosine_similarity"] = fig_cosine
        
        # 3. Activation difference heatmap
        if self.traces:
            fig_heatmap = self._plot_activation_heatmap()
            figures["activation_heatmap"] = fig_heatmap
            
        # 4. Readable traces
        readable = self.get_readable_traces()
        if readable:
            fig_traces = self._plot_readable_traces(readable)
            figures["readable_traces"] = fig_traces
            
        return figures
    
    def _plot_layer_norms(self, results: List[DiffResult]) -> plt.Figure:
        """Plot activation difference norms by layer."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        norms = [r.magnitude for r in results]
        
        ax.plot(layers, norms, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Activation Difference L2 Norm')
        ax.set_title('Activation Differences Across Layers')
        ax.grid(True, alpha=0.3)
        
        # Add annotations for peaks
        peak_idx = np.argmax(norms)
        ax.annotate(f'Peak: {norms[peak_idx]:.3f}',
                   xy=(layers[peak_idx], norms[peak_idx]),
                   xytext=(layers[peak_idx] + 1, norms[peak_idx] + 0.1),
                   arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        return fig
    
    def _plot_cosine_similarity(self, results: List[DiffResult]) -> plt.Figure:
        """Plot cosine similarity between base and FT activations."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        cosine_sims = [r.data["cosine_similarity"] for r in results]
        
        bars = ax.bar(layers, cosine_sims, color='skyblue', edgecolor='navy', alpha=0.7)
        
        # Color code bars
        for bar, sim in zip(bars, cosine_sims):
            if sim < 0.5:
                bar.set_color('red')
            elif sim < 0.8:
                bar.set_color('orange')
                
        ax.set_xlabel('Layer')
        ax.set_ylabel('Cosine Similarity')
        ax.set_title('Activation Similarity Between Base and Fine-tuned Models')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Low similarity')
        ax.axhline(0.8, color='green', linestyle='--', alpha=0.5, label='High similarity')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def _plot_activation_heatmap(self) -> plt.Figure:
        """Create heatmap of activation differences."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Select a subset of traces for visualization
        sample_traces = self.traces[:50]  # First 50 traces
        
        # Create matrix of differences (layers x positions)
        num_layers = len(set(t.layer for t in sample_traces))
        if num_layers == 0:
            return fig
            
        # Flatten and sample positions
        sample_trace = sample_traces[0]
        num_positions = min(100, sample_trace.diff_activation.flatten().shape[0])
        
        matrix = np.zeros((num_layers, num_positions))
        
        for trace in sample_traces:
            if trace.layer < num_layers:
                diff_flat = trace.diff_activation.flatten()[:num_positions]
                matrix[trace.layer, :] = diff_flat.abs().cpu().numpy()
                
        # Create heatmap
        sns.heatmap(matrix, cmap='viridis', cbar_kws={'label': 'Absolute Difference'},
                   xticklabels=False, ax=ax)
        ax.set_xlabel('Position (flattened)')
        ax.set_ylabel('Layer')
        ax.set_title('Activation Difference Heatmap')
        
        plt.tight_layout()
        return fig
    
    def _plot_readable_traces(self, readable_traces: Dict) -> plt.Figure:
        """Visualize readable traces across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = sorted(readable_traces.keys())
        magnitudes = [readable_traces[l]["magnitude"] for l in layers]
        consistencies = [readable_traces[l]["consistency"] for l in layers]
        
        # Create scatter plot with size based on consistency
        scatter = ax.scatter(layers, magnitudes, 
                           s=np.array(consistencies) * 200,
                           c=consistencies, cmap='RdYlGn',
                           alpha=0.7, edgecolors='black')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Trace Magnitude')
        ax.set_title('Readable Traces: Magnitude vs Consistency')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Consistency Score')
        
        # Add annotations
        for layer, mag, cons in zip(layers, magnitudes, consistencies):
            if cons > 0.8:  # Highly consistent
                ax.annotate(f'L{layer}', (layer, mag), 
                          xytext=(5, 5), textcoords='offset points')
                
        plt.tight_layout()
        return fig


class LogitLensAnalyzer(DiffingMethod):
    """Differential LogitLens analysis."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.vocab_size = None
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute logit differences at each layer."""
        if inputs is None:
            raise ValueError("Inputs required for LogitLens analysis")
            
        results = []
        
        # Get model manager and vocab size
        if isinstance(base_model, DualModelManager):
            manager = base_model
            self.vocab_size = manager.base_model.model.config.vocab_size
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            self.vocab_size = base_model.model.config.vocab_size
            
        # Get unembedding matrix
        lm_head = manager.base_model.model.lm_head
        
        # Analyze each layer
        num_layers = len(manager.base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            # Get activations
            base_acts = manager.get_base_activations(inputs, layer_idx)
            ft_acts = manager.get_ft_activations(inputs, layer_idx)
            
            # Project to vocabulary space
            base_logits = lm_head(base_acts)
            ft_logits = lm_head(ft_acts)
            
            # Analyze differences
            result = self._analyze_logit_diff(
                base_logits, ft_logits, layer_idx
            )
            results.append(result)
            
        return results
    
    def _analyze_logit_diff(self, base_logits: torch.Tensor,
                           ft_logits: torch.Tensor,
                           layer: int) -> DiffResult:
        """Analyze logit differences."""
        # Compute probabilities
        base_probs = F.softmax(base_logits, dim=-1)
        ft_probs = F.softmax(ft_logits, dim=-1)
        
        # KL divergence
        kl_div = F.kl_div(
            base_probs.log(), ft_probs, reduction='batchmean'
        ).item()
        
        # Top-k analysis
        k = min(10, base_logits.shape[-1])
        base_top_k = torch.topk(base_logits.mean(dim=(0, 1)), k)
        ft_top_k = torch.topk(ft_logits.mean(dim=(0, 1)), k)
        
        # Token shift analysis
        logit_diff = ft_logits - base_logits
        shifted_tokens = []
        
        # Find tokens with largest probability shifts
        prob_diff = ft_probs - base_probs
        mean_prob_diff = prob_diff.mean(dim=(0, 1))  # Average across batch and sequence
        
        top_gained = torch.topk(mean_prob_diff, k)
        top_lost = torch.topk(-mean_prob_diff, k)
        
        data = {
            "kl_divergence": kl_div,
            "base_top_tokens": base_top_k.indices.cpu().numpy().tolist(),
            "ft_top_tokens": ft_top_k.indices.cpu().numpy().tolist(),
            "tokens_gained_prob": top_gained.indices.cpu().numpy().tolist(),
            "tokens_lost_prob": top_lost.indices.cpu().numpy().tolist(),
            "max_logit_shift": logit_diff.abs().max().item(),
            "mean_logit_shift": logit_diff.abs().mean().item(),
            "entropy_base": -(base_probs * base_probs.log()).sum(dim=-1).mean().item(),
            "entropy_ft": -(ft_probs * ft_probs.log()).sum(dim=-1).mean().item(),
        }
        
        return DiffResult(
            method="logit_lens",
            layer=layer,
            magnitude=kl_div,
            data=data,
            metadata={
                "vocab_size": self.vocab_size,
                "shape": list(base_logits.shape),
            }
        )
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize logit differences."""
        figures = {}
        
        # 1. KL divergence by layer
        fig_kl = self._plot_kl_divergence(results)
        figures["kl_divergence"] = fig_kl
        
        # 2. Token shift analysis
        fig_shifts = self._plot_token_shifts(results)
        figures["token_shifts"] = fig_shifts
        
        # 3. Entropy analysis
        fig_entropy = self._plot_entropy_analysis(results)
        figures["entropy_analysis"] = fig_entropy
        
        return figures
    
    def _plot_kl_divergence(self, results: List[DiffResult]) -> plt.Figure:
        """Plot KL divergence across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        kl_values = [r.data["kl_divergence"] for r in results]
        
        ax.plot(layers, kl_values, 'o-', linewidth=2, markersize=8, color='darkred')
        ax.set_xlabel('Layer')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Logit Distribution Divergence Across Layers')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def _plot_token_shifts(self, results: List[DiffResult]) -> plt.Figure:
        """Visualize which tokens gained/lost probability."""
        # Select middle layer for visualization
        mid_idx = len(results) // 2
        result = results[mid_idx]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Tokens that gained probability
        gained = result.data["tokens_gained_prob"][:10]
        ax1.barh(range(len(gained)), [1] * len(gained), color='green', alpha=0.7)
        ax1.set_yticks(range(len(gained)))
        ax1.set_yticklabels([f"Token {t}" for t in gained])
        ax1.set_xlabel('Probability Gain')
        ax1.set_title(f'Tokens with Increased Probability (Layer {result.layer})')
        
        # Tokens that lost probability
        lost = result.data["tokens_lost_prob"][:10]
        ax2.barh(range(len(lost)), [1] * len(lost), color='red', alpha=0.7)
        ax2.set_yticks(range(len(lost)))
        ax2.set_yticklabels([f"Token {t}" for t in lost])
        ax2.set_xlabel('Probability Loss')
        ax2.set_title(f'Tokens with Decreased Probability (Layer {result.layer})')
        
        plt.tight_layout()
        return fig
    
    def _plot_entropy_analysis(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze entropy changes across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        base_entropy = [r.data["entropy_base"] for r in results]
        ft_entropy = [r.data["entropy_ft"] for r in results]
        
        ax.plot(layers, base_entropy, 'o-', label='Base Model', linewidth=2)
        ax.plot(layers, ft_entropy, 's-', label='Fine-tuned Model', linewidth=2)
        
        # Add difference subplot
        ax2 = ax.twinx()
        entropy_diff = np.array(ft_entropy) - np.array(base_entropy)
        ax2.bar(layers, entropy_diff, alpha=0.3, color='gray', label='Difference')
        ax2.set_ylabel('Entropy Difference', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        ax.set_xlabel('Layer')
        ax.set_ylabel('Entropy')
        ax.set_title('Output Entropy Across Layers')
        ax.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class CrossModelActivationPatcher(DiffingMethod):
    """Cross-Model Activation Patching (CMAP) analysis."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.patching_results = {}
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Perform CMAP analysis to test behavioral modularity."""
        if inputs is None:
            raise ValueError("Inputs required for CMAP analysis")
            
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Get reference outputs
        with torch.no_grad():
            base_output = manager.base_model.forward(inputs)
            ft_output = manager.ft_model.forward(inputs)
            
        base_logits = base_output.logits
        ft_logits = ft_output.logits
        
        # Test patching at each layer
        num_layers = len(manager.base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            # Get FT activations at this layer
            ft_acts = manager.get_ft_activations(inputs, layer_idx)
            
            # Patch into base model
            patched_output = manager.base_model.patch_activations(
                inputs, layer_idx, ft_acts
            )
            patched_logits = patched_output.logits
            
            # Analyze the effect
            result = self._analyze_patching_effect(
                base_logits, ft_logits, patched_logits, layer_idx
            )
            results.append(result)
            
            # Store detailed results
            self.patching_results[layer_idx] = {
                "base_logits": base_logits,
                "ft_logits": ft_logits,
                "patched_logits": patched_logits,
            }
            
        return results
    
    def _analyze_patching_effect(self, base_logits: torch.Tensor,
                                ft_logits: torch.Tensor,
                                patched_logits: torch.Tensor,
                                layer: int) -> DiffResult:
        """Analyze the effect of patching."""
        # Compute how much the patching recovered FT behavior
        recovery_score = self._compute_recovery_score(
            base_logits, ft_logits, patched_logits
        )
        
        # Compute behavioral shift
        base_probs = F.softmax(base_logits, dim=-1)
        patched_probs = F.softmax(patched_logits, dim=-1)
        ft_probs = F.softmax(ft_logits, dim=-1)
        
        # KL divergences
        kl_base_patched = F.kl_div(
            base_probs.log(), patched_probs, reduction='batchmean'
        ).item()
        
        kl_patched_ft = F.kl_div(
            patched_probs.log(), ft_probs, reduction='batchmean'
        ).item()
        
        # Token-level analysis
        patched_top_k = torch.topk(patched_logits.mean(dim=(0, 1)), k=5)
        
        data = {
            "recovery_score": recovery_score,
            "kl_base_to_patched": kl_base_patched,
            "kl_patched_to_ft": kl_patched_ft,
            "patched_top_tokens": patched_top_k.indices.cpu().numpy().tolist(),
            "behavior_shift": kl_base_patched / (kl_base_patched + 1e-6),
            "ft_similarity": 1.0 / (1.0 + kl_patched_ft),
        }
        
        return DiffResult(
            method="cmap",
            layer=layer,
            magnitude=recovery_score,
            data=data,
            metadata={
                "patching_direction": "ft_to_base",
                "successful": recovery_score > 0.5,
            }
        )
    
    def _compute_recovery_score(self, base_logits: torch.Tensor,
                               ft_logits: torch.Tensor,
                               patched_logits: torch.Tensor) -> float:
        """Compute how well patching recovered FT behavior."""
        # Normalize differences
        total_diff = (ft_logits - base_logits).norm()
        recovered_diff = (patched_logits - base_logits).norm()
        
        if total_diff > 0:
            # How much of the difference was recovered
            recovery = recovered_diff / total_diff
            # Bound between 0 and 1
            recovery_score = min(1.0, recovery.item())
        else:
            recovery_score = 0.0
            
        return recovery_score
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize CMAP results."""
        figures = {}
        
        # 1. Recovery scores by layer
        fig_recovery = self._plot_recovery_scores(results)
        figures["recovery_scores"] = fig_recovery
        
        # 2. Behavior shift analysis
        fig_shift = self._plot_behavior_shifts(results)
        figures["behavior_shifts"] = fig_shift
        
        # 3. Critical layer identification
        fig_critical = self._identify_critical_layers(results)
        figures["critical_layers"] = fig_critical
        
        return figures
    
    def _plot_recovery_scores(self, results: List[DiffResult]) -> plt.Figure:
        """Plot recovery scores across layers."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        scores = [r.data["recovery_score"] for r in results]
        
        bars = ax.bar(layers, scores, color='steelblue', edgecolor='black', alpha=0.7)
        
        # Highlight successful patches
        for bar, score, result in zip(bars, scores, results):
            if result.metadata["successful"]:
                bar.set_color('green')
                
        ax.set_xlabel('Layer')
        ax.set_ylabel('Recovery Score')
        ax.set_title('CMAP: Fine-tuned Behavior Recovery by Layer')
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='Success threshold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_behavior_shifts(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze behavioral shifts from patching."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        layers = [r.layer for r in results]
        kl_base_patched = [r.data["kl_base_to_patched"] for r in results]
        kl_patched_ft = [r.data["kl_patched_to_ft"] for r in results]
        
        # Plot 1: KL from base
        ax1.plot(layers, kl_base_patched, 'o-', linewidth=2, color='darkblue')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('KL(Base || Patched)')
        ax1.set_title('Behavioral Shift from Base Model')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: KL to FT
        ax2.plot(layers, kl_patched_ft, 's-', linewidth=2, color='darkgreen')
        ax2.set_xlabel('Layer')
        ax2.set_ylabel('KL(Patched || FT)')
        ax2.set_title('Remaining Distance to Fine-tuned Model')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def _identify_critical_layers(self, results: List[DiffResult]) -> plt.Figure:
        """Identify critical layers for the fine-tuned behavior."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        layers = [r.layer for r in results]
        recovery = [r.data["recovery_score"] for r in results]
        ft_similarity = [r.data["ft_similarity"] for r in results]
        
        # Create a combined criticality score
        criticality = np.array(recovery) * np.array(ft_similarity)
        
        # Create scatter plot
        scatter = ax.scatter(layers, criticality, 
                           s=np.array(recovery) * 200,
                           c=ft_similarity, cmap='RdYlGn',
                           alpha=0.7, edgecolors='black')
        
        # Highlight critical layers
        critical_threshold = np.percentile(criticality, 75)
        for layer, crit in zip(layers, criticality):
            if crit > critical_threshold:
                ax.annotate(f'Layer {layer}', (layer, crit),
                          xytext=(5, 5), textcoords='offset points',
                          bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.5))
                
        ax.set_xlabel('Layer')
        ax.set_ylabel('Criticality Score')
        ax.set_title('Critical Layers for Fine-tuned Behavior')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('FT Similarity')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig


class ModelStitcher(DiffingMethod):
    """Model stitching analysis for behavioral localization."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.stitching_grid = None
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Perform model stitching analysis."""
        if inputs is None:
            raise ValueError("Inputs required for model stitching")
            
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Get reference outputs
        with torch.no_grad():
            base_output = manager.base_model.forward(inputs)
            ft_output = manager.ft_model.forward(inputs)
            
        base_loss = self._compute_task_loss(base_output.logits, inputs)
        ft_loss = self._compute_task_loss(ft_output.logits, inputs)
        
        # Create stitching grid
        num_layers = len(manager.base_model.model.transformer.h)
        self.stitching_grid = np.zeros((num_layers, num_layers))
        
        # Test different stitch points
        stitch_points = list(range(0, num_layers, max(1, num_layers // 10)))
        
        for stitch_point in stitch_points:
            # Create stitched model: base[0:stitch_point] + ft[stitch_point:]
            stitched_output = self._run_stitched_model(
                manager, inputs, stitch_point
            )
            
            stitched_loss = self._compute_task_loss(stitched_output, inputs)
            
            # Analyze the stitching effect
            result = self._analyze_stitching(
                base_loss, ft_loss, stitched_loss, stitch_point
            )
            results.append(result)
            
        return results
    
    def _run_stitched_model(self, manager: DualModelManager,
                           inputs: torch.Tensor,
                           stitch_point: int) -> torch.Tensor:
        """Run a stitched model configuration."""
        # This is a simplified implementation
        # In practice, you'd need to modify the forward pass
        
        # Get activations at stitch point from base model
        if stitch_point == 0:
            # Full FT model
            output = manager.ft_model.forward(inputs)
        else:
            # Run base model up to stitch point
            base_acts = manager.get_base_activations(inputs, stitch_point - 1)
            
            # Continue with FT model from stitch point
            # This is conceptual - actual implementation would need custom forward pass
            output = manager.ft_model.forward(inputs)
            
        return output.logits
    
    def _compute_task_loss(self, logits: torch.Tensor, inputs: torch.Tensor) -> float:
        """Compute task-specific loss."""
        # Simple cross-entropy loss for next token prediction
        # In practice, this would be task-specific
        if logits.shape[1] > 1:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:].contiguous()
            
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='mean'
            )
            return loss.item()
        else:
            return 0.0
    
    def _analyze_stitching(self, base_loss: float, ft_loss: float,
                          stitched_loss: float, stitch_point: int) -> DiffResult:
        """Analyze stitching results."""
        # Compute relative performance
        loss_range = abs(ft_loss - base_loss)
        if loss_range > 0:
            relative_performance = (base_loss - stitched_loss) / loss_range
        else:
            relative_performance = 0.0
            
        # Determine transition sharpness
        transition_sharpness = abs(relative_performance - 0.5) * 2
        
        data = {
            "base_loss": base_loss,
            "ft_loss": ft_loss,
            "stitched_loss": stitched_loss,
            "relative_performance": relative_performance,
            "transition_sharpness": transition_sharpness,
            "improvement_over_base": (base_loss - stitched_loss) / base_loss,
        }
        
        return DiffResult(
            method="model_stitching",
            layer=stitch_point,
            magnitude=relative_performance,
            data=data,
            metadata={
                "stitch_configuration": f"base[0:{stitch_point}] + ft[{stitch_point}:]",
            }
        )
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize model stitching results."""
        figures = {}
        
        # 1. Stitching trajectory
        fig_trajectory = self._plot_stitching_trajectory(results)
        figures["stitching_trajectory"] = fig_trajectory
        
        # 2. Performance landscape
        fig_landscape = self._plot_performance_landscape(results)
        figures["performance_landscape"] = fig_landscape
        
        return figures
    
    def _plot_stitching_trajectory(self, results: List[DiffResult]) -> plt.Figure:
        """Plot performance as a function of stitch point."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stitch_points = [r.layer for r in results]
        relative_perf = [r.data["relative_performance"] for r in results]
        
        ax.plot(stitch_points, relative_perf, 'o-', linewidth=2, markersize=8)
        
        # Add reference lines
        ax.axhline(0, color='blue', linestyle='--', alpha=0.5, label='Base performance')
        ax.axhline(1, color='green', linestyle='--', alpha=0.5, label='FT performance')
        ax.axhline(0.5, color='red', linestyle=':', alpha=0.5, label='Midpoint')
        
        # Find transition point
        transition_idx = np.argmax(np.abs(np.array(relative_perf) - 0.5) < 0.1)
        if transition_idx < len(stitch_points):
            ax.axvline(stitch_points[transition_idx], color='red', alpha=0.3,
                      label=f'Transition ≈ Layer {stitch_points[transition_idx]}')
            
        ax.set_xlabel('Stitch Point (Layer)')
        ax.set_ylabel('Relative Performance')
        ax.set_title('Model Stitching: Performance Trajectory')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_performance_landscape(self, results: List[DiffResult]) -> plt.Figure:
        """Plot detailed performance metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        stitch_points = [r.layer for r in results]
        stitched_losses = [r.data["stitched_loss"] for r in results]
        improvements = [r.data["improvement_over_base"] for r in results]
        
        # Plot 1: Loss curve
        base_loss = results[0].data["base_loss"]
        ft_loss = results[0].data["ft_loss"]
        
        ax1.plot(stitch_points, stitched_losses, 'o-', linewidth=2, label='Stitched')
        ax1.axhline(base_loss, color='blue', linestyle='--', label='Base')
        ax1.axhline(ft_loss, color='green', linestyle='--', label='Fine-tuned')
        ax1.set_xlabel('Stitch Point')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss vs Stitch Point')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.invert_yaxis()  # Lower loss is better
        
        # Plot 2: Improvement over base
        bars = ax2.bar(stitch_points, np.array(improvements) * 100,
                       color='coral', edgecolor='black', alpha=0.7)
        
        # Color positive/negative differently
        for bar, imp in zip(bars, improvements):
            if imp < 0:
                bar.set_color('lightcoral')
                
        ax2.set_xlabel('Stitch Point')
        ax2.set_ylabel('Improvement over Base (%)')
        ax2.set_title('Performance Improvement by Stitch Point')
        ax2.axhline(0, color='black', linewidth=0.5)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig