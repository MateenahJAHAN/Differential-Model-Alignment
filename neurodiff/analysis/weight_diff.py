"""
Tier I Analysis: Weight space differential analysis.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from scipy.linalg import svd
from sklearn.decomposition import PCA

from ..core.base import DiffingMethod, DiffResult, MetricTracker
from ..core.config import DiffConfig
from ..core.model_loader import DualModelManager

logger = logging.getLogger(__name__)


class WeightDifferenceAnalyzer(DiffingMethod):
    """Analyze weight differences between base and fine-tuned models."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.metric_tracker = MetricTracker()
        
    def compute_diff(self, base_model, finetuned_model, 
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute weight differences across all layers and components."""
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Analyze each layer
        num_layers = len(base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            for component in self.config.components_to_analyze:
                try:
                    # Get weight differences
                    weight_diffs = manager.get_weight_diff(layer_idx, component)
                    
                    # Compute metrics for each weight matrix
                    for weight_name, diff_tensor in weight_diffs.items():
                        result = self._analyze_weight_diff(
                            diff_tensor, layer_idx, component, weight_name
                        )
                        results.append(result)
                        
                        # Track metrics
                        self.metric_tracker.update(
                            f"weight_diff_norm_l{layer_idx}_{component}_{weight_name}",
                            result.magnitude
                        )
                        
                except Exception as e:
                    logger.warning(f"Error analyzing layer {layer_idx} {component}: {e}")
                    
        return results
    
    def _analyze_weight_diff(self, diff_tensor: torch.Tensor, 
                           layer: int, component: str, 
                           weight_name: str) -> DiffResult:
        """Analyze a single weight difference tensor."""
        # Compute various norms
        frobenius_norm = torch.norm(diff_tensor, p='fro').item()
        spectral_norm = torch.norm(diff_tensor, p=2).item()  # Largest singular value
        nuclear_norm = torch.norm(diff_tensor, p='nuc').item()  # Sum of singular values
        
        # Compute sparsity
        total_params = diff_tensor.numel()
        nonzero_params = (diff_tensor.abs() > 1e-6).sum().item()
        sparsity = 1.0 - (nonzero_params / total_params)
        
        # Store detailed analysis
        data = {
            "frobenius_norm": frobenius_norm,
            "spectral_norm": spectral_norm,
            "nuclear_norm": nuclear_norm,
            "sparsity": sparsity,
            "shape": list(diff_tensor.shape),
            "nonzero_count": nonzero_params,
            "mean_abs_diff": diff_tensor.abs().mean().item(),
            "max_abs_diff": diff_tensor.abs().max().item(),
            "std_diff": diff_tensor.std().item(),
        }
        
        return DiffResult(
            method="weight_diff",
            layer=layer,
            component=f"{component}.{weight_name}",
            magnitude=frobenius_norm,
            direction=diff_tensor,
            data=data,
            metadata={
                "component_type": component,
                "weight_name": weight_name,
                "layer_depth": layer,
            }
        )
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Create visualizations for weight differences."""
        figures = {}
        
        # 1. Layer-wise heatmap
        fig_heatmap = self._create_layer_heatmap(results)
        figures["layer_heatmap"] = fig_heatmap
        
        # 2. Component comparison
        fig_components = self._create_component_comparison(results)
        figures["component_comparison"] = fig_components
        
        # 3. Sparsity analysis
        fig_sparsity = self._create_sparsity_plot(results)
        figures["sparsity_analysis"] = fig_sparsity
        
        # 4. Norm distribution
        fig_norms = self._create_norm_distribution(results)
        figures["norm_distribution"] = fig_norms
        
        if self.config.save_visualizations:
            self._save_figures(figures)
            
        return figures
    
    def _create_layer_heatmap(self, results: List[DiffResult]) -> plt.Figure:
        """Create a heatmap of weight differences across layers."""
        # Organize data by layer and component
        layer_data = {}
        for result in results:
            if result.layer not in layer_data:
                layer_data[result.layer] = {}
            layer_data[result.layer][result.component] = result.magnitude
            
        # Create matrix for heatmap
        layers = sorted(layer_data.keys())
        components = sorted(set(comp for layer_dict in layer_data.values() 
                              for comp in layer_dict.keys()))
        
        matrix = np.zeros((len(layers), len(components)))
        for i, layer in enumerate(layers):
            for j, comp in enumerate(components):
                matrix[i, j] = layer_data.get(layer, {}).get(comp, 0)
                
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(matrix, 
                   xticklabels=components,
                   yticklabels=layers,
                   cmap='viridis',
                   cbar_kws={'label': 'Weight Difference Norm'},
                   ax=ax)
        
        ax.set_title('Weight Differences Across Layers and Components')
        ax.set_xlabel('Component')
        ax.set_ylabel('Layer')
        
        plt.tight_layout()
        return fig
    
    def _create_component_comparison(self, results: List[DiffResult]) -> plt.Figure:
        """Compare weight differences across different components."""
        # Group by component type
        component_data = {}
        for result in results:
            comp_type = result.metadata.get("component_type", "unknown")
            if comp_type not in component_data:
                component_data[comp_type] = []
            component_data[comp_type].append(result.magnitude)
            
        # Create box plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        data_list = []
        labels = []
        for comp, values in component_data.items():
            data_list.append(values)
            labels.append(f"{comp}\n(n={len(values)})")
            
        box_plot = ax.boxplot(data_list, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = plt.cm.Set3(np.linspace(0, 1, len(box_plot['boxes'])))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            
        ax.set_title('Weight Difference Distribution by Component Type')
        ax.set_ylabel('Frobenius Norm')
        ax.set_xlabel('Component Type')
        ax.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def _create_sparsity_plot(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze sparsity patterns in weight differences."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Extract sparsity data
        layers = []
        sparsity_values = []
        magnitudes = []
        
        for result in results:
            layers.append(result.layer)
            sparsity_values.append(result.data["sparsity"])
            magnitudes.append(result.magnitude)
            
        # Plot 1: Sparsity by layer
        layer_sparsity = {}
        for layer, sparsity in zip(layers, sparsity_values):
            if layer not in layer_sparsity:
                layer_sparsity[layer] = []
            layer_sparsity[layer].append(sparsity)
            
        avg_sparsity = [np.mean(layer_sparsity[l]) for l in sorted(layer_sparsity.keys())]
        ax1.plot(sorted(layer_sparsity.keys()), avg_sparsity, 'o-', linewidth=2)
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Average Sparsity')
        ax1.set_title('Weight Difference Sparsity by Layer')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Sparsity vs Magnitude
        scatter = ax2.scatter(sparsity_values, magnitudes, alpha=0.6, c=layers, cmap='viridis')
        ax2.set_xlabel('Sparsity')
        ax2.set_ylabel('Magnitude (Frobenius Norm)')
        ax2.set_title('Sparsity vs Magnitude of Weight Differences')
        ax2.set_yscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Layer')
        
        plt.tight_layout()
        return fig
    
    def _create_norm_distribution(self, results: List[DiffResult]) -> plt.Figure:
        """Plot distribution of different norms."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()
        
        # Extract norm data
        norm_types = ["frobenius_norm", "spectral_norm", "nuclear_norm", "mean_abs_diff"]
        
        for idx, norm_type in enumerate(norm_types):
            values = [result.data[norm_type] for result in results]
            
            ax = axes[idx]
            ax.hist(values, bins=30, alpha=0.7, edgecolor='black')
            ax.set_xlabel(norm_type.replace('_', ' ').title())
            ax.set_ylabel('Count')
            ax.set_title(f'Distribution of {norm_type.replace("_", " ").title()}')
            
            # Add statistics
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Mean: {mean_val:.3f}')
            ax.axvline(median_val, color='green', linestyle='--', label=f'Median: {median_val:.3f}')
            ax.legend()
            
        plt.tight_layout()
        return fig
    
    def _save_figures(self, figures: Dict[str, plt.Figure]):
        """Save figures to disk."""
        output_dir = self.config.output_dir / "weight_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, fig in figures.items():
            path = output_dir / f"{name}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure to {path}")


class TaskArithmeticAnalyzer(DiffingMethod):
    """Analyze task vectors and perform task arithmetic operations."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.task_vector = None
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Extract task vector (τ = W_ft - W_base)."""
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Extract full task vector
        self.task_vector = {}
        
        num_layers = len(base_model.model.transformer.h)
        for layer_idx in range(num_layers):
            for component in ["attention", "mlp"]:
                try:
                    weight_diffs = manager.get_weight_diff(layer_idx, component)
                    
                    for weight_name, diff_tensor in weight_diffs.items():
                        key = f"layer_{layer_idx}.{component}.{weight_name}"
                        self.task_vector[key] = diff_tensor.detach().cpu()
                        
                        # Create result
                        result = DiffResult(
                            method="task_vector",
                            layer=layer_idx,
                            component=f"{component}.{weight_name}",
                            magnitude=torch.norm(diff_tensor, p='fro').item(),
                            direction=diff_tensor,
                            data={"key": key, "shape": list(diff_tensor.shape)},
                            metadata={"task_vector_component": key}
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error extracting task vector at layer {layer_idx}: {e}")
                    
        logger.info(f"Extracted task vector with {len(self.task_vector)} components")
        return results
    
    def scale_task_vector(self, scaling_factor: float) -> Dict[str, torch.Tensor]:
        """Scale the task vector by a factor (λ * τ)."""
        if self.task_vector is None:
            raise ValueError("Task vector not computed yet")
            
        scaled_vector = {}
        for key, tensor in self.task_vector.items():
            scaled_vector[key] = scaling_factor * tensor
            
        return scaled_vector
    
    def add_task_vectors(self, other_task_vector: Dict[str, torch.Tensor],
                        weights: Tuple[float, float] = (1.0, 1.0)) -> Dict[str, torch.Tensor]:
        """Add two task vectors with optional weighting."""
        if self.task_vector is None:
            raise ValueError("Task vector not computed yet")
            
        combined_vector = {}
        for key in self.task_vector.keys():
            if key in other_task_vector:
                combined_vector[key] = (weights[0] * self.task_vector[key] + 
                                      weights[1] * other_task_vector[key])
            else:
                combined_vector[key] = weights[0] * self.task_vector[key]
                
        return combined_vector
    
    def compute_orthogonality(self, other_task_vector: Dict[str, torch.Tensor]) -> float:
        """Compute orthogonality between task vectors."""
        if self.task_vector is None:
            raise ValueError("Task vector not computed yet")
            
        dot_products = []
        magnitudes1 = []
        magnitudes2 = []
        
        for key in self.task_vector.keys():
            if key in other_task_vector:
                v1 = self.task_vector[key].flatten()
                v2 = other_task_vector[key].flatten()
                
                dot_products.append(torch.dot(v1, v2).item())
                magnitudes1.append(torch.norm(v1).item())
                magnitudes2.append(torch.norm(v2).item())
                
        # Compute average cosine similarity
        cosine_sims = []
        for dp, m1, m2 in zip(dot_products, magnitudes1, magnitudes2):
            if m1 > 0 and m2 > 0:
                cosine_sims.append(dp / (m1 * m2))
                
        avg_cosine_sim = np.mean(cosine_sims) if cosine_sims else 0
        orthogonality = 1.0 - abs(avg_cosine_sim)
        
        return orthogonality
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize task vector properties."""
        figures = {}
        
        # 1. Task vector magnitude distribution
        fig_magnitude = self._plot_magnitude_distribution(results)
        figures["task_vector_magnitudes"] = fig_magnitude
        
        # 2. Layer-wise contribution
        fig_contribution = self._plot_layer_contribution(results)
        figures["layer_contribution"] = fig_contribution
        
        return figures
    
    def _plot_magnitude_distribution(self, results: List[DiffResult]) -> plt.Figure:
        """Plot distribution of task vector component magnitudes."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        magnitudes = [r.magnitude for r in results]
        layers = [r.layer for r in results]
        
        # Create violin plot
        layer_magnitudes = {}
        for layer, mag in zip(layers, magnitudes):
            if layer not in layer_magnitudes:
                layer_magnitudes[layer] = []
            layer_magnitudes[layer].append(mag)
            
        positions = sorted(layer_magnitudes.keys())
        data = [layer_magnitudes[p] for p in positions]
        
        parts = ax.violinplot(data, positions=positions, showmeans=True)
        
        # Customize colors
        for pc in parts['bodies']:
            pc.set_facecolor('skyblue')
            pc.set_alpha(0.7)
            
        ax.set_xlabel('Layer')
        ax.set_ylabel('Component Magnitude')
        ax.set_title('Task Vector Component Magnitudes by Layer')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_layer_contribution(self, results: List[DiffResult]) -> plt.Figure:
        """Plot relative contribution of each layer to the task vector."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute total magnitude per layer
        layer_magnitudes = {}
        for result in results:
            layer = result.layer
            if layer not in layer_magnitudes:
                layer_magnitudes[layer] = 0
            layer_magnitudes[layer] += result.magnitude ** 2  # L2 norm squared
            
        # Convert to percentages
        total_magnitude = sum(layer_magnitudes.values())
        layer_percentages = {l: (m/total_magnitude)*100 for l, m in layer_magnitudes.items()}
        
        layers = sorted(layer_percentages.keys())
        percentages = [layer_percentages[l] for l in layers]
        
        # Create bar plot
        bars = ax.bar(layers, percentages, color='coral', alpha=0.7, edgecolor='black')
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{pct:.1f}%', ha='center', va='bottom')
            
        ax.set_xlabel('Layer')
        ax.set_ylabel('Contribution (%)')
        ax.set_title('Layer Contribution to Task Vector')
        ax.set_ylim(0, max(percentages) * 1.1)
        
        plt.tight_layout()
        return fig


class SpectralAnalyzer(DiffingMethod):
    """Perform spectral analysis on weight differences."""
    
    def __init__(self, config: DiffConfig):
        super().__init__(config)
        self.singular_values = {}
        
    def compute_diff(self, base_model, finetuned_model,
                    inputs: Optional[torch.Tensor] = None) -> List[DiffResult]:
        """Compute spectral decomposition of weight differences."""
        results = []
        
        # Get model manager
        if isinstance(base_model, DualModelManager):
            manager = base_model
        else:
            manager = DualModelManager(base_model.config, finetuned_model.config)
            manager.base_model = base_model
            manager.ft_model = finetuned_model
            
        # Analyze each layer
        num_layers = len(base_model.model.transformer.h)
        layers_to_analyze = self.config.layers_to_analyze or list(range(num_layers))
        
        for layer_idx in layers_to_analyze:
            for component in self.config.components_to_analyze:
                try:
                    weight_diffs = manager.get_weight_diff(layer_idx, component)
                    
                    for weight_name, diff_tensor in weight_diffs.items():
                        result = self._analyze_spectrum(
                            diff_tensor, layer_idx, component, weight_name
                        )
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error in spectral analysis at layer {layer_idx}: {e}")
                    
        return results
    
    def _analyze_spectrum(self, diff_tensor: torch.Tensor,
                         layer: int, component: str,
                         weight_name: str) -> DiffResult:
        """Perform spectral analysis on a weight difference tensor."""
        # Move to CPU for SVD
        diff_np = diff_tensor.detach().cpu().numpy()
        
        # Reshape if needed (for multi-dimensional tensors)
        original_shape = diff_np.shape
        if len(original_shape) > 2:
            diff_np = diff_np.reshape(original_shape[0], -1)
            
        # Compute SVD
        try:
            U, s, Vt = svd(diff_np, full_matrices=False)
        except:
            # Fallback to torch SVD if scipy fails
            U, s, V = torch.svd(torch.from_numpy(diff_np))
            U = U.numpy()
            s = s.numpy()
            Vt = V.T.numpy()
            
        # Store singular values
        key = f"{layer}_{component}_{weight_name}"
        self.singular_values[key] = s
        
        # Compute metrics
        rank = np.sum(s > 1e-6)
        effective_rank = np.exp(-np.sum(s * np.log(s + 1e-10)))
        energy_concentration = np.cumsum(s**2) / np.sum(s**2)
        
        # Find how many components capture 90% of energy
        components_90 = np.argmax(energy_concentration >= 0.9) + 1
        
        data = {
            "rank": rank,
            "effective_rank": effective_rank,
            "largest_singular_value": s[0] if len(s) > 0 else 0,
            "singular_value_decay": s[:10].tolist() if len(s) >= 10 else s.tolist(),
            "components_for_90_energy": components_90,
            "total_components": len(s),
            "condition_number": s[0] / s[-1] if len(s) > 0 and s[-1] > 0 else np.inf,
        }
        
        return DiffResult(
            method="spectral_analysis",
            layer=layer,
            component=f"{component}.{weight_name}",
            magnitude=s[0],  # Largest singular value as magnitude
            direction=torch.from_numpy(U[:, 0]),  # First principal direction
            data=data,
            metadata={
                "component_type": component,
                "weight_name": weight_name,
                "original_shape": original_shape,
            }
        )
    
    def visualize(self, results: List[DiffResult]) -> Dict[str, Any]:
        """Visualize spectral analysis results."""
        figures = {}
        
        # 1. Singular value decay plots
        fig_decay = self._plot_singular_value_decay(results)
        figures["singular_value_decay"] = fig_decay
        
        # 2. Rank analysis
        fig_rank = self._plot_rank_analysis(results)
        figures["rank_analysis"] = fig_rank
        
        # 3. Energy concentration
        fig_energy = self._plot_energy_concentration()
        figures["energy_concentration"] = fig_energy
        
        return figures
    
    def _plot_singular_value_decay(self, results: List[DiffResult]) -> plt.Figure:
        """Plot singular value decay for different layers."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        # Group results by component type
        component_results = {}
        for result in results:
            comp_type = result.metadata.get("component_type", "unknown")
            if comp_type not in component_results:
                component_results[comp_type] = []
            component_results[comp_type].append(result)
            
        for idx, (comp_type, comp_results) in enumerate(component_results.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            
            # Plot decay curves for different layers
            for result in comp_results[:10]:  # Limit to 10 curves for clarity
                layer = result.layer
                decay = result.data.get("singular_value_decay", [])
                if decay:
                    ax.semilogy(decay, label=f'Layer {layer}', alpha=0.7)
                    
            ax.set_xlabel('Component Index')
            ax.set_ylabel('Singular Value')
            ax.set_title(f'Singular Value Decay - {comp_type}')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        return fig
    
    def _plot_rank_analysis(self, results: List[DiffResult]) -> plt.Figure:
        """Analyze rank properties across layers."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        layers = []
        ranks = []
        effective_ranks = []
        
        for result in results:
            layers.append(result.layer)
            ranks.append(result.data["rank"])
            effective_ranks.append(result.data["effective_rank"])
            
        # Plot 1: Rank by layer
        ax1.scatter(layers, ranks, alpha=0.6, label='Numerical Rank')
        ax1.scatter(layers, effective_ranks, alpha=0.6, label='Effective Rank')
        ax1.set_xlabel('Layer')
        ax1.set_ylabel('Rank')
        ax1.set_title('Rank Analysis by Layer')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Rank vs magnitude
        magnitudes = [r.magnitude for r in results]
        ax2.scatter(ranks, magnitudes, alpha=0.6, c=layers, cmap='viridis')
        ax2.set_xlabel('Rank')
        ax2.set_ylabel('Largest Singular Value')
        ax2.set_title('Rank vs Magnitude')
        ax2.set_yscale('log')
        
        plt.tight_layout()
        return fig
    
    def _plot_energy_concentration(self) -> plt.Figure:
        """Plot energy concentration curves."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Select a few representative singular value sets
        selected_keys = list(self.singular_values.keys())[:5]
        
        for key in selected_keys:
            s = self.singular_values[key]
            energy = np.cumsum(s**2) / np.sum(s**2)
            ax.plot(energy, label=key.replace('_', ' '), linewidth=2)
            
        # Add reference lines
        ax.axhline(0.9, color='red', linestyle='--', alpha=0.5, label='90% Energy')
        ax.axhline(0.95, color='orange', linestyle='--', alpha=0.5, label='95% Energy')
        ax.axhline(0.99, color='green', linestyle='--', alpha=0.5, label='99% Energy')
        
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Energy')
        ax.set_title('Energy Concentration in Singular Values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig