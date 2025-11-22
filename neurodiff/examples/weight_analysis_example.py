"""
Example: Weight Space Analysis Demo
"""

import torch
from pathlib import Path

from neurodiff import (
    NeuroDiffConfig,
    DiffConfig,
    ModelConfig,
    WeightDifferenceAnalyzer,
    TaskArithmeticAnalyzer,
    SpectralAnalyzer,
    DualModelManager,
)


def analyze_model_weights():
    """Demonstrate weight space analysis between base and fine-tuned models."""
    
    # Configure models
    base_config = ModelConfig(
        model_name="gpt2",  # Using small model for demo
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
    )
    
    # For demo, we'll use the same model as both base and "fine-tuned"
    # In practice, you'd specify a different fine-tuned model
    ft_config = ModelConfig(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
        # Example LoRA configuration:
        # is_lora=True,
        # lora_path="path/to/lora/adapter",
        # lora_alpha=1.0,
    )
    
    # Create differential analysis configuration
    diff_config = DiffConfig(
        base_model=base_config,
        finetuned_model=ft_config,
        layers_to_analyze=[0, 1, 2],  # Analyze first 3 layers for demo
        components_to_analyze=["attention", "mlp"],
        output_dir=Path("./output/weight_analysis"),
        save_visualizations=True,
    )
    
    # Create model manager
    print("Loading models...")
    manager = DualModelManager(base_config, ft_config)
    manager.load_models()
    
    # 1. Weight Difference Analysis
    print("\n1. Analyzing weight differences...")
    weight_analyzer = WeightDifferenceAnalyzer(diff_config)
    weight_results = weight_analyzer.compute_diff(manager, None)
    
    print(f"Found {len(weight_results)} weight difference results")
    
    # Show some statistics
    for result in weight_results[:3]:  # First 3 results
        print(f"\nLayer {result.layer}, Component: {result.component}")
        print(f"  Frobenius norm: {result.data['frobenius_norm']:.6f}")
        print(f"  Sparsity: {result.data['sparsity']:.2%}")
        print(f"  Shape: {result.data['shape']}")
    
    # Visualize
    print("\nCreating weight difference visualizations...")
    weight_figures = weight_analyzer.visualize(weight_results)
    print(f"Created {len(weight_figures)} figures")
    
    # 2. Task Arithmetic Analysis
    print("\n2. Extracting task vector...")
    task_analyzer = TaskArithmeticAnalyzer(diff_config)
    task_results = task_analyzer.compute_diff(manager, None)
    
    print(f"Extracted task vector with {len(task_results)} components")
    
    # Demonstrate task vector operations
    print("\nTask vector operations:")
    
    # Scale task vector
    scaled_vector = task_analyzer.scale_task_vector(0.5)
    print(f"  Scaled task vector by 0.5: {len(scaled_vector)} components")
    
    # Visualize task vector
    task_figures = task_analyzer.visualize(task_results)
    
    # 3. Spectral Analysis
    print("\n3. Performing spectral analysis...")
    spectral_analyzer = SpectralAnalyzer(diff_config)
    spectral_results = spectral_analyzer.compute_diff(manager, None)
    
    print(f"Completed spectral analysis for {len(spectral_results)} weight matrices")
    
    # Show spectral properties
    for result in spectral_results[:3]:
        print(f"\nLayer {result.layer}, Component: {result.component}")
        print(f"  Rank: {result.data['rank']}")
        print(f"  Effective rank: {result.data['effective_rank']:.2f}")
        print(f"  Largest singular value: {result.data['largest_singular_value']:.6f}")
        print(f"  Components for 90% energy: {result.data['components_for_90_energy']}")
    
    # Visualize spectral analysis
    spectral_figures = spectral_analyzer.visualize(spectral_results)
    
    print(f"\n✅ Analysis complete! Results saved to {diff_config.output_dir}")
    
    return weight_results, task_results, spectral_results


def demonstrate_lora_analysis():
    """Demonstrate LoRA-specific weight analysis."""
    print("\n" + "="*50)
    print("LoRA Weight Analysis Demo")
    print("="*50)
    
    # This demonstrates how to analyze LoRA adapters
    base_config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        load_in_8bit=True,  # Use 8-bit quantization for memory efficiency
    )
    
    ft_config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float16",
        load_in_8bit=True,
        is_lora=True,
        lora_path="path/to/your/lora/adapter",  # Replace with actual path
        lora_alpha=16.0,
    )
    
    print("\nLoRA configuration:")
    print(f"  Base model: {base_config.model_name}")
    print(f"  LoRA path: {ft_config.lora_path}")
    print(f"  LoRA alpha: {ft_config.lora_alpha}")
    print(f"  Quantization: 8-bit")
    
    # The analysis would proceed the same way, but the DualModelManager
    # will automatically use LoRA optimization to save memory
    print("\nWith LoRA optimization:")
    print("  - Only one copy of base weights in memory")
    print("  - Weight differences computed directly from adapter matrices")
    print("  - Memory usage reduced by ~50%")


if __name__ == "__main__":
    # Run the main analysis
    try:
        weight_results, task_results, spectral_results = analyze_model_weights()
    except Exception as e:
        print(f"Error during analysis: {e}")
        print("Make sure you have the required models available.")
    
    # Show LoRA demo (configuration only)
    demonstrate_lora_analysis()