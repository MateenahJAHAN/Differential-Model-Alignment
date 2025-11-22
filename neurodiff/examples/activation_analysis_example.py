"""
Example: Comprehensive Activation Analysis and Steering Demo
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from neurodiff import (
    # Configuration
    NeuroDiffConfig,
    DiffConfig,
    ModelConfig,
    # Model management
    DualModelManager,
    # Tier II Analysis
    ActivationDifferenceAnalyzer,
    LogitLensAnalyzer,
    CrossModelActivationPatcher,
    ModelStitcher,
    ActivationSteering,
    ReadableTraceAnalyzer,
)


def prepare_sample_inputs(tokenizer, texts):
    """Prepare sample inputs for analysis."""
    inputs = []
    for text in texts:
        encoded = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs.append(encoded.input_ids)
    
    # Stack into a batch
    return torch.cat(inputs, dim=0)


def run_activation_analysis_demo():
    """Demonstrate comprehensive activation analysis capabilities."""
    
    print("=" * 60)
    print("NeuroDiff: Activation Analysis Demo")
    print("=" * 60)
    
    # Configure models
    base_config = ModelConfig(
        model_name="gpt2",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
    )
    
    # For demo purposes, using same model
    # In practice, use your fine-tuned model
    ft_config = ModelConfig(
        model_name="gpt2", 
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype="float32",
    )
    
    # Create analysis configuration
    diff_config = DiffConfig(
        base_model=base_config,
        finetuned_model=ft_config,
        layers_to_analyze=[0, 3, 6, 9, 11],  # Sample layers across depth
        components_to_analyze=["attention", "mlp"],
        batch_size=4,
        output_dir=Path("./output/activation_analysis"),
        save_visualizations=True,
    )
    
    # Load models
    print("\n1. Loading models...")
    manager = DualModelManager(base_config, ft_config)
    manager.load_models()
    
    # Get tokenizer
    tokenizer = manager.base_model.tokenizer
    
    # Prepare diverse test inputs
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can exhibit unexpected behaviors.",
        "Paris is the capital of France.",
        "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
        "The weather today is sunny with a chance of rain.",
    ]
    
    inputs = prepare_sample_inputs(tokenizer, test_texts)
    print(f"Prepared {len(test_texts)} test inputs")
    
    # ===== ACTIVATION DIFFERENCE ANALYSIS =====
    print("\n2. Analyzing activation differences...")
    act_analyzer = ActivationDifferenceAnalyzer(diff_config)
    act_results = act_analyzer.compute_diff(manager, None, inputs)
    
    print(f"✓ Computed activation differences for {len(act_results)} layer positions")
    
    # Show statistics
    for i, result in enumerate(act_results[:3]):
        print(f"\n  Layer {result.layer}:")
        print(f"    L2 norm: {result.magnitude:.4f}")
        print(f"    Cosine similarity: {result.data['cosine_similarity']:.4f}")
        print(f"    Max change: {result.data['max_abs_diff']:.4f}")
    
    # Extract readable traces
    readable_traces = act_analyzer.get_readable_traces()
    if readable_traces:
        print(f"\n✓ Found readable traces in {len(readable_traces)} layers")
        for layer, trace_data in readable_traces.items():
            print(f"  Layer {layer}: consistency={trace_data['consistency']:.3f}, "
                  f"magnitude={trace_data['magnitude']:.3f}")
    
    # Visualize activation differences
    act_figures = act_analyzer.visualize(act_results)
    print(f"✓ Created {len(act_figures)} activation visualizations")
    
    # ===== DIFFERENTIAL LOGITLENS =====
    print("\n3. Running Differential LogitLens analysis...")
    logit_analyzer = LogitLensAnalyzer(diff_config)
    logit_results = logit_analyzer.compute_diff(manager, None, inputs)
    
    print(f"✓ Analyzed logit differences across {len(logit_results)} layers")
    
    # Show KL divergence progression
    kl_values = [r.data["kl_divergence"] for r in logit_results]
    print(f"\nKL Divergence progression:")
    for layer, kl in zip([r.layer for r in logit_results], kl_values):
        print(f"  Layer {layer}: {kl:.6f}")
    
    # Show entropy changes
    if logit_results:
        sample_result = logit_results[len(logit_results)//2]  # Middle layer
        print(f"\nEntropy at layer {sample_result.layer}:")
        print(f"  Base model: {sample_result.data['entropy_base']:.3f}")
        print(f"  Fine-tuned: {sample_result.data['entropy_ft']:.3f}")
    
    logit_figures = logit_analyzer.visualize(logit_results)
    print(f"✓ Created {len(logit_figures)} logit visualizations")
    
    # ===== CROSS-MODEL ACTIVATION PATCHING (CMAP) =====
    print("\n4. Performing Cross-Model Activation Patching...")
    cmap_analyzer = CrossModelActivationPatcher(diff_config)
    cmap_results = cmap_analyzer.compute_diff(manager, None, inputs)
    
    print(f"✓ Tested activation patching at {len(cmap_results)} layers")
    
    # Identify successful patches
    successful_patches = [r for r in cmap_results if r.metadata["successful"]]
    print(f"\nSuccessful behavior transfer at {len(successful_patches)} layers:")
    for result in successful_patches:
        print(f"  Layer {result.layer}: recovery score = {result.data['recovery_score']:.3f}")
    
    cmap_figures = cmap_analyzer.visualize(cmap_results)
    print(f"✓ Created {len(cmap_figures)} CMAP visualizations")
    
    # ===== MODEL STITCHING =====
    print("\n5. Running Model Stitching analysis...")
    stitcher = ModelStitcher(diff_config)
    stitch_results = stitcher.compute_diff(manager, None, inputs)
    
    print(f"✓ Tested {len(stitch_results)} stitching configurations")
    
    # Find transition point
    performances = [r.data["relative_performance"] for r in stitch_results]
    if performances:
        transition_idx = np.argmin(np.abs(np.array(performances) - 0.5))
        transition_layer = stitch_results[transition_idx].layer
        print(f"\nBehavior transition detected around layer {transition_layer}")
    
    stitch_figures = stitcher.visualize(stitch_results)
    print(f"✓ Created {len(stitch_figures)} stitching visualizations")
    
    # ===== ACTIVATION STEERING =====
    print("\n6. Computing steering vectors...")
    steering_analyzer = ActivationSteering(diff_config)
    steering_results = steering_analyzer.compute_diff(manager, None, inputs)
    
    print(f"✓ Computed steering vectors for {len(steering_results)} layers")
    
    # Show optimal steering intensities
    print("\nOptimal steering intensities:")
    for result in steering_results:
        if result.metadata["effective"]:
            print(f"  Layer {result.layer}: intensity = {result.data['optimal_intensity']}, "
                  f"FT similarity = {result.data['max_ft_similarity']:.3f}")
    
    # Demonstrate targeted steering
    if len(test_texts) >= 2:
        print("\nComputing targeted steering vector...")
        source_input = prepare_sample_inputs(tokenizer, [test_texts[0]])
        target_input = prepare_sample_inputs(tokenizer, [test_texts[1]])
        
        targeted_vec = steering_analyzer.compute_targeted_steering(
            manager, source_input, target_input, layer=6
        )
        print(f"✓ Created targeted steering vector at layer {targeted_vec.layer}")
    
    steering_figures = steering_analyzer.visualize(steering_results)
    print(f"✓ Created {len(steering_figures)} steering visualizations")
    
    # ===== READABLE TRACE ANALYSIS =====
    print("\n7. Analyzing readable traces...")
    trace_analyzer = ReadableTraceAnalyzer(diff_config)
    
    # Use multiple diverse inputs for trace analysis
    extended_texts = test_texts * 3  # Repeat for more samples
    extended_inputs = prepare_sample_inputs(tokenizer, extended_texts)
    
    trace_results = trace_analyzer.compute_diff(manager, None, extended_inputs)
    
    print(f"✓ Analyzed trace consistency across {len(trace_results)} layers")
    
    # Extract semantic directions
    semantic_dirs = trace_analyzer.extract_semantic_directions()
    if semantic_dirs:
        print(f"\nExtracted semantic directions from {len(semantic_dirs)} layers:")
        for layer in semantic_dirs:
            print(f"  Layer {layer}: readable trace detected")
    
    trace_figures = trace_analyzer.visualize(trace_results)
    print(f"✓ Created {len(trace_figures)} trace visualizations")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("Analysis Summary")
    print("=" * 60)
    
    total_figures = (len(act_figures) + len(logit_figures) + len(cmap_figures) +
                    len(stitch_figures) + len(steering_figures) + len(trace_figures))
    
    print(f"\n✅ Activation analysis complete!")
    print(f"   - Analyzed {len(diff_config.layers_to_analyze)} layers")
    print(f"   - Processed {len(test_texts)} diverse inputs") 
    print(f"   - Generated {total_figures} visualizations")
    print(f"   - Results saved to: {diff_config.output_dir}")
    
    return {
        "activation_results": act_results,
        "logit_results": logit_results,
        "cmap_results": cmap_results,
        "stitch_results": stitch_results,
        "steering_results": steering_results,
        "trace_results": trace_results,
    }


def demonstrate_intervention():
    """Demonstrate model intervention capabilities."""
    print("\n" + "=" * 60)
    print("Intervention Demo: Steering Model Behavior")
    print("=" * 60)
    
    # This would demonstrate:
    # 1. Extracting steering vectors from specific behaviors
    # 2. Applying them to change model outputs
    # 3. Combining multiple steering vectors
    # 4. Testing intervention robustness
    
    print("\nIntervention capabilities:")
    print("  ✓ Behavior amplification/suppression")
    print("  ✓ Task vector arithmetic") 
    print("  ✓ Semantic direction control")
    print("  ✓ Multi-layer coordination")
    

def analyze_lora_adapter():
    """Demonstrate LoRA-specific analysis."""
    print("\n" + "=" * 60)
    print("LoRA Adapter Analysis Demo")
    print("=" * 60)
    
    print("\nLoRA analysis features:")
    print("  ✓ Direct adapter weight inspection")
    print("  ✓ Memory-efficient dual-model analysis")
    print("  ✓ Rank analysis of adaptations")
    print("  ✓ Component-wise contribution metrics")
    
    # Example configuration for LoRA
    print("\nExample LoRA configuration:")
    print("""
    base_config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda",
        load_in_8bit=True,
    )
    
    ft_config = ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        device="cuda",
        load_in_8bit=True,
        is_lora=True,
        lora_path="path/to/adapter",
        lora_alpha=16.0,
    )
    """)


if __name__ == "__main__":
    # Run main activation analysis demo
    try:
        results = run_activation_analysis_demo()
        
        # Additional demonstrations
        demonstrate_intervention()
        analyze_lora_adapter()
        
    except Exception as e:
        print(f"\nError during demo: {e}")
        print("Note: This demo requires a GPU and the 'gpt2' model to be available.")
        print("For full functionality, use your own base and fine-tuned models.")