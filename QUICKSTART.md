# NeuroDiff Quick Start Guide

## Installation

```bash
cd /workspace/neurodiff
pip install -e .
```

## Basic Usage

### 1. Analyzing Weight Differences

```python
import torch
from neurodiff import (
    ModelConfig, DiffConfig, 
    WeightDifferenceAnalyzer, DualModelManager
)

# Configure your models
base_config = ModelConfig(
    model_name="gpt2",  # or path to your base model
    device="cuda" if torch.cuda.is_available() else "cpu"
)

ft_config = ModelConfig(
    model_name="gpt2",  # replace with your fine-tuned model
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# Create analysis configuration
diff_config = DiffConfig(
    base_model=base_config,
    finetuned_model=ft_config,
    output_dir="./outputs"
)

# Load and analyze
manager = DualModelManager(base_config, ft_config)
manager.load_models()

analyzer = WeightDifferenceAnalyzer(diff_config)
results = analyzer.compute_diff(manager, None)
figures = analyzer.visualize(results)
```

### 2. Activation Analysis

```python
from neurodiff import ActivationDifferenceAnalyzer

# Prepare inputs
texts = ["Hello world", "Machine learning is fascinating"]
inputs = tokenizer(texts, return_tensors="pt", padding=True).input_ids

# Analyze activations
act_analyzer = ActivationDifferenceAnalyzer(diff_config)
act_results = act_analyzer.compute_diff(manager, None, inputs)

# Get readable traces
readable_traces = act_analyzer.get_readable_traces()
print(f"Found readable traces in {len(readable_traces)} layers")
```

### 3. Cross-Model Activation Patching (CMAP)

```python
from neurodiff import CrossModelActivationPatcher

# Test behavioral modularity
cmap = CrossModelActivationPatcher(diff_config)
cmap_results = cmap.compute_diff(manager, None, inputs)

# Find critical layers
critical_layers = [r.layer for r in cmap_results if r.metadata["successful"]]
print(f"Critical layers for behavior: {critical_layers}")
```

### 4. Activation Steering

```python
from neurodiff import ActivationSteering

# Extract steering vectors
steering = ActivationSteering(diff_config)
steering_results = steering.compute_diff(manager, None, inputs)

# Find optimal steering parameters
for result in steering_results:
    if result.metadata["effective"]:
        print(f"Layer {result.layer}: optimal intensity = {result.data['optimal_intensity']}")
```

### 5. LoRA-Optimized Analysis

```python
# For LoRA adapters, NeuroDiff automatically optimizes memory usage
base_config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    load_in_8bit=True  # Use quantization
)

ft_config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    load_in_8bit=True,
    is_lora=True,
    lora_path="path/to/your/adapter",
    lora_alpha=16.0
)

# Only one model loaded in memory!
manager = DualModelManager(base_config, ft_config)
manager.load_models()
```

## Running Examples

```bash
# Weight analysis example
python /workspace/neurodiff/examples/weight_analysis_example.py

# Comprehensive activation analysis
python /workspace/neurodiff/examples/activation_analysis_example.py
```

## Key Features

- **Memory Efficient**: Automatic LoRA optimization
- **Comprehensive**: Weight, activation, and behavioral analysis
- **Visual**: Rich plots and heatmaps
- **Modular**: Easy to extend with custom analyzers
- **Research-Grade**: Implements latest interpretability research

## Next Steps

1. Replace the model paths with your actual base and fine-tuned models
2. Experiment with different analysis methods
3. Use the visualizations to understand your model's changes
4. Implement custom analysis methods by extending the base classes

## Support

For issues or questions, please refer to the comprehensive documentation in the README.md file.