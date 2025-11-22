# NeuroDiff: Advanced Model Diffing for Mechanistic Interpretability

NeuroDiff is a comprehensive framework for differential mechanistic interpretability of fine-tuned Large Language Models (LLMs). It provides cutting-edge tools to understand *what* changes when a model is fine-tuned, *where* those changes occur, and *how* they affect behavior.

## 🌟 Key Features

### Tier I: Weight Space Analysis
- **Weight Difference Analysis**: Comprehensive weight diff metrics (Frobenius, spectral, nuclear norms)
- **Task Arithmetic**: Extract and manipulate task vectors (τ = W_ft - W_base)
- **Spectral Analysis**: SVD-based analysis of weight changes and rank properties

### Tier II: Activation Dynamics
- **Activation Difference Analysis**: Track how internal representations change
- **Differential LogitLens**: Compare next-token predictions layer by layer
- **Cross-Model Activation Patching (CMAP)**: Test behavioral modularity
- **Model Stitching**: Localize behaviors by creating hybrid models
- **Activation Steering**: Use difference vectors to control model behavior
- **Readable Trace Analysis**: Detect persistent semantic fingerprints

### Tier III: Advanced Dictionary Learning (Coming Soon)
- **BatchTopK Crosscoders**: Shared dictionary learning without L1 artifacts
- **Latent Scaling Analysis**: Identify model-specific vs shared features
- **Sparse Feature Attribution**: Human-interpretable feature decomposition

## 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/neurodiff/neurodiff.git
cd neurodiff

# Install with pip (recommended)
pip install -e .

# Or install with all optional dependencies
pip install -e ".[dev,serving,research]"
```

## 📖 Quick Start

### Basic Weight Analysis

```python
from neurodiff import (
    ModelConfig, DiffConfig, WeightDifferenceAnalyzer,
    DualModelManager
)

# Configure models
base_config = ModelConfig(model_name="gpt2", device="cuda")
ft_config = ModelConfig(model_name="gpt2-finetuned", device="cuda")

# Create analysis config
diff_config = DiffConfig(
    base_model=base_config,
    finetuned_model=ft_config,
    layers_to_analyze=[0, 1, 2],  # Analyze first 3 layers
    save_visualizations=True,
)

# Load models and analyze
manager = DualModelManager(base_config, ft_config)
manager.load_models()

analyzer = WeightDifferenceAnalyzer(diff_config)
results = analyzer.compute_diff(manager, None)
figures = analyzer.visualize(results)
```

### Activation Analysis with Steering

```python
from neurodiff import (
    ActivationDifferenceAnalyzer, ActivationSteering,
    ReadableTraceAnalyzer
)

# Prepare input data
inputs = tokenizer("The cat sat on the mat", return_tensors="pt").input_ids

# Analyze activation differences
act_analyzer = ActivationDifferenceAnalyzer(diff_config)
act_results = act_analyzer.compute_diff(manager, None, inputs)

# Extract steering vectors
steering = ActivationSteering(diff_config)
steering_results = steering.compute_diff(manager, None, inputs)

# Check for readable traces
trace_analyzer = ReadableTraceAnalyzer(diff_config)
trace_results = trace_analyzer.compute_diff(manager, None, inputs)
```

### LoRA-Optimized Analysis

```python
# NeuroDiff automatically detects and optimizes for LoRA adapters
base_config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    load_in_8bit=True,  # Use quantization
)

ft_config = ModelConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    device="cuda",
    load_in_8bit=True,
    is_lora=True,
    lora_path="path/to/adapter",
    lora_alpha=16.0,
)

# Memory-efficient analysis (only one model in memory!)
manager = DualModelManager(base_config, ft_config)
manager.load_models()  # Automatically uses LoRA optimization
```

## 🏗️ Architecture

NeuroDiff follows a modular architecture:

```
neurodiff/
├── core/           # Core infrastructure
│   ├── base.py     # Base classes and interfaces
│   ├── config.py   # Configuration management
│   └── model_loader.py  # Model loading with LoRA support
├── analysis/       # Analysis modules
│   ├── weight_diff.py     # Tier I: Weight analysis
│   ├── activation_diff.py # Tier II: Activation analysis
│   └── steering.py        # Steering and intervention
├── organisms/      # Model organisms framework
├── evaluation/     # Automated evaluation tools
└── dashboard/      # Interactive visualization
```

## 🔬 Research Foundation

NeuroDiff implements cutting-edge research from:

1. **"Overcoming Sparsity Artifacts in Crosscoders"** - BatchTopK solution for dictionary learning
2. **"Narrow Finetuning Leaves Clearly Readable Traces"** - Activation difference persistence
3. **Cross-Model Activation Patching (CMAP)** - Behavioral modularity testing
4. **Task Arithmetic** - Manipulating model capabilities through weight space

## 🎯 Use Cases

- **Safety Auditing**: Detect hidden capabilities or misalignment in fine-tuned models
- **Capability Analysis**: Understand what specific behaviors fine-tuning introduces
- **Model Debugging**: Identify which layers/components drive behavioral changes
- **Research**: Advance mechanistic understanding of neural network adaptation

## 📊 Visualizations

NeuroDiff provides rich visualizations including:
- Layer-wise weight difference heatmaps
- Activation trajectory plots
- Steering effectiveness curves
- Model stitching performance landscapes
- Readable trace consistency analysis

## 🤝 Contributing

We welcome contributions! Key areas for development:

1. **BatchTopK Crosscoders**: Complete Tier III implementation
2. **Model Organisms**: Expand the library of test models
3. **Dashboard**: Build the interactive web interface
4. **Evaluation**: Implement automated agentic evaluation

## 📚 Documentation

For detailed documentation, see:
- [Architecture Guide](docs/architecture.md)
- [API Reference](docs/api_reference.md) 
- [Research Papers](docs/papers.md)

## ⚖️ License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This project builds on research from Apart Research, Anthropic, and the broader mechanistic interpretability community.

## 📧 Contact

For questions or collaborations, please open an issue or contact the maintainers.