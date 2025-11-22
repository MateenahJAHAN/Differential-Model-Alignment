"""
Core configuration for NeuroDiff system.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum
from pathlib import Path


class DiffingMethod(Enum):
    """Available diffing methods for model comparison."""
    WEIGHT_NORM = "weight_norm"
    ACTIVATION_DIFF = "activation_diff"
    CMAP = "cross_model_activation_patching"
    MODEL_STITCHING = "model_stitching"
    CROSSCODER = "crosscoder"
    LOGIT_LENS = "logit_lens"
    TASK_ARITHMETIC = "task_arithmetic"


class RegularizationType(Enum):
    """Regularization types for sparse dictionary learning."""
    L1 = "l1"
    L2 = "l2"
    BATCHTOPK = "batchtopk"  # Our innovation to avoid shrinkage artifacts


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    model_name: str
    model_path: Optional[Path] = None
    device: str = "cuda"
    dtype: str = "float16"
    use_flash_attention: bool = True
    quantization_bits: Optional[int] = None  # 4, 8, or None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    # LoRA specific
    is_lora: bool = False
    lora_path: Optional[Path] = None
    lora_alpha: float = 1.0
    

@dataclass
class DiffConfig:
    """Configuration for differential analysis."""
    base_model: ModelConfig
    finetuned_model: ModelConfig
    
    # Analysis settings
    methods: List[DiffingMethod] = field(default_factory=lambda: [DiffingMethod.WEIGHT_NORM])
    layers_to_analyze: Optional[List[int]] = None  # None means all layers
    components_to_analyze: List[str] = field(default_factory=lambda: ["attention", "mlp"])
    
    # Activation analysis
    num_samples: int = 1000
    batch_size: int = 32
    max_sequence_length: int = 512
    
    # Crosscoder settings
    crosscoder_config: Optional['CrosscoderConfig'] = None
    
    # Output settings
    output_dir: Path = Path("./outputs")
    save_activations: bool = False
    save_visualizations: bool = True
    

@dataclass 
class CrosscoderConfig:
    """Configuration for Crosscoder training and analysis."""
    dictionary_size: int = 16384  # Number of features to learn
    activation_dim: int = 4096  # Model hidden dimension
    
    # Training settings
    learning_rate: float = 1e-3
    batch_size: int = 256
    num_epochs: int = 10
    gradient_accumulation_steps: int = 4
    
    # Regularization
    regularization_type: RegularizationType = RegularizationType.BATCHTOPK
    l1_coefficient: float = 1e-4  # Only used if regularization_type is L1
    topk: int = 64  # Number of active features for BatchTopK
    
    # Architecture
    encoder_layers: int = 1
    decoder_layers: int = 1
    use_bias: bool = True
    tied_weights: bool = False  # Tie encoder and decoder weights
    
    # Optimization
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    gradient_clip_val: float = 1.0
    
    # Evaluation
    eval_steps: int = 100
    save_steps: int = 500
    log_steps: int = 10
    
    # Latent analysis
    compute_latent_scaling: bool = True
    compute_specificity_scores: bool = True
    

@dataclass
class ServingConfig:
    """Configuration for model serving and inference optimization."""
    backend: str = "pytorch"  # pytorch, vllm, tgi
    use_paged_attention: bool = True
    max_batch_size: int = 32
    max_concurrent_requests: int = 100
    
    # Memory optimization
    kv_cache_dtype: str = "float16"
    use_continuous_batching: bool = True
    
    # Hardware settings
    num_gpus: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    

@dataclass
class EvaluationConfig:
    """Configuration for model organism evaluation."""
    organism_types: List[str] = field(default_factory=lambda: ["sdf", "taboo", "subliminal", "cake"])
    num_eval_samples: int = 100
    
    # Agentic evaluation
    use_agent: bool = True
    agent_model: str = "gpt-4"
    agent_temperature: float = 0.7
    num_agent_questions: int = 5
    
    # Statistical analysis
    use_hibayes: bool = True
    confidence_level: float = 0.95
    

@dataclass
class DashboardConfig:
    """Configuration for interactive dashboard."""
    host: str = "0.0.0.0"
    port: int = 8501
    
    # Visualization settings
    color_scheme: str = "viridis"
    plot_backend: str = "plotly"  # plotly or matplotlib
    
    # Real-time settings
    update_interval: float = 0.5  # seconds
    max_stored_traces: int = 1000
    

@dataclass
class NeuroDiffConfig:
    """Master configuration for the entire NeuroDiff system."""
    diff_config: DiffConfig
    serving_config: ServingConfig = field(default_factory=ServingConfig)
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)
    dashboard_config: DashboardConfig = field(default_factory=DashboardConfig)
    
    # Logging and monitoring
    experiment_name: str = "neurodiff_experiment"
    use_wandb: bool = True
    wandb_project: str = "neurodiff"
    log_level: str = "INFO"
    
    # Paths
    cache_dir: Path = Path("./cache")
    model_organisms_dir: Path = Path("./model_organisms")
    checkpoints_dir: Path = Path("./checkpoints")