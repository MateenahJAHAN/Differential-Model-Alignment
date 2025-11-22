"""
NeuroDiff: A Comprehensive Architectural Framework for Differential Mechanistic Interpretability
"""

__version__ = "0.1.0"

from .core.config import (
    NeuroDiffConfig,
    DiffConfig,
    ModelConfig,
    CrosscoderConfig,
    ServingConfig,
    EvaluationConfig,
    DashboardConfig,
    DiffingMethod,
    RegularizationType,
)

from .core.base import (
    DiffResult,
    ModelWrapper,
    ActivationStore,
    Hook,
    MetricTracker,
    CacheManager,
)

from .core.model_loader import (
    HuggingFaceModelWrapper,
    DualModelManager,
    ModelRegistry,
)

from .analysis.weight_diff import (
    WeightDifferenceAnalyzer,
    TaskArithmeticAnalyzer,
    SpectralAnalyzer,
)

from .analysis.activation_diff import (
    ActivationDifferenceAnalyzer,
    ActivationTrace,
    LogitLensAnalyzer,
    CrossModelActivationPatcher,
    ModelStitcher,
)

from .analysis.steering import (
    ActivationSteering,
    SteeringVector,
    ReadableTraceAnalyzer,
)

__all__ = [
    # Configuration
    "NeuroDiffConfig",
    "DiffConfig", 
    "ModelConfig",
    "CrosscoderConfig",
    "ServingConfig",
    "EvaluationConfig",
    "DashboardConfig",
    "DiffingMethod",
    "RegularizationType",
    # Core infrastructure
    "DiffResult",
    "ModelWrapper",
    "HuggingFaceModelWrapper",
    "DualModelManager",
    "ModelRegistry",
    "ActivationStore",
    "Hook",
    "MetricTracker",
    "CacheManager",
    # Tier I: Weight analysis
    "WeightDifferenceAnalyzer",
    "TaskArithmeticAnalyzer",
    "SpectralAnalyzer",
    # Tier II: Activation analysis
    "ActivationDifferenceAnalyzer",
    "ActivationTrace",
    "LogitLensAnalyzer",
    "CrossModelActivationPatcher",
    "ModelStitcher",
    # Steering and intervention
    "ActivationSteering",
    "SteeringVector",
    "ReadableTraceAnalyzer",
]