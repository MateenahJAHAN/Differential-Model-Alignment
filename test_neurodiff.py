#!/usr/bin/env python3
"""
Simple test script to verify NeuroDiff installation and basic functionality.
"""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test that all main components can be imported."""
    print("Testing imports...")
    try:
        from neurodiff import (
            ModelConfig, DiffConfig, NeuroDiffConfig,
            WeightDifferenceAnalyzer, TaskArithmeticAnalyzer,
            ActivationDifferenceAnalyzer, LogitLensAnalyzer,
            CrossModelActivationPatcher, ActivationSteering,
            DualModelManager
        )
        print("✓ All imports successful!")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_configuration():
    """Test configuration creation."""
    print("\nTesting configuration...")
    try:
        from neurodiff import ModelConfig, DiffConfig
        
        base_config = ModelConfig(
            model_name="gpt2",
            device="cpu",  # Use CPU for testing
            dtype="float32"
        )
        
        ft_config = ModelConfig(
            model_name="gpt2",
            device="cpu",
            dtype="float32",
            is_lora=False
        )
        
        diff_config = DiffConfig(
            base_model=base_config,
            finetuned_model=ft_config,
            layers_to_analyze=[0, 1],
            output_dir=Path("./test_output")
        )
        
        print("✓ Configuration objects created successfully!")
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_analyzer_creation():
    """Test analyzer instantiation."""
    print("\nTesting analyzer creation...")
    try:
        from neurodiff import (
            ModelConfig, DiffConfig,
            WeightDifferenceAnalyzer,
            ActivationDifferenceAnalyzer
        )
        
        # Create minimal config
        base_config = ModelConfig(model_name="gpt2", device="cpu")
        ft_config = ModelConfig(model_name="gpt2", device="cpu")
        diff_config = DiffConfig(base_model=base_config, finetuned_model=ft_config)
        
        # Create analyzers
        weight_analyzer = WeightDifferenceAnalyzer(diff_config)
        act_analyzer = ActivationDifferenceAnalyzer(diff_config)
        
        print("✓ Analyzers created successfully!")
        return True
    except Exception as e:
        print(f"✗ Analyzer creation error: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("NeuroDiff Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Analyzer Creation Test", test_analyzer_creation),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name}...")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 60)
    print(f"Test Summary: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! NeuroDiff is ready to use.")
        print("\nNext steps:")
        print("1. Run the examples in /workspace/neurodiff/examples/")
        print("2. Replace 'gpt2' with your actual model paths")
        print("3. Start analyzing your fine-tuned models!")
    else:
        print("❌ Some tests failed. Please check the installation.")
        sys.exit(1)

if __name__ == "__main__":
    main()