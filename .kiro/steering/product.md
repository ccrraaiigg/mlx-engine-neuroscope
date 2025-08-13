# Product Overview

MLX Engine with NeuroScope Integration is a mechanistic interpretability framework that extends Apple's MLX machine learning framework for LLM inference with activation capture capabilities.

## Core Purpose
- **Primary**: High-performance LLM inference on Apple Silicon using MLX
- **Extension**: Mechanistic interpretability analysis through NeuroScope integration
- **Target**: Researchers and developers analyzing transformer model internals

## Key Features
- Activation hook infrastructure for capturing internal model states
- REST API server with OpenAI-compatible endpoints
- Smalltalk bridge for NeuroScope integration
- Support for Mixture of Experts (MoE) architectures
- Circuit analysis and attention pattern visualization
- Streaming generation with real-time activation capture

## Model Support
- Text models via ModelKit (primary focus)
- Vision models via VisionModelKit
- Quantized models (4-bit, 8-bit)
- Large context models (up to 131k tokens)
- MoE architectures (tested with GPT-OSS-20B)

## Use Cases
- Circuit discovery in transformer models
- Attention pattern analysis
- Mechanistic interpretability research
- Model behavior analysis and debugging
- Educational exploration of model internals