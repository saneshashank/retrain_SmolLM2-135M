# Custom SmolLM2-135M Implementation and Fine-tuning with PyTorch Optimizations

This notebook provides a step-by-step guide to implementing a custom Llama-like causal language model from scratch in PyTorch. It then demonstrates how to load pre-trained weights from `SmolLM2-135M` (a Hugging Face model), fine-tune it on a subset of the TinyStories dataset, and apply several performance optimizations including TF32, Automatic Mixed Precision (AMP) with `torch.amp.autocast` and `GradScaler`, and `torch.compile`. The repo also demonstrates a Gradio-based inference application and conversion to Pytorch Lightning code.

## Table of Contents

1.  [Overview](#overview)
2.  [Model Architecture](#model-architecture)
3.  [Training Setup](#training-setup)
4.  [Performance Optimizations](#performance-optimizations)
    *   [TF32 Precision](#tf32-precision)
    *   [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)
    *   [torch.compile](#torchcompile)
5.  [PyTorch Profiler](#pytorch-profiler)
6.  [PyTorch Lightning Integration](#pytorch-lightning-integration)
7.  [Saving the Fine-tuned Model](#saving-the-fine-tuned-model)
8.  [Gradio Inference App](#gradio-inference-app)
9.  [Setup and Usage](#setup-and-usage)
10. [Requirements](#requirements)

## Overview

The goal of this notebook is to provide a clear, step-by-step guide to building a simplified Llama2-like model, training it on the TinyStories dataset, optimizing its performance, and deploying it for inference using Gradio.

## Model Architecture

The custom Llama2-like model is built using fundamental PyTorch components, replicating key elements of the Llama architecture:

*   **RMSNorm**: An efficient normalization layer, replacing traditional LayerNorm.
*   **RotaryEmbedding (RoPE)**: Implements Rotary Positional Embeddings to inject positional information into attention keys and queries.
*   **LlamaAttention**: The self-attention mechanism, featuring RoPE and Grouped Query Attention (GQA) for efficiency.
*   **LlamaMLP**: The feed-forward network, utilizing the SiLU activation function (SwiGLU).
*   **LlamaDecoderLayer**: A single transformer block combining `LlamaAttention` and `LlamaMLP` with residual connections and `RMSNorm`.
*   **LlamaModel**: The complete model, stacking multiple `LlamaDecoderLayer` instances, an embedding layer, and a final language model head.

The model is initialized with weights from `HuggingFaceTB/SmolLM2-135M` for faster convergence.

## Training Setup

The model is fine-tuned on a subset of the **TinyStories** dataset (`roneneldan/TinyStories`).

**Key components of the training loop:**

*   **Tokenizer**: `AutoTokenizer` from Hugging Face Transformers.
*   **Dataset Preparation**: The TinyStories dataset is tokenized, truncated/padded to `max_length=128`, and loaded into a `DataLoader` with `batch_size=8`.
*   **Optimizer**: `AdamW` is used with a learning rate of `5e-5`.
*   **Loss Function**: `F.cross_entropy` for causal language modeling.
*   **Training Steps**: The model is trained for 5000 steps.

## Performance Optimizations

Several techniques are applied to enhance training speed and efficiency on GPU:

### TF32 Precision

`torch.backends.cuda.matmul.allow_tf32 = True` and `torch.backends.cudnn.allow_tf32 = True` enable **TensorFloat-32 (TF32)** arithmetic for `float32` matrix multiplications and convolutions on compatible NVIDIA GPUs (Ampere and newer). This provides significant speedups with minimal impact on model accuracy by utilizing GPU Tensor Cores.

### Automatic Mixed Precision (AMP)

`torch.amp.autocast` and `torch.cuda.amp.GradScaler` are used for Automatic Mixed Precision training.
AMP performs operations in `float16` where possible, reducing memory usage and speeding up computations, while keeping numerically sensitive operations in `float32` for stability. `GradScaler` prevents gradient underflow issues associated with `float16` training.

### `torch.compile`

Introduced in PyTorch 2.0, `torch.compile(model)` significantly accelerates the model by JIT-compiling the PyTorch code into highly optimized kernels. It achieves this through graph capture, kernel fusion, and memory optimization, often leading to 1.5x-2x speedups.

## PyTorch Profiler

The `torch.profiler` is used to analyze the performance of the optimized training loop. It provides detailed breakdowns of CPU and CUDA activity, helping to identify bottlenecks in data loading, forward pass, and backward pass operations.

## PyTorch Lightning Integration

The entire training process is re-implemented using **PyTorch Lightning** for better structure, reproducibility, and boilerplate reduction.

*   **`LlamaLightningModule`**: Encapsulates the custom `LlamaModel`, forward pass, training step, and optimizer configuration.
*   **`TinyStoriesDataModule`**: Manages dataset loading, tokenization, and `DataLoader` creation.
*   **`pl.Trainer`**: Orchestrates the training, handling `precision="16-mixed"` for AMP, logging with `TensorBoardLogger`, and checkpointing with `ModelCheckpoint`.

## Saving the Fine-tuned Model

After training, the model's state dictionary is saved to `llama_model_weights.pt` with weights converted to `torch.float16` for efficient inference. The tokenizer is also loaded directly from Hugging Face for consistent tokenization.

## Gradio Inference App

A Gradio web interface is created to allow interactive text generation using the fine-tuned model. The app features:

*   **Text Generation Function**: `generate_text` handles tokenization, model inference, and decoding, incorporating sampling parameters like `temperature` and `top_k`.
*   **Interactive Controls**: Sliders for `Max Length`, `Temperature`, and `Top-K Sampling` to customize the generation output.

The inference script (`gradio_app.py`) is designed to be self-contained, including the necessary Llama architecture classes, model loading logic, and the Gradio interface.

## Setup and Usage

1.  **Open the Colab Notebook**: Navigate to the provided notebook code: 
2.  **Run All Cells**: Execute all cells sequentially. This will:
    *   Define the custom Llama architecture.
    *   Load the pre-trained `SmolLM2-135M` weights.
    *   Set up data loading and tokenization.
    *   Perform baseline training.
    *   Apply and demonstrate performance optimizations (TF32, AMP, `torch.compile`).
    *   Integrate PyTorch Lightning for structured training.
    *   Save the fine-tuned model weights.
    *   Launch the Gradio inference application directly within the notebook.
3.  **Interact with Gradio**: Once the Gradio app launches (you'll see a public URL), open it in a new tab to test the text generation.
4.  You can access the hf spaces hosted model here: https://huggingface.co/spaces/saneshashank/retrain_SmolLM2-135M

## Requirements

The necessary Python packages are installed within the Colab environment. Key dependencies include:

```
transformers
datasets
tqdm
torch
pytorch_lightning
gradio
```
