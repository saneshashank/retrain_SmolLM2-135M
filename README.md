# Custom Llama2-135M Implementation and Fine-tuning with PyTorch Optimizations

This notebook provides a step-by-step guide to implementing a custom Llama-like causal language model from scratch in PyTorch. It then demonstrates how to load pre-trained weights from `SmolLM2-135M` (a Hugging Face model), fine-tune it on a subset of the TinyStories dataset, and apply several performance optimizations including TF32, Automatic Mixed Precision (AMP) with `torch.amp.autocast` and `GradScaler`, and `torch.compile`.

Finally, it includes instructions on how to package the fine-tuned model and tokenizer for inference using a Gradio application.

## Table of Contents

1.  [Overview](#overview)
2.  [Custom Llama Architecture](#custom-llama-architecture)
3.  [Training Setup](#training-setup)
4.  [Performance Optimizations](#performance-optimizations)
    *   [TF32 Matmul Precision](#tf32-matmul-precision)
    *   [Automatic Mixed Precision (AMP)](#automatic-mixed-precision-amp)
    *   [Torch.compile](#torchcompile)
5.  [Saving the Fine-tuned Model and Tokenizer](#saving-the-fine-tuned-model-and-tokenizer)
6.  [Gradio Inference App](#gradio-inference-app)
7.  [Requirements](#requirements)


## Overview

This notebook aims to demonstrate the process of building, training, and optimizing a small causal language model. It covers:

*   **Manual Implementation**: Detailed breakdown of a Llama-like architecture (RMSNorm, RotaryEmbedding, LlamaAttention, LlamaMLP, LlamaDecoderLayer, LlamaModel).
*   **Weight Loading**: Transferring pre-trained weights from `HuggingFaceTB/SmolLM2-135M` to the custom model.
*   **Data Preparation**: Loading and tokenizing the TinyStories dataset for fine-tuning.
*   **Optimization Techniques**: Applying `torch.backends.cuda.matmul.allow_tf32`, `torch.amp.autocast` with `GradScaler`, and `torch.compile` to accelerate training.
*   **PyTorch Lightning**: Refactoring the training loop using PyTorch Lightning for better structure and boilerplate reduction.
*   **Inference Deployment**: Creating a Gradio interface for interacting with the fine-tuned model.

## Custom Llama Architecture

The core of this project is a custom implementation of a Llama-like transformer model. This section details each component:

*   **`RMSNorm`**: A Root Mean Square Normalization layer, which is a simpler and often more efficient alternative to LayerNorm, typically used in Llama models.
*   **`RotaryEmbedding`**: Implements Rotary Positional Embeddings (RoPE), a method to inject positional information into attention keys and queries without using absolute position embeddings.
*   **`LlamaAttention`**: The self-attention mechanism, featuring Multi-Head Attention (MHA) or Grouped Query Attention (GQA) for efficiency, and incorporating RoPE.
*   **`LlamaMLP`**: The feed-forward network (FFN), using a SwiGLU (Swish Gated Linear Unit) activation pattern (SiLU activation with gating).
*   **`LlamaDecoderLayer`**: A single block of the transformer, combining `LlamaAttention` and `LlamaMLP` with pre-normalization using `RMSNorm` and residual connections.
*   **`LlamaModel`**: The complete model, stacking multiple `LlamaDecoderLayer` instances, an embedding layer for token inputs, and a linear `lm_head` for output logits.

Each custom component is designed to mirror the structure and functionality of the SmolLM2-135M model, allowing for direct weight transfer.

## Training Setup

The training process involves:

*   **Dataset**: A subset of the `roneneldan/TinyStories` dataset is used for fine-tuning, providing simple, narrative text data.
*   **Tokenizer**: The `AutoTokenizer` from `HuggingFaceTB/SmolLM2-135M` is utilized for consistent tokenization.
*   **DataLoader**: `torch.utils.data.DataLoader` is used to efficiently batch and shuffle the tokenized dataset.
*   **Optimizer**: `torch.optim.AdamW` is chosen as the optimizer for training the model.
*   **Loss Function**: `F.cross_entropy` is employed for calculating the language modeling loss.
*   **Training Loop**: A standard training loop is implemented, iterating over batches, performing forward and backward passes, and updating model weights.
*   **PyTorch Lightning Integration**: The training loop is refactored into a `pl.LightningModule` and `pl.LightningDataModule` for improved structure, readability, and access to advanced features.

## Performance Optimizations

To enhance training speed and efficiency, the following PyTorch optimizations are applied:

### TF32 Matmul Precision

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

This setting enables TensorFloat-32 (TF32) arithmetic for `float32` matrix multiplications and convolutions on compatible NVIDIA GPUs (Ampere architecture and newer). TF32 offers up to 3x speedup compared to full `float32` by combining `float32`'s range with `float16`'s precision internally, with minimal impact on model accuracy for most deep learning workloads.

### Automatic Mixed Precision (AMP)

```python
from torch.amp import autocast, GradScaler
# ...
scaler = GradScaler()
# ...
with autocast('cuda'):
    logits = model(input_ids)
    loss = F.cross_entropy(logits[:,:-1,:].reshape(-1,logits.shape[-1]),
                           input_ids[:,1:].reshape(-1))

optimizer.zero_grad()
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

AMP, facilitated by `torch.amp.autocast` and `GradScaler`, automatically performs operations in `float16` where appropriate to speed up computations and reduce memory usage, while keeping numerically unstable operations in `float32`. `GradScaler` prevents `float16` underflow by scaling loss values before backpropagation.

### Torch.compile

```python
model = torch.compile(model)
```

Introduced in PyTorch 2.0, `torch.compile` is a JIT compiler that optimizes your PyTorch code for significant speedups (often 1.5x - 2x or more). It traces the model's execution to create a computational graph, performs kernel fusion, memory optimizations, and generates highly efficient low-level code tailored for the GPU. This is a one-line change that can drastically improve performance.

## Saving the Fine-tuned Model and Tokenizer

After fine-tuning, the model's weights are saved in `float16` precision to reduce file size and optimize for inference. The tokenizer is also saved to ensure consistency during deployment.

```python
# Save model weights in float16
model_state_dict = model.state_dict()
float16_state_dict = {}
for k, v in model_state_dict.items():
    clean_k = k.replace('_orig_mod.', '') if '_orig_mod.' in k else k
    float16_state_dict[clean_k] = v.to(torch.float16)
torch.save(float16_state_dict, "llama_model_weights.pt")

# Save tokenizer
tokenizer.save_pretrained("smollm_tokenizer")
```

## Gradio Inference App

A Gradio application is provided to interact with the fine-tuned model for text generation. The app includes controls for prompt text, maximum generation length, temperature, and top-k sampling, allowing users to experiment with different generation styles.

**To run the Gradio app:**

1.  Ensure you have `llama_model_weights.pt` and a saved tokenizer (or download `HuggingFaceTB/SmolLM2-135M` tokenizer if not fine-tuning).
2.  Execute the `gradio_inference_script.py` (which is essentially the content of cell `gRxrx0N9DRNs` in this notebook).
    ```bash
    python gradio_inference_script.py
    ```
3.  Access the Gradio interface through the local or public URL provided in your terminal.

## Requirements

To run this notebook and the Gradio application, you will need the following Python packages. It's recommended to create a virtual environment.

```
# requirements.txt
torch>=2.0.0
transformers>=4.0.0
datasets>=2.0.0
tqdm>=4.0.0
gradio>=4.0.0
pytorch-lightning>=2.0.0
```
