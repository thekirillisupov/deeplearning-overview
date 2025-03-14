# deeplearning-overview

### Computational graph 
is dynamically built during the forward pass and consists of all operations that have requires_grad=True.

Detached tensors .detach() - regular tensors with no gradient tracking
``` python
with torch.no_grad():
  # infrence 
```
item() -> transfer to cpu python format, trigger synchronization between CPU and GPU

### PyTorch uses a Caching Memory Allocator
- allocating memory os expensive, so PyTorch reuses memory when possible. It keeps it in a cache instead of freeing memory

https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html

CUDA graphs capture a sequence of specific operations (i.e., kernel launches) that can later be replayed on new inputs with essentially a single kernel launch.
Especially for scalar operations. 
``` python
@torch.compile
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / 1.41421))
```

Model Flops Utilization = ratio of observed FLOPS to theoretical maximum FLOPS on given hardware 

nvidia-smi utilization shows ratio of quant time in less one kernel is runing

<img width="677" alt="Screenshot 2025-03-14 at 10 21 15" src="https://github.com/user-attachments/assets/17096cd4-7206-44dd-b655-51a20f585ae4" />
sign, exponent, mantisa

<img width="840" alt="Screenshot 2025-03-14 at 10 27 45" src="https://github.com/user-attachments/assets/f69093b4-d94c-40f8-abc6-4aa2ec8bd41b" />

<img width="906" alt="Screenshot 2025-03-14 at 11 48 51" src="https://github.com/user-attachments/assets/c2bf9ad9-eb9f-48b9-ac27-6c998b25b649" />
<img width="917" alt="Screenshot 2025-03-14 at 11 51 58" src="https://github.com/user-attachments/assets/8ee45f78-5b72-4498-95da-a273551338d3" />

torch.amp provides convenience methods for **mixed precision**, where some operations use the torch.float32 (float) datatype and other operations use lower precision floating point datatype (lower_precision_fp): torch.float16 (half) or torch.bfloat16. Some ops, like linear layers and convolutions, are much faster in lower_precision_fp. Other ops, like reductions, often require the dynamic range of float32. Mixed precision tries to match each op to its appropriate datatype.
---
In standard FP32 training with Adam, each parameter is typically associated with several tensors stored in memory:
 • Parameter Itself: Stored as an FP32 tensor, each element takes 4 bytes.
 • Gradient: Also stored in FP32, so another 4 bytes per element.
 • First Moment (exp_avg): Adam keeps an exponential moving average of past gradients (the “m” term), stored in FP32 (4 bytes per element).
 • Second Moment (exp_avg_sq): Adam also keeps an exponential moving average of the squared gradients (the “v” term), again in FP32 (4 bytes per element).
---
If you sum these up, each parameter effectively uses:
4 (parameter) + 4 (gradient) + 4 (exp_avg) + 4 (exp_avg_sq) = 16 bytes per parameter.
---
How AMP Lowers Memory Usage
---
When using Automatic Mixed Precision (AMP), many of the tensors—especially the intermediate activations and sometimes gradients during the forward and backward passes—are stored in FP16 instead of FP32. Since FP16 uses 2 bytes per element, this can significantly reduce memory usage in those parts of the training loop.
 • Activations & Intermediate Buffers: These often dominate the memory footprint during training. Storing them in FP16 cuts their memory cost roughly in half.
 • Master Copy: AMP maintains a master copy of the parameters in FP32 for numerical stability during updates. However, since parameters typically account for a smaller fraction of total memory (compared to activations), the savings on activations and gradients can be substantial.
---
Thus, while the optimizer state (the two Adam buffers) remains in FP32 (16 bytes per parameter total as shown), the overall memory usage decreases because many other tensors used during training are now in FP16.
---

<img width="935" alt="Screenshot 2025-03-14 at 12 03 43" src="https://github.com/user-attachments/assets/24cb4c92-3b9a-4431-86ca-bc0eace21524" />
tokkemize and preprocess before training 
<img width="915" alt="Screenshot 2025-03-14 at 12 15 35" src="https://github.com/user-attachments/assets/8be8f45c-2872-4708-b0e6-232fa31f6212" />
<img width="928" alt="Screenshot 2025-03-14 at 12 19 35" src="https://github.com/user-attachments/assets/7d61a6a5-b2ec-4357-a5fe-ed8f3e6a500e" />
<img width="870" alt="Screenshot 2025-03-14 at 12 21 30" src="https://github.com/user-attachments/assets/37f16b0d-89d3-4203-9cde-2a444c97d1dc" />
<img width="853" alt="Screenshot 2025-03-14 at 12 22 09" src="https://github.com/user-attachments/assets/447922a2-1925-4068-96c4-aeb6d216a0c5" />


