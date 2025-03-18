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
 - Parameter Itself: Stored as an FP32 tensor, each element takes 4 bytes.
 - Gradient: Also stored in FP32, so another 4 bytes per element.
 - First Moment (exp_avg): Adam keeps an exponential moving average of past gradients (the “m” term), stored in FP32 (4 bytes per element).
 - Second Moment (exp_avg_sq): Adam also keeps an exponential moving average of the squared gradients (the “v” term), again in FP32 (4 bytes per element).
---
If you sum these up, each parameter effectively uses:
4 (parameter) + 4 (gradient) + 4 (exp_avg) + 4 (exp_avg_sq) = 16 bytes per parameter.
---
How AMP Lowers Memory Usage
---
When using Automatic Mixed Precision (AMP), many of the tensors—especially the intermediate activations and sometimes gradients during the forward and backward passes—are stored in FP16 instead of FP32. Since FP16 uses 2 bytes per element, this can significantly reduce memory usage in those parts of the training loop.
 - Activations & Intermediate Buffers: These often dominate the memory footprint during training. Storing them in FP16 cuts their memory cost roughly in half.
 - Master Copy: AMP maintains a master copy of the parameters in FP32 for numerical stability during updates. However, since parameters typically account for a smaller fraction of total memory (compared to activations), the savings on activations and gradients can be substantial.
---
Thus, while the optimizer state (the two Adam buffers) remains in FP32 (16 bytes per parameter total as shown), the overall memory usage decreases because many other tensors used during training are now in FP16.
---
<img width="992" alt="Screenshot 2025-03-14 at 14 35 59" src="https://github.com/user-attachments/assets/2b512f6b-b76c-48a5-9b82-a24fd6bd6403" />

<img width="935" alt="Screenshot 2025-03-14 at 12 03 43" src="https://github.com/user-attachments/assets/24cb4c92-3b9a-4431-86ca-bc0eace21524" />
tokkemize and preprocess before training 
<img width="915" alt="Screenshot 2025-03-14 at 12 15 35" src="https://github.com/user-attachments/assets/8be8f45c-2872-4708-b0e6-232fa31f6212" />
<img width="928" alt="Screenshot 2025-03-14 at 12 19 35" src="https://github.com/user-attachments/assets/7d61a6a5-b2ec-4357-a5fe-ed8f3e6a500e" />
<img width="870" alt="Screenshot 2025-03-14 at 12 21 30" src="https://github.com/user-attachments/assets/37f16b0d-89d3-4203-9cde-2a444c97d1dc" />
<img width="853" alt="Screenshot 2025-03-14 at 12 22 09" src="https://github.com/user-attachments/assets/447922a2-1925-4068-96c4-aeb6d216a0c5" />

When we use activation function we should store input (as for all layers) cause after we calculate backpropagation and derivation depend input value (example ReLU)


```torch
# Memory leak via profiling checking - compose diagram 
torch.cuda.memory._record_memory_history()
torch.cuda.memory._dump_snapshot(f"snapshot_amp=reranker.pickle")
```

---
Parallelism is a technique used to preform multiple computation at same time. Two common approaches: thread-based and process based. 
- Thread-based - single process create multiple threads that share the same memory space
- Process-based parallelism - each process has its own memory space and runs independently
---
<img width="1724" alt="Screenshot 2025-03-16 at 10 42 04" src="https://github.com/user-attachments/assets/f90a4321-8374-4d0a-b868-9ea7aa650c28" />


---
Multithreading is well-suited for I/O-bound tasks because GIL ensures that only one thread executes Python bytecode at time.
In I/O operation thread enters a waiting time. During this time, the thread release the GIL, allowing other threads to run. 
thread.join() - wait thread finish
```pytorch
def run(rank, size):
    """ Distributed function to be implemented later. """
    if rank!=0:
        dist.barrier()
    print(f'Started {rank}',flush=True)
    if rank==0:
        dist.barrier()

if __name__ == "__main__":
    size = 4
    processes = []
    port = random.randint(25000, 30000)
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run, port))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
```
---
1. Data Parallel (DP):
 - Multithreading-based (single process).
 - Uses Python threads to manage multiple GPUs.
 - Each GPU receives a separate copy of the model.
 - The model weights (parameters) are initially loaded onto GPU-0 and then replicated (copied) to other GPUs.
 - Each GPU receives a different batch of data, runs a forward and backward pass independently, and computes gradients.
 - After each GPU computes gradients, these gradients are gathered back to GPU-0.
 - Gradients from all GPUs are then averaged, and a single update step is executed on GPU-0.
 - Finally, updated weights are copied back to all GPUs. This synchronization occurs after every batch.

This process happens in parallel across GPUs but managed via threads within a single Python process. The Global Interpreter Lock (GIL) isn’t an issue here because the heavy computational workload is GPU-bound and handled by CUDA (outside the GIL).

---

2. Distributed Data Parallel (DDP):
 - Multiprocessing-based (multiple processes).
 - Each GPU has its own independent Python process.
 - Each process independently loads a copy of the model on its GPU.
 - DDP handles synchronization of parameters and gradients via a communication backend (NCCL usually).
 - No single GPU acts as a “leader”: all GPUs/processes independently perform forward/backward computations on their own data batches simultaneously.
 - After gradients are computed on each GPU, DDP synchronizes gradients across GPUs automatically (via an AllReduce operation).
 - Each GPU independently updates its weights, but since gradients are identical (post-synchronization), model weights remain synchronized across GPUs.

---
