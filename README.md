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
