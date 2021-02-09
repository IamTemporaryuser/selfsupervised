import torch
import torch.nn as nn
import torch.nn.functional as F

def sinkhorn(Q, niters=3): 
    Q = Q.T
    Q /= torch.sum(Q) 
    K, B = Q.shape 
    r, c = 1 / K, 1 / B
    for _ in range(niters):
        Q *= (r / torch.sum(Q, dim=1)).unsqueeze(1) 
        Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0) 
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T

def sinkhorn_dst(Q, dst=torch.Tensor([0.2, 0.8]), niters=3): 
    Q = Q.T
    Q /= (torch.sum(Q) + 1)
    K, B = Q.shape 
    B = B + 1
    r, c = 1 / K, 1 / B
    for _ in range(niters):
        Q *= (r / (torch.sum(Q, dim=1) + dst)).unsqueeze(1) 
        Q *= (c / (torch.sum(Q, dim=0) + 1)).unsqueeze(0) 
    return (Q / torch.sum(Q, dim=0, keepdim=True)).T

a = torch.Tensor([[0.3, 0.7], [0.4,0.6], [0.55, 0.85]])
p = sinkhorn(a)
print(p)
print(p.sum(dim=0))
print(p.sum(dim=1))
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)
# print(b(a), b.queue)