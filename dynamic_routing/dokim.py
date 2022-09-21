import torch
import numpy as np

c = torch.randn(2,15, 10)
u = torch.randn(2,15,10,4)
# Το v θέλω να είναι 10,4 shape.
indices = c.argmax(dim=1) # along child capsule dimension

# b = torch.arange(0,150,15) # Create a tensor used for mapping from 2D to 1D.
# u_flat = u.flatten(start_dim=1, end_dim=2) # Flatten the dimensions, result: num_child_caps*num_parent_caps.
# s = u_flat.index_select(1, indices+b) # index_select takes as index only 1D tensors. That's why we had to flatten the tensor.
#                                         # We select the indices that have the biggest ci, for all j digit capsules.
#                                         # So shape of s is [batch_size, number_of_digit_caps, digit_caps_dimension]

c_zero = torch.zeros_like(c)
values, idx = c.max(dim=1)
c_ones = torch.ones_like(c)
c_sparse = c_zero.scatter(1, idx.unsqueeze(dim=1), values.unsqueeze(dim=1))

s = (c_sparse.unsqueeze(dim=3) * u).sum(dim=1)

#########
u_c = u.norm(dim=3) # Take the l2 norm of each vote vector to find which is the longest.
values, idx = u_c.max(dim=1) 
c_zero = torch.zeros_like(u_c)
c_ones = torch.ones_like(u_c)
c_sparse = c_zero.scatter(1, idx.unsqueeze(dim=1), c_ones) # Sparse patrix has zeros everywhere but the winning child capsule (for each parent capsule).
v = (c_sparse.unsqueeze(dim=3) * u).sum(dim=1)