import torch
import numpy
import time

torch_tensor1 = torch.rand(1000, 1000).to('cuda')
torch_tensor2 = torch.rand(1000, 1000).to('cuda')

np_array1 = numpy.ndarray((1000, 1000))
np_array2 = numpy.ndarray((1000, 1000))

torch_tik = time.time()
torch_tensor_res = torch_tensor1 @ torch_tensor2.T
torch_tok = time.time()

np_tik = time.time()
np_tensor_res = np_array1 @ np_array2.T
np_tok = time.time()

print(f"Time with cpu: {np_tok - np_tik} \n \
Time with gpu: {torch_tok - torch_tik}")