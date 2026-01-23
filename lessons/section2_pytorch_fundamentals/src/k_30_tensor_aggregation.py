import torch

"""
Tensor Aggregation:
min, max, sum, etc
"""

# create a tensor
x = torch.arange(0, 100, 10)
print(f"origin tensor x = {x}")

# min
min_a = torch.min(x)  # buildin
print(f"(torch buidin function) min in tensor x is {min_a}")
min_b = x.min()  # class method
print(f"(Tensor class method) min in tensor x is {min_b}")
print(f"Type of min value is still tensor -- {type(min_a)}")

# max
max_a = torch.max(x)  # buildin
print(f"(torch buidin function) max in tensor x is {max_a}")
max_b = x.max()  # class method
print(f"(Tensor class method) max in tensor x is {max_b}")

# Find mean (note: mean can only deal with torch.floatXX type, int type will not work)
# use type() method of Tensor class to cast type of tensor
mean_a = torch.mean(x.type(torch.float32))
print(f"(torch buildin function) mean of tensor x is {mean_a}")

mean_b = x.type(torch.float32).mean()
print(f"(Tensor class method) mean of tensor x is {mean_b}")

# sum
sum_a = torch.sum(x)
print(f"(torch buildin function) sum of tensor x is {sum_a}")

sum_b = x.sum()
print(f"(Tensor class method) sum of tensor x is {sum_b}")
