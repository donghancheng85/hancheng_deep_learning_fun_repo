import torch

# positional min max, useful for softmax function
x = torch.arange(0, 100, 10)
print(f"origin tensor x = {x}")

# find the position in tensor that has the min value
min_index_of_x = x.argmin()
print(f"index of min in x is {min_index_of_x}")

x_1 = torch.arange(1, 100, 10)
print(f"origin tensor x_1 = {x_1}")

min_index_of_x_1 = x_1.argmin()
print(f"index of min in x_1 is {min_index_of_x_1}")

# find the position of max value
max_index_of_x = x.argmax()
print(f"index of max value in x is {max_index_of_x}")
