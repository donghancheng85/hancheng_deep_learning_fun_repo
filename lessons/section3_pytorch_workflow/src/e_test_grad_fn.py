import torch

x = torch.tensor(2.0, requires_grad=True)
y = x**2 + 3 * x

print(y.grad_fn)  # <AddBackward0 object at 0x7f686e438610>

y.backward()
print(x.grad)  # Should be 2x + 3 = 7 -- dy/dx

"""
The above graph is like below
          x
         /  \
   (square) (mul by 3)
       \      /
         (add)
           |
           y

When backward() is called, y trace this graph backward, using chain rule
"""
