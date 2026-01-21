import torch

# rules need to be satisfied for matirx multiple
# Note: shape error is one of the common errors in Deep Learning

"""
Two main rules for matrix muliplication:
1. The inner dimensions must match
(3, 2) @ (4, 4) bad
(2, 3) @ (3, 2) good

2. The resulting matrix has the shape of outer dimensions
(2, 3) @ (3, 2) -> (2, 2)
"""

MATRIX_1 = torch.rand(size=[3, 2], dtype=torch.float32)
MATRIX_2 = torch.rand(size=[2, 4], dtype=torch.float32)

MATRIX_MULTILPILE_RESULT = torch.matmul(MATRIX_1, MATRIX_2)

print(f"Two matrix multiple result = \n {MATRIX_MULTILPILE_RESULT}")
print(f"Two matrix multiple shape is {MATRIX_MULTILPILE_RESULT.shape}")

MATRIX_3 = torch.rand(size=[3, 4], dtype=torch.float32)
# will not work, and this is common error in Deep Learning
# torch.matmul(MATRIX_1, MATRIX_3)
