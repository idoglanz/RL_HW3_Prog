import numpy as np

data = np.ones((5, 4, 2))
print(data)

vector = np.zeros((4, 1))

print(vector)

product = np.tensordot(data, vector, axes=([1, 0], [1, 0]))

print(product)