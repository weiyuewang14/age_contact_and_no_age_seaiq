import numpy as np

a = [[1, 3, 4, 32, 4], [2, 9, 8, 16, 18]]
a.append([3,4,5,6,7,7])
print(a)
a = np.array(a)
print(a.T)
