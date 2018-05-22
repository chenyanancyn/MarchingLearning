# a = [1, 2, 2, 3, 4, 5 , 6,6, 6,6, 7, 7, 7]
# b = set(a)
# c = max(a)
# d = max(a, key=a.count)
# e = max(b, key=a.count)
# # print(e)
# print(round(0.51))

import numpy as np
a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
a = np.array(a)
print(a[2])
print(a[:, 2])