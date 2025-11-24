import numpy as np

a=np.array([1,2,3,4,5])
b=np.zeros((2,3))
c=np.ones((2,5))
d=np.random.randn(4)

# print(a)
# print(b)
# print(c)
# print(d)

# print(a[1])
# print(a[1:4])

b=a*2
c=a+5
d=a**2

# print(a)
# print(b)
# print(c)
# print(d)

# print(a.sum())
# print(a.mean())
# print(a.max())
# print(a.min())

mat=np.arange(6).reshape(1,6)
# print(mat)

a=np.array([[1,2],[3,4]])
b=np.array([[5,6],[7,8]])

c=a@b
print(c)