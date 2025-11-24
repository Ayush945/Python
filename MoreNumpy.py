import numpy as np

flips=np.random.choice([0,1],size=1000)
print("P(heads)=",flips.mean())

users=np.array([
    [5,4,0,0],
    [3,0,0,2],
    [4,4,3,2]
])

similarity=users@users.T
print(similarity)