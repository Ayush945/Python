import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

points={"blue":[[2,4],[1,3],[2,3],[3,2],[2,1]],
        "red":[[5,6],[4,5],[4,6],[6,6],[5,4]]}

new_point=[3,3]

def euclidean_distance(p,q, metric ="euclidean",p_norm=3):
    p,q=np.array(p),np.array(q)

    if metric=="euclidean":
        return np.sqrt(np.sum((p-q)**2))
    
    if metric=="manhattan":
        return np.sum(np.abs(p-q))
    
    if metric=="minkowski":
        return np.sum(np.abs(p-q)**p_norm)**(1/p_norm)

class KNN:
    def __init__ (self, k=3, metric="euclidean"):
        self.k=k
        self.point=None
        self.metric=metric
    
    def fit(self,points):
        self.points=points
    
    def predict(self,new_point):
        distances=[]
        for category in self.points:
            for point in self.points[category]:
                distance=euclidean_distance(point,new_point,metric=self.metric)
                distances.append([distance,category])

        categories=[category[1] for category in sorted(distances)[:self.k]]
        result=Counter(categories).most_common(1)[0][0]
        return result
    
# ---- HYPERPARAMETER TESTING ----

k_values = [1, 3, 5, 7, 9]
metrics = ["euclidean", "manhattan", "minkowski"]

print("\nHYPERPARAMETER TESTING RESULTS:\n")

for metric in metrics:
    print(f"--- Metric = {metric} ---")
    for k in k_values:
        clf = KNN(k=k, metric=metric)
        clf.fit(points)
        pred = clf.predict(new_point)
        print(f"k={k} → prediction={pred}")
    print()


def add_origin_distance(data_dict):
    new_data = {}
    for label in data_dict:
        new_data[label] = []
        for (x, y) in data_dict[label]:
            dist = np.sqrt(x**2 + y**2)
            new_data[label].append([x, y, dist])
    return new_data

# Apply feature engineering
fe_points = add_origin_distance(points)

# Transform new point
new_x, new_y = new_point
fe_new_point = [new_x, new_y, np.sqrt(new_x**2 + new_y**2)]

print("\nFEATURE ENGINEERING TEST (added distance-from-origin):\n")
for k in [1, 3, 5]:
    clf = KNN(k=k)
    clf.fit(fe_points)
    print(f"k={k} → prediction={clf.predict(fe_new_point)}")