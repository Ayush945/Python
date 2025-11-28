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
    
def plot_knn_decision_boundary(clf, points):
    # Create grid
    xx, yy = np.meshgrid(np.linspace(0,7,200), np.linspace(0,7,200))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # Predict for each grid point
    Z = np.array([clf.predict(p) for p in grid])
    Z = Z.reshape(xx.shape)

    # Plot boundary
    plt.contourf(xx, yy, Z == "red", alpha=0.3, cmap='coolwarm')

    # Plot training points
    for color in points:
        pts = np.array(points[color])
        plt.scatter(pts[:,0], pts[:,1], label=color)

    plt.legend()
    plt.show()

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
    
def minmax_scale(array):
    arr=np.array(array)
    mn=arr.min(axis=0) # lowest from each column
    mx=arr.max(axis=0)
    return (arr-mn)/(mx-mn),mn,mx

all_points = []
labels = []

for label in points:
    for p in points[label]:
        all_points.append(p)
        labels.append(label)

scaled_array, mn, mx = minmax_scale(all_points)

scaled_points = {"blue": [], "red": []}
idx = 0
for label in labels:
    scaled_points[label].append(list(scaled_array[idx]))
    idx += 1

# scale new_point too
new_point_scaled = (np.array(new_point) - mn) / (mx - mn)

clf=KNN(k=3)
clf.fit(points)
print(clf.predict(new_point))


clf_scaled = KNN(k=3)
clf_scaled.fit(scaled_points)
print("Scaled prediction:", clf_scaled.predict(new_point_scaled))

plot_knn_decision_boundary(clf, points)
