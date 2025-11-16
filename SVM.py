from sklearn import svm

#data points
X=[
[0.1,2.3],
[-1.5,2.5],
[2.0,-4.3],
]

#labels
y=[0,1,0,]

# nonlinear transformation 
fx=[(x[0],x[1],x[0]**2+x[1]**2) for x in X]


#fit
svm.SVC().fit(fx,y)