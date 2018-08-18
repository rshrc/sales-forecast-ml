from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('ggplot')

data=pd.read_csv('ml_code/clustering/Final2.csv')
print(data.shape)

f1=data['back_camera'].values
f2=data['front_camera'].values
f3=data['resolution_1'].values
f4=data['resolution_2'].values
f5=data['screen_size'].values
f6=data['battery'].values
f7=data['price'].values
f8=data['sales'].values
f9=data['quarter'].values
X=np.array(list(zip(f1,f2,f3,f4,f5,f6,f7,f8,f9)))
#print(X)
#plt.scatter(f8,f7)

kmeans=KMeans(n_clusters=2)
kmeans=kmeans.fit(X)
labels=kmeans.predict(X)
C=kmeans.cluster_centers_
print(C)

colors=['r','g','b']
fig, ax = plt.subplots()
for i in range(3):
   points=np.array([X[j] for j in range(len(X)) if labels[j]==i])
   ax.scatter(points[:,0],points[:,1],s=7,c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')
           
# Test Data Dumping
import pickle
# now you can save it to a file
with open('predictor_cluster', 'wb') as f:
    pickle.dump(kmeans, f)
    
    
## Logistic Regression

