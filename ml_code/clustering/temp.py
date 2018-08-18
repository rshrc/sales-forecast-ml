from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('ggplot')

data=pd.read_csv('ml_code/clustering/dataset.csv')
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
    
    
## Multiple Linear Regression
dataset = pd.read_csv('ml_code/clustering/dataset.csv')
# Splitting the dataset into the Training set and Test set
X = dataset.iloc[:, :-1].values  # This is the parameters column
y = dataset.iloc[:, 8].values    # This is the Sales Column
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression in the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

# Accuracy Testing of Linear Regression
from sklearn.metrics import r2_score
linear_accuracy = r2_score(y_test, y_pred) # 99% accuracy


# Fitting the Random Forest in the Training Set
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)
rf_accuracy = r2_score(y_test, y_pred) # 99% accuracy


