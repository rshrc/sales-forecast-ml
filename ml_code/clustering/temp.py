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
    
    
## Multiple Linear Regression
dataset = pd.read_csv('ml_code/clustering/dataset.csv')
# Splitting the dataset into the Training set and Test set
X = dataset.iloc[:, :-1].values  # This is the parameters column
y = dataset.iloc[:, 8].values    # This is the Sales Column
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling the Dataset
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)

# Fitting the Multiple Linear Regression in the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the test set results
y_pred = regressor.predict(X_test)

y_pred = np.array(y_pred)
y_test = np.array(y_test)

# Accuracy Testing
from sklearn.metrics import r2_score
acuracy = r2_score(y_test, y_pred) # 82% accuracy

# Building the optimal model using backward elimination
import statsmodels.formula.api as sm  
X = np.append(arr = np.ones((507,8)).astype(float), values = X, axis = 1)
X_opt = X[:, [0,1,2,3,4,5,6,7]] # fitting all the possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,1,3,4,5]] # fremoving 2 for 0.909
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,4,5]] # removing 1 for 0.940
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3,5]] # fitting all the possible predictors
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0,3]] # Thus R&D spend is the most powerfull predictor.
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()