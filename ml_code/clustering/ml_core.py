from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize']=(16,9)
plt.style.use('ggplot')

data = pd.read_csv('ml_code/clustering/dataset.csv')

f1 = data['back_camera'].values
f2 = data['front_camera'].values
f3 = data['resolution_1'].values
f4 = data['resolution_2'].values
f5 = data['screen_size'].values
f6 = data['battery'].values
f7 = data['price'].values
f8 = data['sales'].values
f9 = data['quarter'].values
X = np.array(list(zip(f1,f2,f3,f4,f5,f6,f7,f8,f9)))

clf = KMeans(init='k-means++', n_clusters=4, n_init=10)
clf = clf.fit(X)
labels = clf.predict(X)
C = clf.cluster_centers_
print(C)        
colors = ['r','g','b','y']
fig,ax = plt.subplots()
for i in range(4):
    points=np.array([X[j] for j in range(len(X)) if labels[j]==i])
    ax.scatter(points[:,8],points[:,7],s=7,c=colors[i])
ax.scatter(C[:, 8], C[:, 7], marker='*', s=300, c='#050505')                   
print(metrics.silhouette_score(X, labels, metric='euclidean'))        
y = clf.labels_
#print(y)           
data['y']= y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)        
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
print('Accuracy of logistic regression on training set: {:.2f}'.format(logreg.score(X_train, y_train)))
print('Accuracy of logistic regression on test set: {:.2f}'.format(logreg.score(X_test, y_test)))        
kfold = model_selection.KFold(n_splits=10, random_state=7)
results = model_selection.cross_val_score(logreg, X_train, y_train, cv = kfold, scoring = "accuracy")
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))       
X_new = [[5, 1.2, 1536, 2048, 9.7, 11560, 399, 1611, 2]]        
y_pred= logreg.predict(X_new)
t1 = data.loc[data['y'] == y_pred[0]]

# Saving the Logistic Regression Model
import pickle
classifier_model = pickle.dumps(clf)
regression_model = pickle.dumps(logreg)

# Saving the model to a file
from sklearn.externals import joblib
joblib.dump(clf, 'classifier_model.pkl')
joblib.dump(logreg, 'regression_model.pkl')