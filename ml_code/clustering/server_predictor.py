import pickle

from sklearn.externals import joblib

classifier_model = joblib.load('ml_code/classifier_model.pkl')
logistic_regression_model = joblib.load('ml_code/logistic_regression_model.pkl')
linear_regression_model = joblib.load('ml_code/linear_regression_model.pkl')

clf = pickle.loads(classifier_model)
logreg = pickle.loads(logistic_regression_model)
lreg = pickle.loads(linear_regression_model)


def get_prediction(back_camera, front_camera, resolution_1, resolution_2, screen_size, battery, price,
                   ):
    y_pred = logreg.predict([[back_camera, front_camera, resolution_1, resolution_2, screen_size, battery,
                            price]])
    return y_pred
