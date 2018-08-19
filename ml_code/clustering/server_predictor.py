from sklearn.externals import joblib
import pickle


classifier_model = joblib.load('classifier_model.pkl')
regression_model = joblib.load('regression_model.pkl')

clf = pickle.loads(classifier_model)
reg = pickle.loads(regression_model)

def get_prediction():
    pass