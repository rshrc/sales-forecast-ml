import pickle

from sklearn.externals import joblib

classifier_model = joblib.load('ml_code/classifier_model.pkl')
regression_model = joblib.load('ml_code/regression_model.pkl')

clf = pickle.loads(classifier_model)
reg = pickle.loads(regression_model)


def get_prediction(back_camera, front_camera, resolution_1, resolution_2, screen_size, battery, price,
                   pre_release_demand, sales, quarter):
    classifier_prediction = clf.predict(back_camera, front_camera, resolution_1, resolution_2, screen_size, battery,
                                        price,
                                        pre_release_demand, sales, quarter)
    regression_prediction = reg.predict(back_camera, front_camera, resolution_1, resolution_2, screen_size, battery,
                                   price,
                                   pre_release_demand, sales, quarter)
