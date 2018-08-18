import pickle
# now you can save it to a file


# and later you can load it
with open('predictor_cluster', 'rb') as f:
    clf = pickle.load(f)# -*- coding: utf-8 -*-

something = clf.predict(12)
