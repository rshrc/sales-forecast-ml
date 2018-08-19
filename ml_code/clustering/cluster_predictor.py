import pickle
# now you can save it to a file


# and later you can load it
with open('predictor_cluster', 'rb') as f:
    clf = pickle.load(f)

something = clf.predict(12)
