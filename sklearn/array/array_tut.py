#!/usr/bin/env python

import numpy
import numpy as np
import rospy

from sklearn.svm import SVC
from sklearn.externals import joblib

def object_classifier_train(fvect, lvect):
    clf = SVC(C=1.0, cache_size=200, class_weight=None,
              coef0=0.0, degree=3, gamma=0.0, kernel='linear',
              max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.001, verbose=False)
    clf.fit(fvect, lvect)
    joblib.dump(clf, 'svm.pkl')
    

def object_classifier_trainer_handler(req):
    cols = len(req)
    rows = 3
    print cols

    #req = np.arange(9).reshape(3, 3)
    req = np.array(req)
    b = req.reshape(-1, 5)
    print b[0]
    print b[1]
    print b.shape[0]
    #req = np.array(req)
    #print req
    
    #fvect = object_classifier_construct_data(req, rows, cols);
    #lvect = convert_to_numpy_array(req.labels)


if __name__ == "__main__":
    print 'In Main...'
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = [10, 20, 4, 5, 6, 7, 10,60, 7, 122]
    object_classifier_trainer_handler(a)
