# /usr/bin/env python
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib

def svm_trainer():
    X = np.array([[-1, -1],
                  [-2, -1],
                  [1, 1],
                  [2, 1]])
    y = np.array([1, 1, 2, 2])
    clf = SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, degree=3,
              gamma='auto', kernel='linear', max_iter=-1, probability=True,
              random_state=None, shrinking=True, tol=0.001, verbose=False)
    clf.fit(X, y)
    joblib.dump(clf, 'svm.pkl')
    print len(X)
    print len(y)
    
    
def svm_predictor(fvector):
    fvector = np.reshape(fvector,-1,1)
    clf = joblib.load('svm.pkl')
    #response = clf.predict(fvector)
    response = clf.predict_proba(fvector)
    clf.classes_
    print response
    print clf.predict_log_proba

def svm_option(opt, fvect=[]):
    if opt == 'TRAIN':
        svm_trainer()
        return 0
    elif opt == 'PREDICT':
        return svm_predictor(fvect)

def numpy_array_convertor(fvect, lvect):
    print 'num'
    

if __name__ == "__main__":
    svm_option('TRAIN')
    fvector = np.array(([1.8, -1]), dtype=float)
    sample = np.reshape(fvector, (1,-1))
    svm_option('PREDICT', sample)
