#!/usr/bin/env python

from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

import numpy as np
import cPickle
import math

#iris = datasets.load_iris()
iris = None
with open('my_dumped_classifier.pkl', 'rb') as fid:
    iris = cPickle.load(fid)

gnb1 = GaussianNB()
gnb2 = GaussianNB()
gnb3 = GaussianNB()
gnb4 = GaussianNB()

#target = np.where(iris.target, 2, 1) # 2 class
target = iris.target

gnb1.fit(iris.data[:, 0].reshape(150,1), target)
gnb2.fit(iris.data[:, 1].reshape(150,1), target)
gnb3.fit(iris.data[:, 2].reshape(150,1), target)
gnb4.fit(iris.data[:, 3].reshape(150,1), target)

#y_pred = gnb.predict(iris.data)
index = 100
y_prob1 = gnb1.predict_proba(iris.data[index,0].reshape(1,1))
y_prob2 = gnb2.predict_proba(iris.data[index,1].reshape(1,1))
y_prob3 = gnb3.predict_proba(iris.data[index,2].reshape(1,1))
y_prob4 = gnb4.predict_proba(iris.data[index,3].reshape(1,1))

# with open('my_dumped_classifier.pkl', 'wb') as fid:
#     cPickle.dump(iris, fid)

#print y_prob1, "\n", y_prob2, "\n", y_prob3, "\n", y_prob4 , "\n"

#print y_prob1 + y_prob2 + y_prob3 + y_prob4

priori = 0.5
prob1 = math.log(y_prob1[:,1] * priori/y_prob1[:,0] * priori, 2)
prob2 = math.log(y_prob2[:,1] * priori/y_prob2[:,0] * priori, 2)
prob3 = math.log(y_prob3[:,1] * priori/y_prob3[:,0] * priori, 2)
prob4 = math.log(y_prob4[:,1] * priori/y_prob4[:,0] * priori, 2)

# pos = y_prob1[:,1] + y_prob2[:,1] + y_prob3[:,1] + y_prob4[:,1]
# neg = y_prob1[:,0] + y_prob2[:,0] + y_prob3[:,0] + y_prob4[:,0]
# print pos
# print neg
                                                           
#y_prob = prob1 * prob2 * prob3 * prob4
#print "\n", y_prob, "\n"


gnb = GaussianNB()
gnb.fit(iris.data, iris.target)



#y_prob = np.zeros(shape=(1,3), dtype="float")
y_prob = gnb.predict_proba(iris.data[45:55])

y_prob = np.around(y_prob, 3)
print y_prob , "\n"

print np.sum(y_prob, axis=0)
