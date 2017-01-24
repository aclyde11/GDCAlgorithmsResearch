# Standard Imports
import os
import csv
import dill
import collections
import subprocess
import numpy as np

# C mut info
import ctypes
from numpy.ctypeslib import ndpointer
lib = ctypes.cdll.LoadLibrary("include/libMIToolbox.so")

_doublepp = ndpointer(dtype=np.uintp, ndim=1, flags='C_CONTIGUOUS') 
_single   = ndpointer(dtype=np.uint32, flags="C_CONTIGUOUS")

mutInfo          = lib.calcMutualInformation
mutInfo.restype  = ctypes.c_double
mutInfo.argtypes = [_single, _single, ctypes.c_int32]

condInfo          = lib.calcConditionalMutualInformation
condInfo.restype  = ctypes.c_double
condInfo.argtypes = [_single, _single, _single, ctypes.c_int32]


# Data Science
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing

# Machine Learning
from sklearn.feature_selection import SelectFromModel, chi2, f_classif, mutual_info_classif
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron

def jointMI(i, j, X, y):
    return(float(condInfo(X[:,i], y, X[:,j], y.size) + mutInfo(X[:,j], y, y.size)))

def minJointMI(f, Sset, X, y):
    func = np.vectorize(lambda p: jointMI(f,p, X, y))
    return(np.min(func(Sset)))

def mutInfoP(x,y):
    return(float(mutInfo(x,y, y.size)))

'''
Wrapper to hold the dataset and run a variety of feature selectors on.
'''
class FeatureSelection(object):

    def __init__(self, dataset):
        print("***** Feature Selection Object Created *****")
        self.df = dataset
        self.initializeDataset()

    def jmimForwardSearch(self, k, n_jobs=1):
        Fset = np.array(list(range(self.X.shape[1])))
        #find top MI
        scores = Parallel(n_jobs=n_jobs)(delayed(mutInfoP)(f, self.y) for f in self.X.T)
        Sset = np.array(Fset[np.argmax(scores)], dtype=Fset.dtype)
        Fset = np.delete(Fset, np.argwhere(Fset==Sset))
        print("past first select")
        print(Sset)
        #add features to it
        while(k > 1):
            scores = Parallel(n_jobs=n_jobs)(delayed(minJointMI)(f, Sset, self.X, self.y) for f in Fset)
            fStar = Fset[np.argmax(scores)]
            Sset  = np.append(Sset, fStar)
            Fset = np.delete(Fset, np.argwhere(Fset==Sset[-1]))
            k = k - 1
            print(fStar)
        return(Sset)

    def initializeDataset(self):
        print("***** Initializing Dataset *****")
        self.X    = np.array(self.df.drop('y', axis=1), dtype=np.uint32)
        le = preprocessing.LabelEncoder()
        le.fit(self.df['y'].value_counts().index)
        self.y  = le.transform(np.array(self.df['y'])) # labels
        self.y  = self.y.astype(np.uint32)
        self.fnames = np.array(self.df.drop('y', axis=1).columns.values)


def main():
    pickleLoc = "/space3/clyde/testPickle.pkl"
    print("***** Script Started *****")
    df = pd.read_csv("~/tester.csv")
    fs = FeatureSelection(df)
    print(fs.jmimForwardSearch(2, n_jobs=1))


if  __name__ =='__main__':
    main()