from pomegranate import *
import numpy as np

class MC(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.model = MarkovChain(MultivariateGaussianDistribution, n_components=len(np.unique(y)), X=X)
        self.model.fit(X)
    def predict(self,X_Test):
        preds=self.model.predict(X_Test)
        return preds
    def predict_proba(self,X_Test):
        preds_proba=self.model.predict_proba(X_Test)
        return preds_proba