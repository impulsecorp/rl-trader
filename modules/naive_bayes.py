from pomegranate import *

class NB(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.model = NaiveBayes.from_samples(NormalDistribution, X, y)
        #self.model.fit(X,y,verbose=False)
    def predict(self,X_Test):
        preds=self.model.predict(X_Test)
        return preds
    def predict_proba(self,X_Test):
        preds_proba=self.model.predict_proba(X_Test)
        return preds_proba
        
