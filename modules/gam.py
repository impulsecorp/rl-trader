from pygam import LogisticGAM


class GAMCLF(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        self.clf=  LogisticGAM().gridsearch(X, y)
        self.clf.fit(X,y)
    def predict(self,X_test):
        return self.clf.predict(X_test)
    
    def predict_proba(self,X_test):
         return self.clf.predict_proba(X_test)
        
        
        