from modules.researchboosting import ResearchGradientBoostingBase, InfiniteBoosting, InfiniteBoostingWithHoldout
from modules.researchlosses import MSELoss, LogisticLoss, LambdaLossNDCG, compute_lambdas_numpy, compute_lambdas
from modules.researchtree import BinTransformer, build_decision_numpy, build_decision
import numpy as np

class ResearchGradientBoostingBaseCLF(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        X = BinTransformer().fit_transform(X)

        self.clf = ResearchGradientBoostingBase(l2_regularization=1.0, learning_rate=0.1,loss=MSELoss(), max_depth=3, max_features=1.0,n_estimators=100)
        self.num_classes = len(np.unique(y))
        self.clf.fit(X, y)
    def predict(self,X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        return self.clf.decision_function(X_Test).astype('int32')
    
    def predict_proba(self,X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        classes = self.clf.decision_function(X_Test).astype('int32').reshape(-1)
        return np.eye(self.num_classes)[classes]
    
        

class InfiniteBoostingWithHoldoutCLF(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        X = BinTransformer().fit_transform(X)
        self.num_classes = len(np.unique(y))
        self.clf = InfiniteBoostingWithHoldout(l2_regularization=1.0,loss=MSELoss(),use_all_in_update=True, max_depth=5,n_estimators=100)
        self.clf.fit(X, y)
    def predict(self,X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        return self.clf.decision_function(X_Test).astype('int32')
 
    def predict_proba(self, X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        classes = self.clf.decision_function(X_Test).astype('int32')
        return np.eye(self.num_classes)[classes]


class InfiniteBoostingCLF(object):
    def __init__(self):
        pass
    
    def fit(self,X,y):
        X = BinTransformer().fit_transform(X)
        self.num_classes = len(np.unique(y))
        self.clf = InfiniteBoosting(capacity=100,l2_regularization=1.0, learning_rate=0.1,loss=MSELoss(), max_depth=3,n_estimators=100)
        self.clf.fit(X, y)
    
    def predict(self,X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        return self.clf.decision_function(X_Test).astype('int32')
    
    def predict_proba(self, X_Test):
        X_Test = BinTransformer().fit_transform(X_Test)
        classes = self.clf.decision_function(X_Test).astype('int32')
        return np.eye(self.num_classes)[classes]
