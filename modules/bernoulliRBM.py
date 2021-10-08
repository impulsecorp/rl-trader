from sklearn.neural_network import BernoulliRBM
from sklearn import linear_model
from sklearn.pipeline import Pipeline

class SKBernoulliRBM():
    def __init__(self):
        self.rbm = BernoulliRBM(random_state=0, verbose=True)
        self.logistic = linear_model.LogisticRegression()
        self.classifier = Pipeline(steps=[('rbm', self.rbm), ('logistic', self.logistic)])
    def fit(self,X,y):
        

        # #############################################################################
        # Training

        # Hyper-parameters. These were set by cross-validation,
        # using a GridSearchCV. Here we are not performing cross-validation to
        # save time.
        self.rbm.learning_rate = 0.06
        self.rbm.n_iter = 20
        # More components tend to give better prediction performance, but larger
        # fitting time
        self.rbm.n_components = 2
        logistic.C = 6000.0

        # Training RBM-Logistic Pipeline
        self.rbm.fit(X, Y)
        return self.rbm
    
    def predict(self,X):
        
        return self.rbm.predict(X)
    
    def predict_proba(self,X):
        
        return 1/(1-np.exp(-self.rbm.predict(X)))
        