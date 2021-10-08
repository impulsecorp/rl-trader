from modules.defragTrees import DefragModel
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

class DefragTreesClassifier():

	def __init__(self):
		pass

	def fit(self,X_train,y_train):
		forest = GradientBoostingClassifier(min_samples_leaf=10)
		forest.fit(X_train, y_train)

		Kmax = 10
		splitter = DefragModel.parseSLtrees(forest) # parse sklearn tree ensembles into the array of (feature index, threshold)
		self.model = DefragModel(modeltype='classification', maxitr=100, qitr=0, tol=1e-6, restart=20, verbose=0)
		self.model.fit(X_train, y_train, splitter, Kmax, fittype='FAB')

	def predict(self,X_test):
		return self.model.predict(X_test)

	def predict_proba(self,X_test):

		self.model.predict_proba(X_test)
