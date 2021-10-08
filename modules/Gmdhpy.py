

from gmdhpy.gmdh import Classifier


class GMDH():
	"""Multilayered iterational algorithm of the Group Method of Data Handling """

	def __init__(self):
		pass

	def fit(self,X_train,y_train):
		self.model = Classifier()
		self.model.fit(X_train, y_train)
	def predict(self,X_test):
		return self.model.predict(X_test)

	def predict_proba(self,X_test):

		return self.model.predict_proba(test_x)
