from catboost import CatBoostClassifier
import numpy as np
from sklearn.base import BaseEstimator

class CatBoostClassifierWrapper(BaseEstimator):
	"""A wrapper for CatBoost"""
	def __init__(self, **kwargs):
		self.args = kwargs
		self.model = CatBoostClassifier()

	def fit(self, X, Y, **kwargs):
		if(len(np.unique(Y)) > 2):
			if("loss_function" in self.args.keys()):
				del self.args["loss_function"]
			self.model = CatBoostClassifier(loss_function="MultiClass", **self.args)
		else:
			self.model = CatBoostClassifier(**self.args)

		self.model.fit(X, Y, **kwargs)

	def predict(self, X):
		return self.model.predict(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)

	def get_params(self, **kwargs):
		return self.model.get_params(**kwargs)

	def set_params(self, **kwargs):
		return self.model.set_params(**kwargs)

