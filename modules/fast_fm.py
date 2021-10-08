from fastFM import sgd

class FastFM():
	def __init__(self):
		self.model= sgd.FMClassification(n_iter=1000, init_stdev=0.1, rank=10, l2_reg_w=0.1, l2_reg_V=0.5,step_size=0.1)

	def fit(self,X_train, y_train):
		return self.model.fit(X_train, y_train)
	def predict_proba(self,X_test):
		return self.model.predict_proba(X_test)
	def predict(self,X_test):
		return self.model.predict(X_test)
