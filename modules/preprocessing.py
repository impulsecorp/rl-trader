"""
This class runs the data through preprocessing before being used in a model.
List of available encodings:
* LabelEncoder
* Hashing
* Binary
* Helmert
"""

from datacleaner import autoclean
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datacleaner import autoclean
from category_encoders.binary import BinaryEncoder
from category_encoders.hashing import HashingEncoder
from category_encoders.helmert import HelmertEncoder
from category_encoders.backward_difference import BackwardDifferenceEncoder

class Preprocessor(object):
	"""Preprocess the data"""
	def preprocess_data(self, data):
		if("pandas" not in str(type(data))):
			data = pd.DataFrame(data)
		
		data = data.fillna(data.median()).apply(lambda l: l.fillna(l.value_counts().index[0]))


		schema = {}
		encoders = {}
		for col in data.columns:
			schema[col] = data[col].dtype

		for col in data.columns:
			if(data[col].dtype == "object"):
				le = self.EncoderFactory()
				data[col] = le.fit_transform(data[col].values)
				encoders[col] = le

		self.encoders = encoders
		return data

	def preprocess_x_y(self, X, Y):
		if("pandas" not in str(type(X))):
			X = pd.DataFrame(X)
	
		X = X.fillna(X.median()).apply(lambda l: l.fillna(l.value_counts().index[0]))


		encoders = {}
		for col in X.columns:
			if(X[col].dtype == "object"):
				le = self.EncoderFactory()
				X[col] = le.fit_transform(X[col].values)
				encoders[col] = le

		if(Y.dtype == "object"):
			le = self.EncoderFactory()
			Y = le.fit_transform(Y)
			self.encoders = encoders
			self.le = le
			return X, Y

		self.encoders = encoders
		return X, Y

	def preprocess_test_data(self, data):
		if("pandas" not in str(type(data))):
			data = pd.DataFrame(data)
		
		data = data.fillna(data.median()).apply(lambda l: l.fillna(l.value_counts().index[0]))

		for col in self.encoders.keys():
			try:
				data[col] = self.encoders[col].transform(data[col].values)
			except Exception as e:
				print(e)
				continue

		return data

	def preprocess_test_x_y(self, X, Y=[]):
		if("pandas" not in str(type(X))):
			X = pd.DataFrame(X)
		
		X = X.fillna(X.median()).apply(lambda l: l.fillna(l.value_counts().index[0]))

		for col in self.encoders.keys():
			X[col] = self.encoders[col].transform(X[col].values)

		if(len(Y) > 0 and self.le is not None):
			Y = self.le.transform(Y)
			return X, Y

		return X

	def preprocess_dataset(self, dataset):
		return autoclean(dataset)

	def __init__(self, encoding="LabelEncoder"):
		self.encoding = encoding

	def EncoderFactory(self):
		if(self.encoding == "LabelEncoder"):
			return LabelEncoder()
		elif(self.encoding == "Binary"):
			return BinaryEncoder()
		elif(self.encoding == "Hashing"):
			return HashingEncoder()
		elif(self.encoding == "Helmert"):
			return HelmertEncoder()
		elif(self.encoding == "Backward"):
			return BackwardDifferenceEncoder()
