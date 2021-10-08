from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import keras.backend as K
import numpy as np

class FeatureLearning(object):
	def __init__(self, layer_sizes, input_shape, mode="binary_classifier"):
		model = Sequential()
		model.add(Dense(layer_sizes[0], input_shape=input_shape))
		model.add(Dropout(0.3))
		model.add(BatchNormalization())
		for i in range(1, len(layer_sizes) - 1):
			model.add(Dense(layer_sizes[i]))
			model.add(Dropout(0.3))
			model.add(BatchNormalization())

		self.transform_fn = K.function([model.input]+ [K.learning_phase()], [model.layers[-3].output])
		
		if(mode == "binary_classifier"):
			model.add(Dense(layer_sizes[-1], activation='sigmoid'))
			model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
		elif(mode == "multi_classifier"):
			model.add(Dense(layer_sizes[-1], activation='softmax'))
			model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
		else:
			model.add(Dense(layer_sizes[-1], activation='linear'))
			model.compile(loss="mean_squared_error", optimizer="adam", metrics=["mean_absolute_error", "mean_absolute_percentage_error"])

		self.model = model

	def fit_transform(self, X, Y):
		if(not isinstance(Y, np.ndarray)):
			Y = np.array(Y)
		if(not isinstance(X, np.ndarray)):
			X = np.array(X)

		fl_x, val_x, fl_y, val_y = train_test_split(X, Y)
		self.model.fit(fl_x, fl_y, validation_data=[val_x, val_y], callbacks=[EarlyStopping(patience=4), TerminateOnNaN(), ReduceLROnPlateau(patience=2)], epochs=1000)
		return self.transform_fn([X, 0])

	def transform(self, X):
		return self.transform_fn([X, 0])