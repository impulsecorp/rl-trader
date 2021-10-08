import numpy as np
from keras.models import Sequential
from keras.layers.wrappers import Bidirectional
from keras.layers.recurrent import LSTM as Keras_LSTM
from keras.layers.recurrent import GRU as Keras_GRU
from keras.layers.recurrent import SimpleRNN as Keras_SimpleRNN
from keras.layers.core import Dense
from keras.layers import Input, merge, Activation
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import keras
from keras.models import Model

import keras
from sklearn.model_selection import train_test_split

class LSTM(object):
	"""Train an LSTM model on the data"""
	def __init__(self, input_shape, output_size, layers=1, units=32, activation='relu', bidirectional=False):
		self.layers = layers
		self.units = units
		self.activation = activation
		self.bidirectional = bidirectional
		self.input_shape = input_shape
		self.output_size = output_size

	def construct_model(self, optimizer="adam", loss='categorical_crossentropy'):
		self.model = Sequential()
		if(self.bidirectional):
			if(self.layers > 1):
				self.model.add(Bidirectional(Keras_LSTM(self.units,
					input_shape=self.input_shape, activation=self.activation, return_sequences=True)))
			else:
				self.model.add(Bidirectional(Keras_LSTM(self.units, input_shape=self.input_shape, activation=self.activation)))

		else:
			if(self.layers > 1):
				self.model.add(Keras_LSTM(self.units, input_shape=self.input_shape,
					activation=self.activation, return_sequences=True))
			else:
				self.model.add(Keras_LSTM(self.units, input_shape=self.input_shape, activation=self.activation))
		for i in range(self.layers - 1):
			if(i == self.layers - 2):
				self.model.add(Keras_LSTM(self.units))
			else:
				self.model.add(Keras_LSTM(self.units, return_sequences=True))
		self.model.add(Dense(self.output_size, activation='softmax'))
		self.model.compile(optimizer=optimizer, loss=loss,
			metrics=['accuracy'])

	def fit(self, X_Train, Y_Train, X_Test, Y_Test, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, test_size=0.1)
		self.model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate(X_Test, Y_Test, batch_size=batch_size))

	def fit_generator(self, generator, X_Val, Y_Val, steps_per_epoch, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		self.model.fit_generator(generator, steps_per_epoch, epochs, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate_generator(generator, steps=steps_per_epoch))

	def predict_classes(self, X):
		return self.model.predict_classes(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)
		
class GRU(object):
	"""Train an GRU model on the data"""
	def __init__(self, input_shape, output_size, layers=1, units=32, activation='relu', bidirectional=False):
		self.layers = layers
		self.units = units
		self.activation = activation
		self.bidirectional = bidirectional
		self.input_shape = input_shape
		self.output_size = output_size

	def construct_model(self, optimizer="adam", loss='categorical_crossentropy'):
		self.model = Sequential()
		if(self.bidirectional):
			if(self.layers > 1):
				self.model.add(Bidirectional(Keras_GRU(self.units,
					input_shape=self.input_shape, activation=self.activation, return_sequences=True)))
			else:
				self.model.add(Bidirectional(Keras_GRU(self.units, input_shape=self.input_shape, activation=self.activation)))

		else:
			if(self.layers > 1):
				self.model.add(Keras_GRU(self.units, input_shape=self.input_shape,
					activation=self.activation, return_sequences=True))
			else:
				self.model.add(Keras_GRU(self.units, input_shape=self.input_shape, activation=self.activation))
		for i in range(self.layers - 1):
			if(i == self.layers - 2):
				self.model.add(Keras_GRU(self.units))
			else:
				self.model.add(Keras_GRU(self.units, return_sequences=True))
		self.model.add(Dense(self.output_size, activation='softmax'))
		self.model.compile(optimizer=optimizer, loss=loss,
			metrics=['accuracy'])

	def fit(self, X_Train, Y_Train, X_Test, Y_Test, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, test_size=0.1)
		self.model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate(X_Test, Y_Test, batch_size=batch_size))

	def fit_generator(self, generator, X_Val, Y_Val, steps_per_epoch, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		self.model.fit_generator(generator, steps_per_epoch, epochs, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate_generator(generator, steps=steps_per_epoch))

	def predict_classes(self, X):
		return self.model.predict_classes(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)
				
class SimpleRNN(object):
	"""Train an SimpleRNN model on the data"""
	def __init__(self, input_shape, output_size, layers=1, units=32, activation='relu', bidirectional=False):
		self.layers = layers
		self.units = units
		self.activation = activation
		self.bidirectional = bidirectional
		self.input_shape = input_shape
		self.output_size = output_size

	def construct_model(self, optimizer="adam", loss='categorical_crossentropy'):
		self.model = Sequential()
		if(self.bidirectional):
			if(self.layers > 1):
				self.model.add(Bidirectional(Keras_SimpleRNN(self.units,
					input_shape=self.input_shape, activation=self.activation, return_sequences=True)))
			else:
				self.model.add(Bidirectional(Keras_SimpleRNN(self.units, input_shape=self.input_shape, activation=self.activation)))

		else:
			if(self.layers > 1):
				self.model.add(Keras_SimpleRNN(self.units, input_shape=self.input_shape,
					activation=self.activation, return_sequences=True))
			else:
				self.model.add(Keras_SimpleRNN(self.units, input_shape=self.input_shape, activation=self.activation))
		for i in range(self.layers - 1):
			if(i == self.layers - 2):
				self.model.add(Keras_SimpleRNN(self.units))
			else:
				self.model.add(Keras_SimpleRNN(self.units, return_sequences=True))
		self.model.add(Dense(self.output_size, activation='softmax'))
		self.model.compile(optimizer=optimizer, loss=loss,
			metrics=['accuracy'])

	def fit(self, X_Train, Y_Train, X_Test, Y_Test, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train, Y_Train, test_size=0.1)
		self.model.fit(X_Train, Y_Train, epochs=epochs, batch_size=batch_size, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate(X_Test, Y_Test, batch_size=batch_size))

	def fit_generator(self, generator, X_Val, Y_Val, steps_per_epoch, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		self.model.fit_generator(generator, steps_per_epoch, epochs, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate_generator(generator, steps=steps_per_epoch))

	def predict_classes(self, X):
		return self.model.predict_classes(X)

	def predict_proba(self, X):
		return self.model.predict_proba(X)
				
class ResNet(object):
	"""Train a Resnet model on the data"""
	def __init__(self, input_shape, output_size):
		self.input_shape = input_shape
		self.output_size = output_size

	def construct_model(self, n_feature_maps=64):
		x = Input(shape=self.input_shape)
		conv_x = keras.layers.normalization.BatchNormalization()(x)
		conv_x = keras.layers.Conv2D(n_feature_maps, (8, 1), border_mode='same')(conv_x)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = Activation('relu')(conv_x)

		print ('build conv_y')
		conv_y = keras.layers.Conv2D(n_feature_maps, 5, 1, border_mode='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = Activation('relu')(conv_y)

		print ('build conv_z')
		conv_z = keras.layers.Conv2D(n_feature_maps, 3, 1, border_mode='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		is_expand_channels = not (self.input_shape[-1] == n_feature_maps)
		if is_expand_channels:
			shortcut_y = keras.layers.Conv2D(n_feature_maps, 1, 1,border_mode='same')(x)
			shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
		else:
			shortcut_y = keras.layers.normalization.BatchNormalization()(x)
		print ('Merging skip connection')
		y = merge([shortcut_y, conv_z], mode='sum')
		y = Activation('relu')(y)

		print ('build conv_x')
		x1 = y
		conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, border_mode='same')(x1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = Activation('relu')(conv_x)

		print ('build conv_y')
		conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, border_mode='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = Activation('relu')(conv_y)

		print ('build conv_z')
		conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, border_mode='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		is_expand_channels = not (self.input_shape[-1] == n_feature_maps*2)
		if is_expand_channels:
			shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,border_mode='same')(x1)
			shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
		else:
			shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
		print ('Merging skip connection')
		y = merge([shortcut_y, conv_z], mode='sum')
		y = Activation('relu')(y)

		print ('build conv_x')
		x1 = y
		conv_x = keras.layers.Conv2D(n_feature_maps*2, 8, 1, border_mode='same')(x1)
		conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
		conv_x = Activation('relu')(conv_x)

		print ('build conv_y')
		conv_y = keras.layers.Conv2D(n_feature_maps*2, 5, 1, border_mode='same')(conv_x)
		conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
		conv_y = Activation('relu')(conv_y)

		print ('build conv_z')
		conv_z = keras.layers.Conv2D(n_feature_maps*2, 3, 1, border_mode='same')(conv_y)
		conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

		is_expand_channels = not (self.input_shape[-1] == n_feature_maps*2)
		if is_expand_channels:
			shortcut_y = keras.layers.Conv2D(n_feature_maps*2, 1, 1,border_mode='same')(x1)
			shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)
		else:
			shortcut_y = keras.layers.normalization.BatchNormalization()(x1)
		print ('Merging skip connection')
		y = merge([shortcut_y, conv_z], mode='sum')
		y = Activation('relu')(y)

		full = keras.layers.pooling.GlobalAveragePooling2D()(y)   
		out = Dense(self.output_size, activation='softmax')(full)
		self.model = Model(input=x, output=out)
		optimizer = keras.optimizers.Adam()
		self.model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])


	def fit(self, X, Y, X_Test, Y_Test, epochs=5, batch_size=32):
		X_Train_Reshape = X.reshape((X.shape + (1,1, )))
		X_Test_Reshape = X_Test.reshape((X_Test.shape + (1,1, )))

		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		X_Train, X_Val, Y_Train, Y_Val = train_test_split(X_Train_Reshape, Y, test_size=0.1)
		self.model.fit(X_Train, np_utils.to_categorical(Y_Train), epochs=epochs, batch_size=batch_size,
			callbacks=[early_stop], validation_data=(X_Val, np_utils.to_categorical(Y_Val)))
		print(self.model.evaluate(X_Test_Reshape, np_utils.to_categorical(Y_Test), batch_size=batch_size))

	def fit_generator(self, generator, steps_per_epoch, epochs=5, batch_size=32):
		self.construct_model()
		early_stop = EarlyStopping(patience=1)
		self.model.fit_generator(generator, steps_per_epoch, epochs, callbacks=[early_stop],
			validation_data=(X_Val, Y_Val))
		print(self.model.evaluate_generator(generator, steps=steps_per_epoch))

	def predict_classes(self, X):
		return np.argmax(self.model.predict(X), axis=1)

	def predict_proba(self, X):
		return self.model.predict_proba(X)
		

