from keras.models import Sequential, save_model, load_model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping, TerminateOnNaN, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
import keras.backend as K
import numpy as np
import types
import tempfile

class Keras_NN(object):
    def __init__(self, layer_sizes, input_shape, mode="binary_classifier"):
        model = Sequential()
        model.add(Dense(layer_sizes[0], input_shape=input_shape))
        model.add(Dropout(0.3))
        model.add(BatchNormalization())
        for i in range(1, len(layer_sizes) - 1):
            model.add(Dense(layer_sizes[i]))
            model.add(Dropout(0.3))
            model.add(BatchNormalization())

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

    def fit(self, X, Y):
        if(not isinstance(Y, np.ndarray)):
            Y = np.array(Y)
        if(not isinstance(X, np.ndarray)):
            X = np.array(X)

        fl_x, val_x, fl_y, val_y = train_test_split(X, Y)
        self.model.fit(fl_x, fl_y, validation_data=[val_x, val_y], callbacks=[EarlyStopping(patience=4), TerminateOnNaN(), ReduceLROnPlateau(patience=2)], epochs=1000)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict_classes(self, X):
        return self.model.predict_classes(X)

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            save_model(self.model, fd.name, overwrite=True)
            model_str = fd.read()
        d = { 'model_str': model_str }
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = load_model(fd.name)
            self.model = model
        self.__dict__ = self.model.__dict__
