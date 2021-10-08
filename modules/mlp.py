from keras.models import Model
from keras.layers import Input, Dense, Dropout
import numpy as np
import keras 
from keras.callbacks import ReduceLROnPlateau

class MLP(object):
    def __init__(self):
        pass
    
    def fit(self,X,Y):
        input_dim=X.shape[1]
        output_dim=Y.shape[1]
        self.model=self.create_model(input_dim,output_dim)
        reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,patience=200, min_lr=0.1)
        self.model.fit(X,Y,batch_size=32, epochs=2,verbose=1, validation_split=0.05, callbacks=[reduce_lr])
    
    def predict(self,X_Test):
        
        return self.model.predict(X_Test)
        
    def create_model(self,input_dim,output_dim):
        x = Input(shape=(input_dim,))
        y= Dropout(0.1)(x)
        y = Dense(500, activation='relu')(y)
        y = Dropout(0.2)(y)
        y = Dense(500, activation='relu')(y)
        y = Dropout(0.2)(y)
        y = Dense(500, activation = 'relu')(y)
        y = Dropout(0.3)(y)
        out = Dense(units=output_dim, activation='softmax')(y)
        model = Model(inputs=x, outputs=out)
        optimizer = keras.optimizers.Adadelta()    
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model
        
