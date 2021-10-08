from keras.models import Model
import numpy as np
import keras 
from keras.callbacks import ReduceLROnPlateau
class FCN(object):
    def __init__(self):
        pass
    
   
    def fit(self,X,Y):
        X=np.reshape(X,(X.shape[0],X.shape[1],1,1))
        nb_classes=Y.shape[1]
        self.model=self.create_model(nb_classes)
        reduce_lr = ReduceLROnPlateau(monitor = 'loss', factor=0.5,patience=50, min_lr=0.0001) 
        self.model.fit(X, Y, batch_size=32, epochs=20,verbose=1, validation_split=0.05, callbacks = [reduce_lr])
    def predict(self,X_Test):
        X_Test=np.reshape(X_Test,(X_Test.shape[0],X_Test.shape[1],1,1))
        return self.model.predict(X_Test)
        
        

    def create_model(self,nb_classes):
        x = keras.layers.Input(shape=(None,None,1))
        #    drop_out = Dropout(0.2)(x)
        conv1 = keras.layers.Conv2D(128, (8, 1), padding='same')(x)
        conv1 = keras.layers.normalization.BatchNormalization()(conv1)
        conv1 = keras.layers.Activation('relu')(conv1)

        #    drop_out = Dropout(0.2)(conv1)
        conv2 = keras.layers.Conv2D(256,(5, 1), padding='same')(conv1)
        conv2 = keras.layers.normalization.BatchNormalization()(conv2)
        conv2 = keras.layers.Activation('relu')(conv2)

        #    drop_out = Dropout(0.2)(conv2)
        conv3 = keras.layers.Conv2D(128, (3, 1), padding='same')(conv2)
        conv3 = keras.layers.normalization.BatchNormalization()(conv3)
        conv3 = keras.layers.Activation('relu')(conv3)

        full = keras.layers.pooling.GlobalAveragePooling2D()(conv3)    
        out = keras.layers.Dense(nb_classes, activation='softmax')(full)


        model = Model(inputs=x, outputs=out)
        optimizer = keras.optimizers.Adam()
        model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])
        return model