import numpy as np
from keras.layers import Dense,LSTM
from keras.models import Sequential

class LSTM_TMS(object):
    def __init__(self):
        self.model=None
          

    def create_model(self,input_dim,output_shape):
        model = Sequential()
        model.add(LSTM(128, dropout=0.2,recurrent_dropout=0.2,input_shape=(1,input_dim),return_sequences=True))
        model.add(LSTM(128,dropout=0.2,recurrent_dropout=0.2,return_sequences=True))
        model.add(LSTM(128))
        model.add(Dense(128))
        model.add(Dense(128))
        model.add(Dense(output_shape, activation='sigmoid'))
        print(model.summary)
        model.compile(loss='binary_crossentropy', optimizer='adam',metrics=['accuracy'])
        return model

    def fit(self,X,Y):
        X=np.reshape(X,(X.shape[0],1,X.shape[1]))
        output_shape=Y.shape[1]
        self.model =self.create_model(X.shape[2],output_shape) 
        self.model.fit(X,Y,epochs=20,batch_size=32,verbose=1)

    def predict(self,X_Test):
        X_Test=np.reshape(X_Test,(X_Test.shape[0],1,X_Test.shape[1]))
        x=self.model.predict(X_Test)
        return x
