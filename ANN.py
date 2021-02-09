import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras import optimizers
from sklearn.model_selection import train_test_split
import pandas as pd

class PreferenceModelNetwork():

    """
    The Preference Model Network class itself.
    Initialize with data matrix and preference vector.
       The defualt values in the argument list is the ones I found by experimentation, and you do not need
       to set these if you don't want.
     * X: Data matrix [N:M], N foods, M nutrition types.
     * y: Preference vector [N:1], N preference values (value between 0 and 1).
     * N_layers: number of layers in the network.
     * N_units: list of number of hidden units in each layer.
     * learning_rate: The learning rate of the network.
     * decay_rate: The decay rate of the network.
    
    Functions:
     - train - Trains the model. Uses the last trained network (if any) as a base network, otherwise:
         The defualt values in the argument list is the ones I found by experimentation, and you do not need
         to set these if you don't want.
       * num_epochs: The number of epochs for the network to train over.
       * batch: The batch size of the network to train with.
       * X_new: The new data matrix to use for training instead.
       * y_new: The new preference vector to use for training instead.
       
     - predict - Returns a list of N estimated food preferences given a data matrix X_test:
       * X_test: Data matrix [N:M], N foods, M nutrition types.
    """
    
    def __init__(self, X, y, N_layers=3, N_units=[390, 750, 500], learning_rate=1e-3, decay_rate=1e-5, momentum = 0.5):
        # Initialize variables
        assert(isinstance(X, pd.DataFrame))
        self.X = X.values
        self.foods = X.index
        self.y = y
        # Generate network here
        self.nn = Sequential()
        self.nn.add(Dense(N_units[0], input_shape=(self.X.shape[1],)))
        self.nn.add(Activation('relu'))
        self.nn.add(Dropout(0.5))
        for i in range(1,N_layers):
            self.nn.add(Dense(N_units[i]))
            self.nn.add(Activation('relu'))
            self.nn.add(Dropout(0.5))
        self.nn.add(Dense(1))
        self.nn.add(Activation('hard_sigmoid'))
        opti = optimizers.Adam(lr=learning_rate, epsilon=1e-08, decay=decay_rate)
        #opti = optimizers.SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
        self.nn.compile(loss='binary_crossentropy', optimizer=opti, metrics=['accuracy'])

        
    def train(self, num_epochs=10, batch=5, X_new = None, y_new = None):
        if X_new is not None and y_new is not None:
            assert(isinstance(X_new,pd.DataFrame))
            idx = ~X_new.index.isin(self.foods)
            self.X = np.append(self.X, X_new[idx].values, 0)
            self.foods = np.append(self.foods, X_new.index[idx].values, 0)
            self.y = np.append(self.y, [i for indx,i in enumerate(y_new) if idx[indx]], 0)
        self.nn.fit(self.X, self.y, epochs = num_epochs, batch_size=batch, shuffle=False, verbose=0)
    
    
    def predict(self, X_test):
        assert(isinstance(X_test,pd.DataFrame))
        return self.nn.predict(X_test.values)
    
    