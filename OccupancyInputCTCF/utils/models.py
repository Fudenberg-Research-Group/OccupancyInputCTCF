from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def Regressionmodel(features, learningrate, dense_layers = 1, activation=None, regularization_strength=0.01):
    model = Sequential()
    model.add(Input(shape=(features.shape[1], features.shape[2])))
    model.add(Flatten())
    
    # Add the dense layer with the given activation and regularization (if specified)
    model.add(Dense(dense_layers, activation=activation, kernel_regularizer=l2(regularization_strength))) 
    
    # Compile the model with MSE loss for continuous target values
    model.compile(optimizer=Adam(learning_rate=learningrate), loss='mean_squared_error')
    
    return model
