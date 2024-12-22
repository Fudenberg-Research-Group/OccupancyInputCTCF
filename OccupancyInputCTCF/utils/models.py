from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.optimizers import Adam



def model_linear_reg(features, learningrate):
    model = Sequential()
    model.add(Input(shape=(features.shape[1], features.shape[2])))
    model.add(Flatten())
    model.add(Dense(1))
    model.compile(optimizer= Adam(learning_rate = learningrate), loss = 'mean_squared_error')
    return model

def model_logistic_reg(features, learningrate):
    model = Sequential()
    model.add(Input(shape=(features.shape[1], features.shape[2])))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid')) 
    model.compile(optimizer= Adam(learning_rate = learningrate), loss = 'mean_squared_error')
    return model