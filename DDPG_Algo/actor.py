from keras.models import Sequential, Model;
from keras.layers import Dense, Dropout, Input;
from keras.layers.merge import Add, Multiply;
from keras.optimizers import Adam;
import keras.backend as K;

class Actor(object):
    '''
        This function defines and creates the model of the actor
        Change this function if a change in actor model is needed.
        This function must return:
            - model
            - trainable weights of the model
            - input layer of the model
    '''
    def _createModel(self):
        inputLayer = Input(shape = [self._stateSpaceSize]);
        h = Dense(24, activation = "relu")(inputLayer);
        h = Dense(48, activation = "relu")(h);
        h = Dense(24, activation = "relu")(h);
        outputLayer = Dense(self._actionSpaceSize, activation = "relu")(h);

        model = Model(inputs = inputLayer, outputs = outputLayer);
        optimizer = Adam(lr = self._learningRate);
        model.compile(loss = "mse", optimizer = optimizer);
        return model, model.trainable_weights, inputLayer;
    ###############################################################


    def __init__(self,
                stateSpaceSize,
                actionSpaceSize,
                learningRate = 0.01,
    ):
        self._stateSpaceSize = stateSpaceSize;
        self._actionSpaceSize = actionSpaceSize;
        self._learningRate = learningRate;

        self._model, self._trainableWeights, self._modelStateInputs = self._createModel();


    def getModel(self): return self._model;
    def getWeights(self): return self._trainableWeights;
    def getStateInputs(self): return self._modelStateInputs;
