from keras.models import Sequential, Model;
from keras.layers import Dense, Dropout, Input;
from keras.layers.merge import Add, Multiply;
from keras.optimizers import Adam;
import keras.backend as K;

class Critic(object):
    '''
        This function defines and creates the model of the critic
        Change this function if a change in critic model is needed.
        This function must return:
            - model
            - trainable weights of the model
            - input layer of states of the model
            - input layer of actions of the model
    '''
    def _createModel(self):
        stateInputLayer = Input(shape = [self._stateSpaceSize]);
        state_h2 = Dense(6, activation = "relu")(stateInputLayer);

        actionInputLayer = Input(shape = [self._actionSpaceSize]);
        action_h1 = Dense(6)(actionInputLayer);

        merge = Add()([state_h2, action_h1]);
        merged_h1 = Dense(12, activation = "relu")(merge);
        outputLayer = Dense(1, activation = "relu")(merged_h1);

        inputLayer = [stateInputLayer, actionInputLayer];
        model = Model(inputs = inputLayer, outputs = outputLayer);
        optimizer = Adam(lr = self._learningRate);
        model.compile(loss = "mse", optimizer = optimizer);
        
        return model, model.trainable_weights, stateInputLayer, actionInputLayer;
    ###############################################################


    def __init__(self,
                stateSpaceSize,
                actionSpaceSize,
                learningRate = 0.01,
    ):
        self._stateSpaceSize = stateSpaceSize;
        self._actionSpaceSize = actionSpaceSize;
        self._learningRate = learningRate;

        self._model, self._trainableWeights, self._modelStateInputs, self._modelActionInputs = self._createModel();


    def getModel(self): return self._model;
    def getWeights(self): return self._trainableWeights;
    def getStateInputs(self): return self._modelStateInputs;
    def getActionInputs(self): return self._modelActionInputs;