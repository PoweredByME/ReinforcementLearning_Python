from collections import deque;
import tensorflow as tf;
import keras.backend as K
import random;
import numpy as np;


class DDPG(object):
    def __init__(self,
                env,
                stateSpaceSize,
                actionSpaceSize,
                actorClass,
                criticClass,
                learningRate = 0.001,
                tau = 0.001,
                replayBufferSize = 2000,
                trainingBatchSize = 64,
                epsilon = 0.1,
                epsilonDecay = 0.95,
                gamma = 0.1
    ):

        # init a tensorflow and keras backend session
        self._session = tf.Session();
        K.set_session(self._session); 

        self._env = env;
        self._stateSpaceSize = stateSpaceSize;
        self._actionSpaceSize = actionSpaceSize;
        self._learningRate = learningRate;
        self._epsilon = epsilon;
        self._epsilonDecay = epsilonDecay;
        self._replayBufferSize = replayBufferSize;
        self._trainingBatchSize = trainingBatchSize;
        self._gamma = gamma;

        # init replay memory buffer
        self._replayBuffer = deque(maxlen = replayBufferSize);

        # init main and target models for the actor and critic
        self._actor = actorClass(stateSpaceSize, actionSpaceSize, learningRate);
        self._critic = criticClass(stateSpaceSize, actionSpaceSize, learningRate);
        self._targetActor = actorClass(stateSpaceSize, actionSpaceSize, learningRate);
        self._targetCritic = criticClass(stateSpaceSize, actionSpaceSize, learningRate);

        self._initActorModel();
        self._initCriticModel();

        self._session.run(tf.initialize_all_variables());

    def _initCriticModel(self):
        ################################################################
        # Init the critic model for the DDPG model.
        ################################################################
        self._criticModel = self._critic.getModel();
        self._criticWeights = self._critic.getWeights();
        self._criticStateInputs = self._critic.getStateInputs();
        self._criticActionInputs = self._critic.getActionInputs();
        self._targetCriticModel = self._targetCritic.getModel();

        self._criticGradients = tf.gradients(
                                self._criticModel.output,
                                self._criticActionInputs
                                );  # calculate de/dC

    def _initActorModel(self):
        ################################################################
        # Init the actor model for the DDPG Algorithm
        # Get the gradient of chaging the actor network params in
		# getting closest to the final value network predictions, de/dA
		# Calculate de/dA as = de/dC * dC/dA, where e is error
        ################################################################
        # from actor get
        #   - model
        #   - trainable weights
        #   - model state input
        ################################################################
        self._actorModel = self._actor.getModel();
        self._actorWeights = self._actor.getWeights();
        self._actorStateInputs = self._actor.getStateInputs();
        self._targetActorModel = self._targetActor.getModel();

        # placeholder for de/dC (from critic)
        self._actorCriticGradient = tf.placeholder(tf.float32, [None, self._actionSpaceSize]);
        # calculate dC/dA
        self._actorGradient = tf.gradients(
                                self._actorModel.output,
                                self._actorWeights,
                                - self._actorCriticGradient
                                ); 
        gradients = zip(self._actorGradient, self._actorWeights);
        self._optimize = tf.train.AdamOptimizer(self._learningRate).apply_gradients(gradients);

    def remember(self, sample):
        if len(self._replayBuffer) > self._replayBufferSize:
            self._replayBuffer.popLeft();
        self._replayBuffer.append(sample);

    def _trainActor(self, samples):
        for sample in samples:
            s, a, r, s1, done = sample;
            # predict action from current state "s"
            _a = self._actorModel.predict(s);
            
            # calculate de/dC
            gradients = self._session.run(self._criticGradients, feed_dict = {
                self._criticStateInputs : s,
                self._criticActionInputs : _a
            })[0];

            self._session.run(self._optimize, feed_dict = {
                self._actorStateInputs : s,
                self._actorCriticGradient : gradients
            });

    def _trainCritic(self, samples):
        c = 0;
        for sample in samples:
            s, a, r, s1, done = sample;
            if not done:
                targetAction = self._targetActorModel.predict(s1);
                futureReward = self._targetCriticModel.predict([s1, targetAction])[0][0];
                r += self._gamma * futureReward;
            c += 1;
            verbose = 0;
            if c == len(samples): verbose = 0;
            self._criticModel.fit([s,a],r,verbose = verbose);

    def train(self):
        if len(self._replayBuffer) < self._trainingBatchSize:
            return;
        samples = random.sample(self._replayBuffer, self._trainingBatchSize);
        self._trainCritic(samples);
        self._trainActor(samples);

    def _updateActorTargetModel(self):
        amw = self._actorModel.get_weights();
        atw = self._targetActorModel.get_weights();

        for i in range(len(atw)):
            atw[i] = self._tau * amw[i] + (1 - self._tau) * atw[i];
        self._targetActorModel.set_weights(atw);

    def _updateCriticTargetModel(self):
        cmw = self._criticModel.get_weights();
        ctw = self._targetCriticModel.get_weights();

        for i in range(len(ctw)):
            ctw[i] = self._tau * cmw[i] + (1 - self._tau) * ctw[i];
        self._targetCriticModel.set_weights(ctw);

    def updateTargetModels(self):
        self._updateActorTargetModel();
        self._updateCriticTargetModel();

    def Act(self, s):
        self._epsilon *= self._epsilonDecay;
        if np.random.random() < self._epsilon:
            return self._env.action_space.sample();
        return self._actorModel.predict(s);

