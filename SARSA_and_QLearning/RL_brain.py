import random;

class RL_Brain(object):
    '''
        The parent class for SARSA and QLearning
        algorithm classes.
    '''
    def __init__(self,
                agent,
                gamma = 0.9,
                alpha = 0.1, 
                epsilon = 0.1, 
                alpha_decay = 0.005, 
                epsilon_decay = 0.001, 
                stepsPerEpisode = 50, 
                noOfEpisodes = 2000
                ):
        self.agent = agent;
        self.stepsPerEpisode = stepsPerEpisode;
        self.noOfEpisodes = noOfEpisodes;
        self.gamma = gamma;
        self.alpha_decay = alpha_decay;
        self.alpha = alpha;
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay;

    '''
        This function performs all the episodes.
        and it also decays the alpha. 
    '''                
    def Process(self, initState = None, showPolicyAfterEachBatch = False, showEpisodeNumberOfEpisodesElapsed = False):
        t = 1.0;
        for _e in range(self.noOfEpisodes + 1):
            if _e % 100 == 0:
                t += self.epsilon_decay;
                self.epsilon = self.epsilon / t;
            if _e % int(self.noOfEpisodes / 5) == 0:
                if showEpisodeNumberOfEpisodesElapsed:
                    print ("--> Episodes done:" + str(_e));
                if showPolicyAfterEachBatch:
                    print self.agent.environment.getPolicy();
            self.takeStepsInEpisode(initState);

    def takeStepsInEpisode(self, initState):
        pass;
#######################################################################

class RL_SARSA(RL_Brain):
    '''
        This class implement the SARSA approach of the Temporal
        Difference Learning.
        Here the value function is extimated as:
            Q(s,a) = Q(s,a) + alpha * (r + gamma * Q(s`,a`) - Q(s,a))
        
        This is on-policy algorithm.

        This function performs an episode.
        It ends if the required number of steps are completed or
        a terminal state has been achieved.
    '''
    def takeStepsInEpisode(self, initState):
        if initState is not None:
            self.agent.setState(initState[0], initState[1]);
        else:
            self.agent.randomizeState();
        s = self.agent.currentState();
        (a, loc, r) = self.agent.getAction(self.epsilon);
        for _steps in range(0,self.stepsPerEpisode):
            if self.agent.isGameOver(): break;
            if loc is None: break;
            
            (_s, r) = self.agent.setState(loc[0], loc[1]);
            if r is None: 
                r = self.agent.environment.model.Value(loc[0],loc[1]); # if terminal
            (_a, loc, _r) = self.agent.getAction(self.epsilon);
            SARSA_states = (s,a,r,_s,_a);
            if _a is not None : # if not terminal
                self.agent.environment.Q[s][a] += self.alpha * (r + self.gamma * self.agent.environment.Q[_s][_a] - self.agent.environment.Q[s][a]);
                self.agent.environment.statesVisited.append(s);                
            else:
                break;                
            s = _s;
            a = _a;
            self.agent.environment.updateCounts_sa[s][a] += self.alpha_decay;
            self.alpha = self.alpha / self.agent.environment.updateCounts_sa[s][a];
#######################################################################

class RL_QLearning(RL_Brain):
    '''
        This class implement the SARSA approach of the Temporal
        Difference Learning.
        Here the value function is extimated as:
            Q(s,a) = Q(s,a) + alpha * (r + gamma * argmax(Q(s`,a`)) - Q(s,a))
    
        This is off-policy algorithm.

        This function performs an episode.
        It ends if the required number of steps are completed or
        a terminal state has been achieved.
    '''
    def takeStepsInEpisode(self, initState):
        if initState is not None:
            self.agent.setState(initState[0], initState[1]);
        else:
            self.agent.randomizeState();
        s = self.agent.currentState();
        (a, loc, r) = self.agent.getAction(self.epsilon);
        for _steps in range(0,self.stepsPerEpisode):
            if self.agent.isGameOver(): break;
            if loc is None: break;
            
            (_s, r) = self.agent.setState(loc[0], loc[1]);
            if r is None: 
                r = self.agent.environment.model.Value(loc[0],loc[1]); # if terminal
            (_a, loc, _r) = self.agent.getAction(self.epsilon);
            (max_a, max_l, max_r) = self.agent.getAction(-0.1);
            if _a is not None : # if not terminal
                self.agent.environment.Q[s][a] += self.alpha * (r + self.gamma * self.agent.environment.Q[_s][max_a] - self.agent.environment.Q[s][a]);
                self.agent.environment.statesVisited.append(s);
            else:
                break;                
            s = _s;
            a = _a;
            self.agent.environment.updateCounts_sa[s][a] += self.alpha_decay;
            self.alpha = self.alpha / self.agent.environment.updateCounts_sa[s][a];
#######################################################################

class RL_Double_QLearning(RL_Brain):
    '''
        This class implements the Double Q Learning approach.
        There are two Q value functions. With a probability of
        i = 0.5. One of the value function is updated on the basis
        of the next action of the other value function.
        The equation that is followed is:

        Q[i](s,a) = Q[i](s,a) + alpha * (r + gamma * Q[i-1](s`,a`) - Q[i](s,a))
    '''
    
    def __init__(self,
                agent,
                gamma = 0.9,
                alpha = 0.1, 
                epsilon = 0.1, 
                alpha_decay = 0.005, 
                epsilon_decay = 0.001, 
                stepsPerEpisode = 50, 
                noOfEpisodes = 2000
        ):
        self.agent1 = agent;
        super(RL_Double_QLearning, self).__init__(
            agent = agent,
            gamma = gamma,
            alpha = alpha,
            epsilon = epsilon,
            alpha_decay = alpha_decay,
            epsilon_decay = epsilon_decay,
            stepsPerEpisode = stepsPerEpisode,
            noOfEpisodes = noOfEpisodes
        )

    def takeStepsInEpisode(self, initState):
        if initState is not None:
            self.agent.setState(initState[0], initState[1]);
            self.agent1.setState(initState[0], initState[1]);
        else:
            self.agent.randomizeState();
        s = self.agent.currentState();
        (a, loc, r) = self.agent.getAction(self.epsilon);
        for _steps in range(0,self.stepsPerEpisode):
            if self.agent.isGameOver(): break;
            if loc is None: break;
            
            (_s, r) = self.agent.setState(loc[0], loc[1]);
            if r is None: 
                r = self.agent.environment.model.Value(loc[0],loc[1]); # if terminal
            (_a, loc, _r) = self.agent.getAction(self.epsilon);
            (max_a, max_l, max_r) = self.agent.getAction(-0.1);
            (max_a1, max_l1, max_r1) = self.agent1.getAction(-0.1);
            if _a is not None : # if not terminal
                Q_choice = random.choice(range(0,2));
                if Q_choice == 0:
                    self.agent.environment.Q[s][a] += self.alpha * (r + self.gamma * self.agent1.environment.Q[_s][max_a1] - self.agent.environment.Q[s][a]);
                else:
                    self.agent1.environment.Q[s][a] += self.alpha * (r + self.gamma * self.agent.environment.Q[_s][max_a] - self.agent1.environment.Q[s][a]);
                self.agent.environment.statesVisited.append(s);
                self.agent1.environment.statesVisited.append(s);
            else:
                break;                
            s = _s;
            a = _a;
            self.agent.environment.updateCounts_sa[s][a] += self.alpha_decay;
            self.alpha = self.alpha / self.agent.environment.updateCounts_sa[s][a];