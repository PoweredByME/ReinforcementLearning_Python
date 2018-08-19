

class RL_SARSA(object):
    def __init__(self, agent, gamma = 0.9, alpha = 0.1, epsilon = 0.1, stepsPerEpisode = 50, noOfEpisodes = 2000):
        self.agent = agent;
        self.stepsPerEpisode = stepsPerEpisode;
        self.noOfEpisodes = noOfEpisodes;
        self.gamma = gamma;
        self.alpha = alpha;
        self.epsilon = epsilon

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
            else:
                break;                
            s = _s;
            a = _a;
            self.agent.environment.updateCounts_sa[s][a] += 0.005;
            self.alpha = self.alpha * self.agent.environment.updateCounts_sa[s][a];
                
    def Process(self, initState = None):
        for _e in range(self.noOfEpisodes):
            self.takeStepsInEpisode(initState);    




class RL_QLearning(object):
    def __init__(self, agent, gamma = 0.9, alpha = 0.1, epsilon = 0.1, stepsPerEpisode = 50, noOfEpisodes = 2000):
        self.agent = agent;
        self.stepsPerEpisode = stepsPerEpisode;
        self.noOfEpisodes = noOfEpisodes;
        self.gamma = gamma;
        self.alpha = alpha;
        self.epsilon = epsilon

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
            (max_a, max_l, max_r) = self.agent.getAction(0.0);
            SARSA_states = (s,a,r,_s,_a);
            if _a is not None : # if not terminal
                self.agent.environment.Q[s][a] += self.alpha * (r + self.gamma * self.agent.environment.Q[_s][max_a] - self.agent.environment.Q[s][a]);
            else:
                break;                
            s = _s;
            a = _a;
            self.agent.environment.updateCounts_sa[s][a] += 0.005;
            self.alpha = self.alpha / self.agent.environment.updateCounts_sa[s][a];
                
    def Process(self, initState = None):
        for _e in range(self.noOfEpisodes):
            self.takeStepsInEpisode(initState);    


