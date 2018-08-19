import environment as env;
import agent;
import RL_brain;

def main():
    print "Using SARSA";
    problem(RL_brain.RL_SARSA);
    print "Using Q-Learning";
    problem(RL_brain.RL_QLearning);


    

def problem(learningClass):
    g = env.GridWorld(  
                    5, 5,
                    init_number = -0.65,
                    obstacles = [(0,1), (1,2), (2,2), (3,2), (2,1), (3,1)],
                    rewards = [(1,1,34), (3,3,-50), (1,3,10)]
                    );
    e = env.RL_Environment(g);
    a = agent.Agent(e);
    
    brain = learningClass(a, noOfEpisodes = 20);
    brain.Process();

    print e.getPolicy();

main();