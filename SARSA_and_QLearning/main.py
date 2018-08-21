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
                    4, 4,
                    init_number = -0.10,
                    obstacles = [(3,2),(1,2)],
                    rewards = [(0,0,1),(0,3,10),(1,3,-1)]
                    );
    e = env.RL_Environment(g);
    a = agent.Agent(e);

    brain = learningClass(  
                        a, 
                        noOfEpisodes = 200, 
                        epsilon = 0.5, 
                        epsilon_decay = 10.0/5000.0
                        );
    brain.Process(showPolicyAfterEachBatch = False);
    print "Final Policy"
    print e.getPolicy();

main();