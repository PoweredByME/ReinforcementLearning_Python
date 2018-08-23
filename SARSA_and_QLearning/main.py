import environment as env;
import agent;
import RL_brain;

def main():
    print "Using SARSA";
    problem(RL_brain.RL_SARSA);
    print "Using Q-Learning";
    problem(RL_brain.RL_QLearning);
    print "Using Double Q-Learning";
    problem(RL_brain.RL_Double_QLearning);
    print "Using SARSA Lambda";
    problem(RL_brain.RL_SARSA_Lambda);



def problem(learningClass):
    g = env.GridWorld(  
                    4, 4,
                    init_number = -0.10,
                    obstacles = [(3,2),(1,2)],
                    rewards = [(0,0,1),(0,3,100),(1,3,-1)]
                    );
    e = env.RL_Environment(g);
    a = agent.Agent(e);

    brain = learningClass(  
                        a, 
                        noOfEpisodes = 50, 
                        epsilon = 0.5, 
                        epsilon_decay = 1e-5
                        );
    brain.Process(
        initState = None,
        showPolicyAfterEachBatch = False,
        showEpisodeNumberOfEpisodesElapsed = True
        );
    print "Final Policy"
    print e.getPolicy();


if __name__ == "__main__":
    main();