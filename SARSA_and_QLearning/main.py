import environment as env;
import agent;
import RL_brain;

def main():
    g = env.GridWorld(  
                    5, 5,
                    init_number = -0.95,
                    obstacles = [(0,1), (1,2), (2,2), (3,2), (2,1), (3,1)],
                    rewards = [(1,1,34), (3,3,-50), (1,3,10)]
                    );
    e = env.RL_Environment(g);
    a = agent.Agent(e);
    brain = RL_brain.RL_QLearning(a, noOfEpisodes = 2000);
    brain.Process((4,4));

    print e.getPolicy();

main();