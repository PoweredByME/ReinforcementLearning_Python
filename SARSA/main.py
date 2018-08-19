import environment as env;
import agent;
import RL_brain;

def main():
    g = env.GridWorld(  
                    5, 5,
                    init_number = -3.0,
                    obstacles = [(0,1), (1,2), (2,2), (3,2), (2,1), (3,1)],
                    rewards = [(1,1,34), (3,3,-50), (1,3,10)]
                    );
    e = env.RL_Environment(g);
    a = agent.Agent(e);
    brain = RL_brain.RL_SARSA(a, noOfEpisodes = 10);
    brain.Process((2,3));
    print e.getPolicy();
main();