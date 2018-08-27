import gym;
from ddpg import DDPG as Agent;
from actor import Actor;
from critic import Critic;
import time;
import matplotlib.pyplot as plt;

EPISODES = 500;
STEPS_PER_EPISODE = 200;

def main():
    env = gym.make("Pendulum-v0");
    actionSpaceSize = env.action_space.shape[0]
    stateSpaceSize = env.observation_space.shape[0];
    agent = Agent(env = env,
                stateSpaceSize = stateSpaceSize,
                actionSpaceSize = actionSpaceSize,
                actorClass = Actor,
                criticClass = Critic
    )

    rewardsList = [];

    for _ in range(EPISODES):
        print "EPISODE : " + str(_); 
        totalReward = 0.0;
        s = env.reset();
        a = env.action_space.sample();
        t_start = time.time();
        for i in range(STEPS_PER_EPISODE):
            #env.render();
            # reshape the state "s" for Agent's model.
            s = s.reshape((1, env.observation_space.shape[0]));
            # get an action from agent on current state "s"
            a = agent.Act(s);
            # reshape the action "a" for Agent's model.
            a = a.reshape((1, env.action_space.shape[0]));

            _s, r, done, _ = env.step(a);
            _s = _s.reshape((1,env.observation_space.shape[0]));
            totalReward += r;
            agent.remember((s,a,r,_s,done));
            agent.train();
            s = _s;
            if i == int(STEPS_PER_EPISODE/2):
                print "update target models";
                agent.updateTargetModels();
        print "Total Reward = " + str(totalReward);
        t_end = time.time();
        print "Time elapsed = " + str(- t_start + t_end) + "seconds"
        rewardsList.append(totalReward);
    plt.plot(rewardsList);
    plt.show();

if __name__ == "__main__":
    main();