import gym;
from ddpg import DDPG as Agent;
from actor import Actor;
from critic import Critic;


EPISODES = 200;
STEPS_PER_EPISODE = 100;

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


    for _ in range(EPISODES):
        print "EPISODE : " + str(_); 
        totalReward = 0.0;
        s = env.reset();
        a = env.action_space.sample();
        for _ in range(STEPS_PER_EPISODE):
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

            if _ == int(STEPS_PER_EPISODE/2):
                agent.updateTargetModels();
        print "Average Reward = " + str(totalReward / STEPS_PER_EPISODE);


if __name__ == "__main__":
    main();