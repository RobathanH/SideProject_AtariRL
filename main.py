import gym
import numpy as np
import random

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Dot
from keras.models import load_model
from keras import backend as K

from DDQN import DDQN


# constants - game choice
# ENV = gym.make('AirRaid-ram-v0')
# ENV = gym.make('CartPole-v0')
ENV = gym.make('Boxing-ram-v0')
# ENV = gym.make('Breakout-ram-v0')
INPUT_LEN = len(ENV.observation_space.sample())
OUTPUT_LEN = ENV.action_space.n


# training constants
DISCOUNT_RATE = 0.9




        
        







if __name__ == "__main__":

    net = DDQN(INPUT_LEN, OUTPUT_LEN, DISCOUNT_RATE)

    states = []
    actions = []
    rewards = []
    newStates = []
    gameOvers = []
    episodeRewards = [] # summed rewards over whole episode

    for epoch in range(50):
        if len(states) >= 20000:
            states = []
            actions = []
            rewards = []
            newStates = []
            gameOvers = []
            episodeRewards = []

        oldReplayLen = len(states)

        firstRunPerEpoch = True

        while len(states) - oldReplayLen < 5000:
            done = False
            obs = ENV.reset()

            ticks = 0
            totalReward = 0
            
            while not done: # and ticks < 1000:
                if firstRunPerEpoch:
                    ENV.render()

                # calculate action
                if random.random() < 1 - 0.02 * epoch: # less exploration in each subsequent training round
                    action = ENV.action_space.sample()
                else:
                    action = net.action(obs)

                # take action
                newObs, reward, done, info = ENV.step(action)

                # create 1-hot array of action
                actionArr = np.zeros(OUTPUT_LEN)
                actionArr[action] = 1

                # save to experience buffer
                states.append(obs)
                actions.append(actionArr)
                rewards.append(reward)
                gameOvers.append(done)
                newStates.append(newObs)

                # update counters
                totalReward += reward
                ticks += 1

                # limit experience buffer growth between training sessions
                if len(states) - oldReplayLen >= 5000:
                    break

                # update current state to new state
                obs = newObs

            episodeRewards += ([totalReward] * ticks)

            firstRunPerEpoch = False
        
        net.train(10, states, actions, rewards, newStates, gameOvers)
        print("Training Round: ", epoch)


    print("Done training!")

    net.saveModel("latestModel.h5")
    
    while True:
        done = False
        obs = ENV.reset()
        while not done:
            ENV.render()
            action = net.action(obs)

            obs, _, done, _ = ENV.step(action)

    ENV.close()





