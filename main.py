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

    states = np.empty((0, INPUT_LEN), float)
    actions = np.empty((0, OUTPUT_LEN), int)
    rewards = np.empty((0,), float)
    newStates = np.empty((0, INPUT_LEN), float)
    gameOvers = np.empty((0,), bool)

    overallActionCounter = 0

    for epoch in range(50):
        if len(states) >= 20000:
            states = np.empty((0, INPUT_LEN), float)
            actions = np.empty((0, OUTPUT_LEN), int)
            rewards = np.empty((0,), float)
            newStates = np.empty((0, INPUT_LEN), float)
            gameOvers = np.empty((0,), bool)

        oldReplayLen = overallActionCounter

        firstRunPerEpoch = True

        while overallActionCounter - oldReplayLen < 5000:
            done = False
            obs = ENV.reset()

            ticks = 0
            
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
                states = np.append(states, [obs], axis=0)
                actions = np.append(actions, [actionArr], axis=0)
                rewards = np.append(rewards, [reward], axis=0)
                gameOvers = np.append(gameOvers, [done], axis=0)
                newStates = np.append(newStates, [newObs], axis=0)

                # update counters
                ticks += 1
                overallActionCounter += 1

                # limit experience buffer growth between training sessions
                if overallActionCounter - oldReplayLen >= 5000:
                    break

                # update current state to new state
                obs = newObs

            firstRunPerEpoch = False
        
        net.train(10, states, actions, rewards, gameOvers, newStates)
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





