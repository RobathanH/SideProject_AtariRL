import gym
import numpy as np
import random

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Dot
from keras.models import load_model
from keras import backend as K

from DDQN import DDQN
from ExpBuffer import ExpBuffer


# constants - game choice
# ENV = gym.make('AirRaid-ram-v0')
# ENV = gym.make('CartPole-v0')
ENV = gym.make('Boxing-ram-v0')
# ENV = gym.make('Breakout-ram-v0')
INPUT_LEN = len(ENV.observation_space.sample())
OUTPUT_LEN = ENV.action_space.n


# buffer constants
BUFFER_SIZE = 100000


# training constants
RL_TRAIN_SAMPLE_SIZE = 50000

DISCOUNT_RATE = 0.9




        
        







if __name__ == "__main__":

    net = DDQN(INPUT_LEN, OUTPUT_LEN, DISCOUNT_RATE)

    buffer = ExpBuffer(BUFFER_SIZE)

    overallActionCounter = 0

    for epoch in range(50):
        oldReplayLen = overallActionCounter

        firstRunPerEpoch = True

        while overallActionCounter - oldReplayLen < 5000:
            done = False
            obs = ENV.reset()

            ticks = 0
            
            while not done and overallActionCounter - oldReplayLen < 5000:
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
                buffer.add((obs, actionArr, reward, done, newObs))

                # update counters
                ticks += 1
                overallActionCounter += 1

                # update current state to new state
                obs = newObs

            firstRunPerEpoch = False

        # sample from experience buffer and train
        states, actions, rewards, gameOvers, newStates = buffer.sample(RL_TRAIN_SAMPLE_SIZE)
        
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





