import gym
import numpy as np
import random
from collections import deque
from sklearn.model_selection import train_test_split

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Dot
from keras.models import load_model
from keras import backend as K

from DDQN import DDQN
from ExpBuffer import ExpBuffer
from VAE import VAE


# constants - game choice
ENV = gym.make('AirRaid-v0')
# ENV = gym.make('Boxing-v0')
# ENV = gym.make('Breakout-v0')
INPUT_LEN = len(ENV.observation_space.sample().flatten())
OUTPUT_LEN = ENV.action_space.n


# buffer constants
BUFFER_SIZE = 100000


# training constants
RL_TRAIN_SAMPLE_SIZE = 50000

DISCOUNT_RATE = 0.9



# auto-encoder constants
ENCODED_LEN = 200
VAE_VISIBLE_FRAMES = 2 # auto-encoder compresses two subsequent frames into the feature vector
VAE_TRAINING_SIZE = 20000



# normalizes input data into values from 0 to 1
# assumes original input elements are scaled 0 to 255
def normalizeImgs(data):
    data = np.array(data).astype('float64')
    data *= 1/255.
        

# initializes auto-encoder training with observations from random actions in environment
def initVAE():
    # no need to use ExpBuffer for this
    buffer = []
    bufLen = 0

    queue = deque()
    
    while bufLen < VAE_TRAINING_SIZE:
        done = False
        obs = ENV.reset()

        normalizeImgs(obs)
        queue.append(obs)
        if len(queue) > VAE_VISIBLE_FRAMES:
            queue.popleft()
        if len(queue) == VAE_VISIBLE_FRAMES:
            sample = np.array(list(queue))
            buffer.append(sample)
            bufLen += 1

        while not done and bufLen < VAE_TRAINING_SIZE:
            action = ENV.action_space.sample()
            obs, reward, done, info = ENV.step(action)

            normalizeImgs(obs)
            queue.append(obs)
            if len(queue) > VAE_VISIBLE_FRAMES:
                queue.popleft()

            if len(queue) == VAE_VISIBLE_FRAMES:
                sample = np.array(list(queue))
                buffer.append(sample)
                bufLen += 1

    
    vae = VAE(buffer[0].shape, ENCODED_LEN)

    buffer = np.array(buffer)

    trainBuf, testBuf = buffer[:-1000], buffer[-1000:]
    
    vae.train(trainBuf, 10, 100)

    vae.saveModel('vae_2_frame.md5')

    #vae.loadModel('vae_2_frame.md5')
    
    print(vae.overall.evaluate(testBuf, testBuf))

    # display progress
    vae.display(np.array(random.sample(list(buffer), 3)))


initVAE()


if __name__ == "__main__":

    net = DDQN(INPUT_LEN, OUTPUT_LEN, DISCOUNT_RATE)
    print(net.predictRewardModel.summary())
    exit(1)

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
                buffer.add((obs.flatten(), actionArr, reward, done, newObs.flatten()))

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





