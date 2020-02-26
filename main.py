import gym
import numpy as np
import random

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Dot
from keras.models import load_model
from keras import backend as K


# constants - game choice
# ENV = gym.make('AirRaid-ram-v0')
# ENV = gym.make('CartPole-v0')
ENV = gym.make('Boxing-ram-v0')
# ENV = gym.make('Breakout-ram-v0')
INPUT_LEN = len(ENV.observation_space.sample())
OUTPUT_LEN = ENV.action_space.n


# training constants
DISCOUNT_RATE = 0.9



# neural net class
class DeepQNet:
    def __init__(self):
        self.predictRewardModel, self.chooseActionModel, self.maxRewardModel = self.createNN() # computes the reward of the current state-action
        # _, _, self.maxRewardModel = self.createNN() # computes rewards of future state-actions, only updated periodically to match predictModel

    def createNN(self):
        # set input and output size constants for each game
        # assumes input is Box
        # assumes output is Discrete

        stateInput = Input(shape=(INPUT_LEN,))
        actionInput = Input(shape=(OUTPUT_LEN,)) # used for loss function part of NN, input is 1-hot array of chosen action

        innerLayer = Dense(2 * INPUT_LEN, activation='sigmoid')(stateInput)
        innerLayer = Dense(4 * INPUT_LEN, activation='sigmoid')(innerLayer)
        innerLayer = Dense(8 * INPUT_LEN, activation='sigmoid')(innerLayer)
        innerLayer = Dense(4 * INPUT_LEN, activation='sigmoid')(innerLayer)
        innerLayer = Dense(2 * INPUT_LEN, activation='sigmoid')(innerLayer)

        rewardPerAction = Dense(OUTPUT_LEN, use_bias=True)(innerLayer) # linear activation

        # argmax of rewardPerAction gives index of chosen action
        # ignores value of actionInput
        # not used for training, only for deciding the agent's actions
        bestAction = Lambda(lambda x: K.argmax(x))(rewardPerAction)

        # max of rewardPerAction gives max predicted reward
        # ignores value of actionInput
        # not used for training, used for separate targetModel, which predicts discounted future rewards and is only updated periodically
        bestReward = Lambda(lambda x: K.max(x))(rewardPerAction)

        # sets all rewardsPerAction values to zero except the one corresponding to actionInputs 1-hot element
        # allows us to isolate the expected reward of a state-action pair
        expectedReward = Dot(axes=-1)([rewardPerAction, actionInput])

        actionChoiceModel = Model(input=stateInput, output=bestAction) # not trained on
        bestRewardModel = Model(input=stateInput, output=bestReward) # not trained on

        predictRewardModel = Model(input=[stateInput, actionInput], output=expectedReward)
        predictRewardModel.compile(optimizer='sgd', loss='mse', metrics=['mae'])
        
        return predictRewardModel, actionChoiceModel, bestRewardModel

    def saveModel(self, saveName):
        self.predictRewardModel.save(saveName)

    def loadModel(self, saveName):
        self.predictRewardModel = load_model(saveName)


    # gives expected best action index
    def action(self, state):
        return self.chooseActionModel.predict(state.reshape(1, INPUT_LEN))[0]


    # states = list of observations
    # actions = list of actions
    # rewards = list of reward given for that state-action pair
    # newStates = list of resultant state from each corresponding state-action pair
    # episodeRewards = list of overall reward summed across entire episode, listed for each tick
    def trainOnReplay(self, states, actions, rewards, newStates, episodeRewards):
        # calculate future-discounted target rewards
        targetReward = []
        for i in range(len(states)):
            # target = episodeRewards[i]
            target = rewards[i] + DISCOUNT_RATE * self.maxRewardModel.predict(newStates[i].reshape(1, INPUT_LEN))
            targetReward.append(target)

        # find sampleWeights from episodeRewards
        minEpReward = min(episodeRewards)
        maxEpReward = max(episodeRewards)
        midPointEpReward = 0.5 * (maxEpReward + minEpReward)
        sampleWeights = []
        for epReward in episodeRewards:
            sampleWeights.append(1) # abs(epReward - midPointEpReward))

        print("min episode reward: ", minEpReward)
        print("max episode reward: ", maxEpReward)
        
        # convert to numpy arrays
        sampleWeights = np.array(sampleWeights)
        states = np.array(states)
        actions = np.array(actions)
        targetReward = np.array(targetReward)

        self.predictRewardModel.fit([states, actions], targetReward, epochs=10, sample_weight=sampleWeights)









        
        







if __name__ == "__main__":

    net = DeepQNet()

    states = []
    actions = []
    rewards = []
    newStates = []
    episodeRewards = [] # summed rewards over whole episode

    for epoch in range(50):
        if len(states) >= 20000:
            states = []
            actions = []
            rewards = []
            newStates = []
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
        
        net.trainOnReplay(states, actions, rewards, newStates, episodeRewards)
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





