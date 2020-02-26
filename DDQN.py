import numpy as np

from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Dot
from keras.models import load_model
from keras import backend as K






class DDQN:
    # only supports 1D inputLen inputs and outputLen action options
    # architecture option allows easy testing of different network architectures
    def __init__(inputLen, outputLen, discountRate, architecture=1):
        self.DISCOUNT_RATE = discountRate
        
        if (architecture == 1):
            self.predictRewardModel, self.actionChoiceModel = self.basicDDQNStructure(inputLen, outputLen)
            self.targetModel = self.basicDDQNStructure(inputLen, outputLen, target=True)


        self.updateTargetModel()


    # simple utility functions
    def saveModel(self, saveName):
        self.predictRewardModel.save(saveName)

    def loadModel(self, saveName):
        self.predictRewardModel = load_model(saveName)
        self.actionChoiceModel.set_weights(self.predictRewardModel.get_weights())
        self.targetModel.set_weights(self.predictRewardModel.get_weights())

    # training functions
    def train(self, epochs, states, actions, rewards, gameOvers, newStates):
        if not (len(states) == len(actions) == len(rewards) == len(gameOvers) == len(newStates)):
            print("Given training lists are not matching lengths! Exitting.")
            exit(1)

        # does not include future rewards of state-action pairs that resulted in gameOver
        targetRewards = rewards + self.DISCOUNT_RATE * np.logical_not(gameOvers).astype(int) * self.targetModel.predict(newStates[i].reshape(1, INPUT_LEN))

        self.predictRewardModel.fit([states, actions], targetRewards, epochs=epochs)

        # update target model after completing training step
        self.updateTargetModel()
            
    def updateTargetModel(self):
        self.targetModel.set_weights(self.predictRewardModel.get_weights())


    # ARCHITECTURES
        
    # target = False: returns two network outputs:
            # predictRewardModel predicts the reward for a state-action pair (action input is a 1-hot array of chosen action) - this is what gets trained
            # actionChoiceModel returns the index of the action with the best expected reward (used for agent to choose actions)
    # target = True: returns one network output: predictRewardModel aka targetModel
            # predictRewardModel returns the best predicted reward (only updated to nontarget network weights periodically)
    def basicDDQNStructure(self, inputLen, outputLen, target=False, optimizer='sgd', loss='mse', metrics=['mse']):
        # assumes input is 1D Box, output is Discrete (OpenAI Gym state-action classes)

        stateInput = Input(shape=(inputLen,))
        actionInput = Input(shape=(outputLen,))

        innerLayer = Dense(2 * inputLen, activation='sigmoid')(stateInput)
        innerLayer = Dense(4 * inputLen, activation='sigmoid')(innerLayer)
        innerLayer = Dense(8 * inputLen, activation='sigmoid')(innerLayer)

        rewardPerAction = Dense(outputLen, use_bias=True)(innerLayer) # linear activation


        if target:
            # max of rewardPerAction gives max predicted reward
            # ignores value of actionInput
            # not used for training, used for separate targetModel, which predicts discounted future rewards and is only updated periodically
            bestReward = Lambda(lambda x: K.max(x))(rewardPerAction)

            bestRewardModel = Model(input=stateInput, output=bestReward)
            bestRewardModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            return bestRewardModel

        else:
            # argmax of rewardPerAction gives index of chosen action
            # ignores value of actionInput
            # not used for training, only for deciding the agent's actions
            bestAction = Lambda(lambda x: K.argmax(x))(rewardPerAction)

            actionChoiceModel = Model(input=stateInput, output=bestAction) # not trained on
            actionChoiceModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # sets all rewardsPerAction values to zero except the one corresponding to actionInputs 1-hot element
            # allows us to isolate the expected reward of a state-action pair
            expectedReward = Dot(axes=-1)([rewardPerAction, actionInput])

            predictRewardModel = Model(input=[stateInput, actionInput], output=expectedReward)
            predictRewardModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            return predictRewardModel, actionChoiceModel
