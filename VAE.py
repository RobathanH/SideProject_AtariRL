import numpy as np

from keras.layers import Lambda, Input, Dense, Flatten, Reshape, Concatenate, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.layers import Conv2D, ZeroPadding2D
from keras.models import Model
from keras.optimizers import SGD, RMSprop
from keras.losses import mse, binary_crossentropy
from keras import backend as K
import matplotlib.pyplot as plt




# samples from mean and vars
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


class VAE:
    def __init__(self, decodedShape, encodedLen, architecture=1):
        if architecture == 1:
            self.encoder, self.decoder, self.overall = self.defaultImgVAE(decodedShape, encodedLen)
        elif architecture == 2:
            self.encoder, self.decoder, self.overall = self.light2DVAE(decodedShape, encodedLen)


        self.encoder.summary()
        self.decoder.summary()
        self.overall.summary()


    # train
    def train(self, trainData, epochs, batchSize):
        self.overall.fit(trainData, trainData, epochs=epochs, batch_size=batchSize)

    # displays reconstructions
    # takes a list of VAE inputs (list of frames), and plots original and reconstructions of each
    def display(self, samples):
        for sample in samples:
            predicted = self.overall.predict(np.array([sample]))[0]
            for frameInd in range(len(sample)):
                if samples.shape[4] == 3: 
                    plt.imshow(sample[frameInd])
                elif samples.shape[4] == 1: #if only single channel, this must be reshaped to print
                    plt.imshow(np.reshape(sample[frameInd], samples.shape[2:4]))
                    
                plt.show()

                
                if samples.shape[4] == 3: 
                    plt.imshow(predicted[frameInd])
                elif samples.shape[4] == 1: #if only single channel, this must be reshaped to print
                    plt.imshow(np.reshape(predicted[frameInd], samples.shape[2:4]))
                plt.show()



    # utility
    def saveModel(self, saveName):
        self.overall.save_weights(saveName)

    def loadModel(self, saveName):
        self.overall.load_weights(saveName)



    # ----------- network structures ------------

    # decodedShape must be tuple with 4 dimensions (multiple frames with multiple channels)
    # encodedLen is length of flattened feature array after encoding
    def defaultImgVAE(self, decodedShape, encodedLen):
        decodedFlattenedLen = np.prod(list(decodedShape))
        if len(decodedShape) == 4:
            channelNum = decodedShape[-1]
            frameNum = decodedShape[0]
        else:
            print("Unsupported shape for VAE: Shape must be 4D. Received shape", decodedShape)
            exit(1)
        
        # encoder

        decoded_inputs = Input(decodedShape)

        innerLayer = Conv3D(16, (1,5,5), strides=(1,1,1), padding="same", activation="relu", data_format="channels_last", input_shape=decodedShape)(decoded_inputs)
        innerLayer = MaxPooling3D(pool_size=(1,4,4), strides=(1,4,4), padding="same")(innerLayer) # no maxpooling between frames

        innerLayer = Conv3D(16, (1,5,5), padding="same", activation="relu")(innerLayer)
        innerLayer = MaxPooling3D(pool_size=(1,4,4), strides=(1,4,4), padding="same")(innerLayer)

        # only last conv layer inspects relation between frames
        innerLayer = Conv3D(8, (frameNum,3,3), strides=(frameNum,1,1), padding="same", activation="relu")(innerLayer)

        # save shape before flatten - remove None values at start of shape
        preFlattenShape = list(K.int_shape(innerLayer))
        temp = []
        for el in preFlattenShape:
            if el != None:
                temp.append(el)
        preFlattenShape = tuple(temp)
        postFlattenLen = np.prod(temp)

        innerLayer = Flatten()(innerLayer)

        encoded_means = Dense(encodedLen)(innerLayer)
        encoded_log_vars = Dense(encodedLen)(innerLayer)

        encoded_sampled = Lambda(sampling, output_shape=(encodedLen,))([encoded_means, encoded_log_vars])        

        # final encoder declaration
        encoder = Model(decoded_inputs, [encoded_sampled, encoded_means, encoded_log_vars])
        

        # decoder
        encoded_inputs = Input((encodedLen,))

        innerLayer = Dense(postFlattenLen, activation="relu")(encoded_inputs)

        innerLayer = Reshape(preFlattenShape)(innerLayer)

        # duplicate up to three frames, hopefully conv layer with 'same' padding can produce correct changes between frames
        innerLayer = UpSampling3D(size=(frameNum,1,1))(innerLayer)

        
        innerLayer = Conv3D(8, (channelNum,3,3), padding="same")(innerLayer)
        
        
        innerLayer = UpSampling3D(size=(1,4,4))(innerLayer)

        # cut if shape doesn't match shape at this point of encoder (ignore channels and frames)
        # since maxpool uses 'same' padding, the current shape will only ever be 1,2,3 elements too large on index 1 or 2
        innerLayer = Cropping3D(((0,0),
                                 (0, int((16 - (decodedShape[1] % 16)) / 4) % 4),
                                 (0, int((16 - (decodedShape[2] % 16)) / 4) % 4)))(innerLayer)


        innerLayer = Conv3D(16, (1,5,5), padding="same")(innerLayer)

        
        innerLayer = UpSampling3D(size=(1,4,4))(innerLayer)

        # cut if shape doesn't match shape at this point of encoder (ignore channels and frames)
        # since maxpool uses 'same' padding, the current shape will only ever be 1,2,3 elements too large on index 1 or 2
        innerLayer = Cropping3D(((0,0),
                                 (0, (4 - (decodedShape[1] % 4)) % 4),
                                 (0, (4 - (decodedShape[2] % 4)) % 4)))(innerLayer)
        
        innerLayer = Conv3D(16, (1,5,5), padding="same")(innerLayer)

        decoded_outputs = Conv3D(channelNum, (1,1,1))(innerLayer)

        # final decoder declaration
        decoder = Model(encoded_inputs, decoded_outputs)

        # encoder piped into decoder for training
        decoded_outputs = decoder(encoder(decoded_inputs)[0])
        vae = Model(decoded_inputs, decoded_outputs)

        '''
        # reconstruction loss function
        reconstruction_loss = K.mean(binary_crossentropy(decoded_inputs, decoded_outputs))
        reconstruction_loss *= decodedFlattenedLen
        kl_loss = 1 + encoded_log_vars - K.square(encoded_means) - K.exp(encoded_log_vars)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(loss)
        '''

        sgd = SGD(lr=1e-5)
        vae.compile(optimizer='adadelta', loss='binary_crossentropy')

        return encoder, decoder, vae


    # simpler architecture for testing, uses conv2d layers
    def light2DVAE(self, decodedShape, encodedLen):
        if len(decodedShape) != 2:
            print("Unsupported shape. Must be 2d input.")
            exit(1)

        # encoder
        decoded_inputs = Input(decodedShape)

        innerLayer = Reshape(decodedShape + (1,))(decoded_inputs)

        innerLayer = Conv2D(8, (5,5), padding="valid")(innerLayer)
        innerLayer = Conv2D(8, (5,5), padding="valid")(innerLayer)
        innerLayer = Conv2D(8, (5,5), padding="valid")(innerLayer)

        # save shape before flatten - remove None values at start of shape
        preFlattenShape = list(K.int_shape(innerLayer))
        temp = []
        for el in preFlattenShape:
            if el != None:
                temp.append(el)
        preFlattenShape = tuple(temp)
        postFlattenLen = np.prod(temp)

        innerLayer = Flatten()(innerLayer)

        # dense to encoded stuff
        encoded_means = Dense(encodedLen)(innerLayer)
        encoded_log_vars = Dense(encodedLen)(innerLayer)

        encoded_sampled = Lambda(sampling, output_shape=(encodedLen,))([encoded_means, encoded_log_vars])        

        # final encoder declaration
        encoder = Model(decoded_inputs, [encoded_sampled, encoded_means, encoded_log_vars])
  

        # decoder
        encoded_inputs = Input((encodedLen,))

        innerLayer = Dense(postFlattenLen)(encoded_inputs)
        innerLayer = Reshape(preFlattenShape)(innerLayer)
        innerLayer = ZeroPadding2D(((2,2),(2,2)))(innerLayer)
        innerLayer = Conv2D(8, (5,5), padding="same")(innerLayer)
        innerLayer = ZeroPadding2D(((2,2),(2,2)))(innerLayer)
        innerLayer = Conv2D(8, (5,5), padding="same")(innerLayer)
        innerLayer = ZeroPadding2D(((2,2),(2,2)))(innerLayer)
        innerLayer = Conv2D(1, (5,5), padding="same")(innerLayer)

        decoded_outputs = Reshape(decodedShape)(innerLayer)

        decoder = Model(encoded_inputs, decoded_outputs)

        # overall vae
        decoded_outputs = decoder(encoder(decoded_inputs)[0])
        vae = Model(decoded_inputs, decoded_outputs)

        vae.compile(optimizer='adadelta', loss='binary_crossentropy')

        return encoder, decoder, vae
