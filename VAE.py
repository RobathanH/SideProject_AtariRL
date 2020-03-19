import numpy as np

from keras.layers import Lambda, Input, Dense, Flatten, Reshape, Concatenate, Conv3D, MaxPooling3D, UpSampling3D, Cropping3D
from keras.models import Model
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
    def __init__(self, encodedShape, decodedLen, architecture=1):
        if architecture == 1:
            self.encoder, self.decoder, self.overall = self.defaultImgVAE(encodedShape, decodedLen)


    # train
    def train(self, trainData, epochs, batchSize):
        self.overall.fit(trainData, epochs=epochs, batch_size=batchSize)

    # displays reconstructions
    # takes a list of VAE inputs (list of frames), and plots original and reconstructions of each
    def display(self, samples):
        for sample in samples:
            plt.imshow(sample)
            plt.show()
            plt.imshow(self.overall.predict([sample])[0])
            plt.show()



    # utility
    def saveModel(self, saveName):
        self.overall.save(saveName)

    def loadModel(self, saveName):
        self.overall = load_model(saveName)



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
        innerLayer = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="same")(innerLayer) # no maxpooling between frames
        firstMaxPoolShape = K.int_shape(innerLayer)[1:] # for decoder adjustment if maxpooling with odd number of inputs

        innerLayer = Conv3D(32, (1,3,3), padding="same", activation="relu")(innerLayer)
        innerLayer = MaxPooling3D(pool_size=(1,2,2), strides=(1,2,2), padding="same")(innerLayer)

        # only last conv layer inspects relation between frames
        innerLayer = Conv3D(64, (frameNum,3,3), strides=(frameNum,1,1), padding="same", activation="relu")(innerLayer)

        # save shape before flatten - remove None values at start of shape
        preFlattenShape = list(K.int_shape(innerLayer))
        temp = []
        for el in preFlattenShape:
            if el != None:
                temp.append(el)
        preFlattenShape = tuple(temp)
        postFlattenLen = np.prod(temp)

        innerLayer = Flatten()(innerLayer)

        innerLayer = Dense(256, activation="relu")(innerLayer)

        encoded_means = Dense(encodedLen)(innerLayer)
        encoded_log_vars = Dense(encodedLen)(innerLayer)

        encoded_sampled = Lambda(sampling, output_shape=(encodedLen,))([encoded_means, encoded_log_vars])        

        # final encoder declaration
        encoder = Model(decoded_inputs, [encoded_sampled, encoded_means, encoded_log_vars])
        

        # decoder
        encoded_inputs = Input((encodedLen,))

        innerLayer = Dense(256, activation="relu")(encoded_inputs)
        innerLayer = Dense(postFlattenLen, activation="relu")(innerLayer)

        innerLayer = Reshape(preFlattenShape)(innerLayer)

        # duplicate up to three frames, hopefully conv layer with 'same' padding can produce correct changes between frames
        innerLayer = UpSampling3D(size=(frameNum,1,1))(innerLayer)
        
        innerLayer = Conv3D(64, (3,3,3), padding="same")(innerLayer)
        innerLayer = UpSampling3D(size=(1,2,2))(innerLayer)

        # cut if shape doesn't match shape at this point of encoder (ignore channels and frames)
        # since maxpool uses 'same' padding, curShape will only ever be 1 element too large on index 1 or 2
        innerLayer = Cropping3D(((0,0),
                                 (0, 1 if decodedShape[1] % 4 == 2 else 0),
                                 (0, 1 if decodedShape[2] % 4 == 2 else 0)))(innerLayer)
        
        innerLayer = Conv3D(32, (1,3,3), padding="same")(innerLayer)
        innerLayer = UpSampling3D(size=(1,2,2))(innerLayer)

        # cut if shape doesn't match shape at this point of encoder (ignore channels and frames)
        # since maxpool uses 'same' padding, curShape will only ever be 1 element too large on index 1 or 2
        innerLayer = Cropping3D(((0,0),
                                 (0, 1 if decodedShape[1] % 2 == 1 else 0),
                                 (0, 1 if decodedShape[2] % 2 == 1 else 0)))(innerLayer)
        
        innerLayer = Conv3D(16, (1,3,3), padding="same")(innerLayer)

        decoded_outputs = Conv3D(channelNum, (1,1,1))(innerLayer)

        # final decoder declaration
        decoder = Model(encoded_inputs, decoded_outputs)

        # encoder piped into decoder for training
        decoded_outputs = decoder(encoder(decoded_inputs)[0])
        vae = Model(decoded_inputs, decoded_outputs)

        # reconstruction loss function
        reconstruction_loss = binary_crossentropy(decoded_inputs, decoded_outputs)
        reconstruction_loss *= decodedFlattenedLen
        kl_loss = 1 + encoded_log_vars - K.square(encoded_means) - K.exp(encoded_log_vars)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        loss = K.mean(reconstruction_loss + kl_loss)
        vae.add_loss(loss)
        vae.compile(optimizer='Adam')

        return encoder, decoder, vae
