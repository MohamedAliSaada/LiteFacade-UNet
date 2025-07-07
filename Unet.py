from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose, MaxPooling2D, Concatenate)
from tensorflow.keras.models import Model

class Unet:
    def __init__(self, input_shape=None, last_activation="relu" , classes=3):
        self.input_shape = input_shape
        self.last_activation = last_activation
        self.classes=classes
        self.model = self.build_model()
    
    def build_model(self):
        
        ########################
        ####     Encoder    ####
        ########################
        
        inputs = Input(shape=self.input_shape)

        conv1 = Conv2D(64, (3, 3), activation="relu", padding='same')(inputs)
        conv1 = Conv2D(64, (3, 3), activation="relu", padding='same')(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), activation="relu", padding='same')(pool1)
        conv2 = Conv2D(128, (3, 3), activation="relu", padding='same')(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = Conv2D(256, (3, 3), activation="relu", padding='same')(pool2)
        conv3 = Conv2D(256, (3, 3), activation="relu", padding='same')(conv3)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = Conv2D(512, (3, 3), activation="relu", padding='same')(pool3)
        conv4 = Conv2D(512, (3, 3), activation="relu", padding='same')(conv4)
        pool4 = MaxPooling2D((2, 2))(conv4)

        ########################
        ####    Bottleneck   ####
        ########################

        bottleneck = Conv2D(1024, (3, 3), activation="relu", padding='same')(pool4)
        bottleneck = Conv2D(1024, (3, 3), activation="relu", padding='same')(bottleneck)

        ########################
        ####     Decoder     ####
        ########################

        upconv1 = Conv2DTranspose(512, (2, 2), strides=2, padding='same')(bottleneck)
        concat1 = Concatenate()([upconv1, conv4])
        conv5 = Conv2D(512, (3, 3), activation="relu", padding='same')(concat1)
        conv5 = Conv2D(512, (3, 3), activation="relu", padding='same')(conv5)

        upconv2 = Conv2DTranspose(256, (2, 2), strides=2, padding='same')(conv5)
        concat2 = Concatenate()([upconv2, conv3])
        conv6 = Conv2D(256, (3, 3), activation="relu", padding='same')(concat2)
        conv6 = Conv2D(256, (3, 3), activation="relu", padding='same')(conv6)

        upconv3 = Conv2DTranspose(128, (2, 2), strides=2, padding='same')(conv6)
        concat3 = Concatenate()([upconv3, conv2])
        conv7 = Conv2D(128, (3, 3), activation="relu", padding='same')(concat3)
        conv7 = Conv2D(128, (3, 3), activation="relu", padding='same')(conv7)

        upconv4 = Conv2DTranspose(64, (2, 2), strides=2, padding='same')(conv7)
        concat4 = Concatenate()([upconv4, conv1])
        conv8 = Conv2D(64, (3, 3), activation="relu", padding='same')(concat4)
        conv8 = Conv2D(64, (3, 3), activation="relu", padding='same')(conv8)

        outputs = Conv2D(self.classes, (1, 1), activation=self.last_activation)(conv8)

        model = Model(inputs, outputs)

        return model



if __name__ == "__main__":
    # Build the model 
    unet = Unet(input_shape=(224, 224, 3), last_activation="sigmoid")
    model = unet.model
    model.summary()
