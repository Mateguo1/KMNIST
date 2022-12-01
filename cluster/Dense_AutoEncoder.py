"""
Dimensionality Reduction using Dense_AutoEncoder
input:data in shape (784,n), Dimension
output:data in shape (Dimension,n)
"""

from keras.layers import *
from keras.models import Model

def dense_AE(data,dim):
    x_train = data / 255
    x_train = x_train.reshape((x_train.shape[0], -1))

    encoding_dim = dim

    # input  28*28
    input_img = Input(shape=(784,))

    # Encoder
    encoded = Dense(128, activation='relu')(input_img)
    encoded = Dense(64, activation='relu')(encoded)
    encoded = Dense(32, activation='relu')(encoded)
    encoded = Dense(10, activation='relu')(encoded)
    encoder_output = Dense(encoding_dim, activation='tanh')(encoded)

    # Decoder
    decoded = Dense(10, activation='relu')(encoder_output)
    decoded = Dense(32, activation='relu')(decoded)
    decoded = Dense(64, activation='relu')(decoded)
    decoded = Dense(128, activation='relu')(decoded)
    decoded = Dense(784, activation='tanh')(decoded)

    autoencoder = Model(inputs=input_img, outputs=decoded)
    encoder = Model(inputs=input_img, outputs=encoder_output)
    autoencoder.compile(optimizer='adam', loss='mse')

    # train the model
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True)

    encoded_imgs = encoder.predict(x_train)

    return encoded_imgs