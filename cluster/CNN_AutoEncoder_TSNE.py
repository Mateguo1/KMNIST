"""
Dimensionality Reduction  using Convolutional Autoencoder and T-sne
input:data in shape (28,28,n)
output:data in shape (2,n)
"""
from sklearn.manifold import TSNE
from keras import Model
from keras.layers import Dense,Input,MaxPooling2D,Conv2D,UpSampling2D
from keras.callbacks import TensorBoard

def cnn_AE_Tsne(data):
    x_train4D = data.reshape(data.shape[0], 28, 28, 1).astype('float32')
    x_train_normalize = x_train4D / 255

    input_image = Input((28, 28, 1))

    # Encoder
    encoder = Conv2D(16, (3, 3), padding='same', activation='relu')(input_image)
    encoder = MaxPooling2D((2, 2))(encoder)
    encoder = Conv2D(8, (3, 3), padding='same', activation='softmax')(encoder)
    encoder_out = MaxPooling2D((2, 2))(encoder)
    encoder_model = Model(inputs=input_image, outputs=encoder_out)

    # Decoder
    decoder = UpSampling2D((2, 2))(encoder_out)
    decoder = Conv2D(8, (3, 3), padding='same', activation='softmax')(decoder)
    decoder = UpSampling2D((2, 2))(decoder)
    decoder = Conv2D(16, (3, 3), padding='same', activation='relu')(decoder)
    decoder_out = Conv2D(1, (3, 3), padding='same', activation='sigmoid')(decoder)

    autoencoder = Model(input_image, decoder_out)

    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    #train the model
    autoencoder.fit(x_train_normalize, x_train_normalize, epochs=100, batch_size=256, shuffle=True, verbose=1,
                    callbacks=[TensorBoard(log_dir='autoencoder_cnn_log')])
    #using encoder to get the predicted data
    latent = encoder_model.predict(x_train_normalize)

    # Dimensionality Reduction using T-sne
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    encoded_imgs = tsne.fit_transform(latent.reshape(60000, 7 * 7 * 8))

    return encoded_imgs