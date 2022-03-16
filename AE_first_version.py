import numpy as np
import keras
from keras import layers
from keras.callbacks import TensorBoard

# DEFs
#autoencoder = encoder + decoder | image 28*28*3 -> image 28*28*3
#encoder | image 28*28*3 -> vector x
#decoder | vector x -> image 28*28*3


# Number of values in the latent space, the encoder will giva an array of
# x values in output
latent_dim=512


from matplotlib import image
from matplotlib import pyplot
# Load images as pixel array
image1 = image.imread('imgAE/face_m_m_b.png')
image2 = image.imread('imgAE/face_m_b_b.png')
image3 = image.imread('imgAE/face_m_m_r.png')
image4 = image.imread('imgAE/face_m_m_v.png')
image5 = image.imread('imgAE/face_m_b_r.png')
image6 = image.imread('imgAE/face_m_m_ma.png')

# A way to display images
pyplot.imshow(image1)
pyplot.show()

# The shape of the images 28 pixels by 28 pixels with 3 values per pixel
data=np.reshape(np.asarray(image1),(28*28*3))
data2=np.reshape(np.asarray(image2),(28*28*3))
data3=np.reshape(np.asarray(image3),(28*28*3))
data4=np.reshape(np.asarray(image4),(28*28*3))
data5=np.reshape(np.asarray(image5),(28*28*3))
data6=np.reshape(np.asarray(image6),(28*28*3))

# The list of values we will use to train and test our autoencoder
datatest=np.array([data,data2,data3,data4,data5,data6])
data=np.array([data,data2,data3,data4,data,data,data,data2,data3,data4,data,data2,data3,data4,data,data,data2,data,data,data2,data3,data4,data,data2,data])
#######################################################################################################################################################################
###Layers###
#######################################################################################################################################################################

# Definition of the input -> which shape we pass in parameter
input_img = keras.Input(shape=(28*28*3,))
# Our first and only encoder layer which takes our input_img as input
encoded = layers.Dense(latent_dim, activation='relu')(input_img)
# Our first and only decoder layer which takes the output of the encoder as input
decoded = layers.Dense(28*28*3, activation='sigmoid')(encoded)

# We create the model of our autoencoder which takes the input images in
# argument and return the decoded images
autoencoder = keras.Model(input_img, decoded)

#######################################################################################################################################################################
###Encoder###
#######################################################################################################################################################################

# We create the model of our encoder which takes the input images in
# argument and return the encoded images
encoder = keras.Model(input_img, encoded)

#######################################################################################################################################################################
###Decoder###
#######################################################################################################################################################################

# We define the input of our decoder, a vector of the size of our latent space
encoded_input = keras.Input(shape=(latent_dim,))
# We get the last layer of our autoencoder so the only layer of our decoder
decoder_layer = autoencoder.layers[-1]
# We create the model of our decoder which takes the vector of our latent space in
# argument and return this vector decoded by our decoder layer
decoder = keras.Model(encoded_input, decoder_layer(encoded_input))


# By now we have 3 main things, our autoencoder, our encoder and our decoder
# they aren't in the same variables but use the same layers so when we train wether one
# or the other we train each layer for each things


#######################################################################################################################################################################
###Training###
#######################################################################################################################################################################

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Here we choose a number of 500 epochs and our data to train our autoencoder
# doing more epochs doesn't make a lot of sense because the loss doesn't get smaller
# it's mostly because the training images are really basic
autoencoder.fit(data, data,epochs=500,batch_size=256, shuffle=True,validation_data=(data, data),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
#######################################################################################################################################################################
###Checks###
#######################################################################################################################################################################

# Here we pass our test dataset in the encoder then in the decoder to check how
# well it encode and decode this image.
# Here there is 4 images that were in the training dataset and some new images
# the goal is to see how it goes with unknown features
encoded_imgs = encoder.predict(datatest)
decoded_imgs = decoder.predict(encoded_imgs)

# We make a new shape of the images vector to allow matplotlib to display our brand new images
datatest=np.reshape(datatest,(6,28,28,3))
decoded_imgs=np.reshape(decoded_imgs,(6,28,28,3))

"""
#######################################################################################################################################################################
###Euclidian distance###
#######################################################################################################################################################################

# We encoded 2 images we make a mean of their values and put it in a new vector that will
# be the one decoded to check what it does
# ex : [1,0,0] shuffle with [0,0,1] gives [0.5,0,0.5]
#         Red       +        Blue    =    Dark magenta
# The vector provided by our latent space isn't made of pixel values but it's just
# to illustrate

created_img=np.zeros((1,512))
for i in range(0,512):
    created_img[0][i]=(encoded_imgs[0][i]+encoded_imgs[1][i])/2

#######################################################################################################################################################################
###Crossing Over###
#######################################################################################################################################################################

# Here we chose randomly a value out of our images and put it in a new array that will be decoded
# ex : [1,0,0] suffle with [0,0,1] gives [0,0,0] or [0,0,1] or [1,0,0] or [1,0,1]
#       Red         +       Blue    =     Black  or  Blue   or  Red    or  Magenta
# The vector provided by our latent space isn't made of pixel values but it's just
# to illustrate

for i in range(0,512):
    r=np.random.rand()
    if r<0.5:
        created_img[0][i]=encoded_imgs[0][i]
    else:
        created_img[0][i]=encoded_imgs[1][i]

#######################################################################################################################################################################
###Display###
#######################################################################################################################################################################


# We recreate everything using our decoder and here I concatenate the whole new image
# with the 2 we used to create it just to display them
decoded_imgs = np.concatenate((decoder.predict(encoded_imgs), decoder.predict(created_img)), axis=0)
datatest=np.reshape(datatest,(2,28,28,3))
decoded_imgs=np.reshape(decoded_imgs,(3,28,28,3))

# Display of images (not interesting we won't use it in the final product)
# just to see all the images

n = 2  # How many digits we will display
m = 3  # How many digits we will display
pyplot.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = pyplot.subplot(2, m, i + 1)
    pyplot.imshow(datatest[i])
    pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i in range(m):
    # Display reconstruction
    ax = pyplot.subplot(2, m, i + 1 + m)
    pyplot.imshow(decoded_imgs[i])
    pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
pyplot.show()
"""

# Display of images (not interesting we won't use it in the final product)
# just to see all the images

n = 6  # How many digits we will display
pyplot.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = pyplot.subplot(2, n, i + 1)
    pyplot.imshow(datatest[i])
    pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = pyplot.subplot(2, n, i + 1 + n)
    pyplot.imshow(decoded_imgs[i])
    pyplot.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
pyplot.show()




#
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# Under these lines there is a lot more ununderstandable code I didn't succeed to run it
# We can probably use it in the future to improve the autoencoder up here but I won't comment it
# as I don't know how to make it working 😅 
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!









"""
#######################################################################################################################################################################
###Encoder###
#######################################################################################################################################################################

inputs=keras.Input(shape=(28,28,1))
x=layers.Conv2D(32,3,strides=2,padding="same",activation="relu",name="layer1")(inputs)
x=layers.Conv2D(64,3,strides=2,padding="same",activation="relu",name="layer2")(x)
x=layers.Conv2D(64,3,strides=2,padding="same",activation="relu",name="layer3")(x)
x=layers.Conv2D(64,3,strides=2,padding="same",activation="relu",name="layer4")(x)
x=layers.Flatten()(x)
x=layers.Dense(512,activation="relu",name="layer5")(x)

z_mean=layers.Dense(latent_dim,name="z_mean")(x)
z_log_var=layers.Dense(latent_dim,name="z_log_var")(x)


epsilon=keras.backend.random_normal(shape=(batch_size,latent_dim))
z=z_mean+tf.exp(0.5*z_log_var)*epsilon


encoder=keras.Model(inputs,[z_mean,z_log_var,z],name="encoder")

encoder.compile()

inputs=keras.Input(shape=(28,28,1))

z_mean,z_log_var,z=encoder(inputs)

#######################################################################################################################################################################
###Decoder###
#######################################################################################################################################################################

inputs=keras.Input(shape=(latent_dim,))
x=layers.Dense(7*7*64,activation="relu")(inputs)
x=layers.Reshape((7,7,64))(x)
x=layers.Conv2DTranspose(64,3,strides=1,padding="same",activation="relu")(x)
x=layers.Conv2DTranspose(64,3,strides=2,padding="same",activation="relu")(x)
x=layers.Conv2DTranspose(32,3,strides=2,padding="same",activation="relu")(x)
outputs=layers.Conv2DTranspose(1,3,padding="same",activation="sigmoid")(x)

decoder=keras.Model(inputs,outputs,name="decoder")
decoder.compile()
"""

#######################################################################################################################################################################
###VAE###
#######################################################################################################################################################################


#######################################################################################################################################################################
###Callback###
#######################################################################################################################################################################


#######################################################################################################################################################################
###Training###
#######################################################################################################################################################################


#######################################################################################################################################################################
###Checks###
#######################################################################################################################################################################

print("done")
