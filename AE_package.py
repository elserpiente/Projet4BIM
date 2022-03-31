import numpy as np
import keras
from keras import layers
from keras.callbacks import TensorBoard
from matplotlib import pyplot

        
def initAutoencoder():
    latent_dim=512
    #######################################################################################################################################################################
    ###Layers###
    #######################################################################################################################################################################

    # Definition of the input -> which shape we pass in parameter
    input_img = keras.Input(shape=(178*218*3,))
    # Our first and only encoder layer which takes our input_img as input
    encoded = layers.Dense(latent_dim, activation='relu',name='encoder')(input_img)
    # Our first and only decoder layer which takes the output of the encoder as input
    decoded = layers.Dense(178*218*3, activation='sigmoid',name='decoder')(encoded)

    # We create the model of our autoencoder which takes the input images in
    # argument and return the decoded images
    autoencoder = keras.Model(input_img, decoded)

    #######################################################################################################################################################################
    ###Encoder###
    #######################################################################################################################################################################

    # We create the model of our encoder which takes the input images in
    # argument and return the encoded images
    encoder = keras.Model(input_img, encoded)
    encoder.compile()

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
    decoder.compile()


    # By now we have 3 main things, our autoencoder, our encoder and our decoder
    # they aren't in the same variables but use the same layers so when we train wether one
    # or the other we train each layer for each things
    return autoencoder,encoder,decoder

def trainAE(autoencoder,data,epochs):
    #######################################################################################################################################################################
    ###Training###
    #######################################################################################################################################################################
    
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Here we choose a number of 500 epochs and our data to train our autoencoder
    # doing more epochs doesn't make a lot of sense because the loss doesn't get smaller
    # it's mostly because the training images are really basic
    autoencoder.fit(data, data,epochs=epochs,batch_size=256, shuffle=True,validation_data=(data, data),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])

def saveAutoencoder(autoencoder,encoder,decoder):
    autoencoder.save('./AE/ae.h5')
    encoder.save('./AE/encoder.h5')
    decoder.save('./AE/decoder.h5')

def getAutoencoder():
    latent_dim=512

    autoencoder = keras.models.load_model('./AE/ae.h5')
    
    encoder = keras.models.load_model('./AE/encoder.h5')

    decoder = keras.models.load_model('./AE/decoder.h5')

    return autoencoder,encoder,decoder
    
    
def testShow(dataIn,dataOut):
    # Display of images (not interesting we won't use it in the final product)
    # just to see all the images

    n=len(dataIn)  # How many digits we will display
    pyplot.figure(figsize=(20, 4))
    for i in range(n):
        # Display original
        ax = pyplot.subplot(2, n, i + 1)
        pyplot.imshow(dataIn[i])
        pyplot.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = pyplot.subplot(2, n, i + 1 + n)
        pyplot.imshow(dataOut[i])
        pyplot.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    pyplot.show()


def save_in(f_name,vectors):
    fichsauv = open(f_name,"w")
    for vector in vectors:
        dim=len(vector)
        str_vect=""
        for n in vector:
            str_vect=str_vect+str(n)+' '
        fichsauv.write(str_vect+"\n")


def saveLVect(file_name,vectors):
    """
    Take in argument a text file and a list of vectors
    Save them in a file with one vector per line and each value separated with a space
    Ex:
       
        input vector :
                [[0.1,0.5,0.4][0.8,0.2,0.2]]
        file :
                0.1 0.5 0.4/n
                0.8 0.2 0.2/n
    """
    f = open(file_name,"w")
    for vector in vectors:
        dim=len(vector)
        str_vect=""
        for n in vector:
            str_vect=str_vect+str(n)+' '
        f.write(str_vect+"\n")
    print("Saved successfully")

def getLVect(file_name):
    """
    Take in argument a text file with one vector saved in one line with each value separated by a space
    Return a numpy array of shape (number of vector/line,number of value in each vector)
    Ex:
        file :
                0.1 0.5 0.4/n
                0.8 0.2 0.2/n
        output vector :
                [[0.1,0.5,0.4][0.8,0.2,0.2]]
    """
    f=open(file_name,'r')
    nb_vect=0
    vect=[]
    for l in f:
        nb_vect+=1
        n=""
        len_vect=0
        for char in l:
            if char==' ':
                vect.append(float(n))
                n=""
                len_vect+=1
            elif char!='\\' and char!='n' :
                n+=char
    vectors=np.ones(nb_vect*len_vect)
    vectors=vectors*vect
    return(np.reshape(vectors,(nb_vect,len_vect)))


from tensorflow.keras.callbacks import Callback
import numpy as np
import os

                
class BestModelCallback(Callback):

    def __init__(self, filename= './run_dir/best-model.h5', verbose=0 ):
        self.filename = filename
        self.verbose  = verbose
        self.loss     = np.Inf
        os.makedirs( os.path.dirname(filename), mode=0o750, exist_ok=True)
                
    def on_train_begin(self, logs=None):
        self.loss = np.Inf
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get("loss")
        if current < self.loss:
            self.loss = current
            self.model.save(self.filename)
            if self.verbose>0: print(f'Saved - loss={current:.6f}')
