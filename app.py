import numpy as np
import AE_package as ae
import sklearn.datasets as dt


faces=dt.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)['data']
faces=np.reshape(faces,(len(faces),4096))

autoencoder,encoder,decoder=ae.initAutoencoder()

ae.trainAE(autoencoder,faces,500)

encoded_imgs = encoder.predict(faces[19:24])

ae.saveLVect("vector.txt",encoded_imgs)


decoded_imgs = decoder.predict(ae.getLVect("vector.txt"))

# We make a new shape of the images vector to allow matplotlib to display our brand new images
datatest=np.reshape(faces[19:24],(5,64,64,1))
decoded_imgs=np.reshape(decoded_imgs,(5,64,64,1))

ae.testShow(datatest,decoded_imgs)
