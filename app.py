import numpy as np
import AE_package as ae
import sklearn.datasets as dt
import genetic_algorithm as ga

faces=dt.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)['data']
f=[]
for i in range(len(faces)):
    if i%10==0:
        f.append(faces[i])
faces=np.array(f)
faces=np.reshape(faces,(len(faces),4096))
print(len(faces))

#autoencoder,encoder,decoder=ae.initAutoencoder()

autoencoder,encoder,decoder=ae.getAutoencoder()

#ae.trainAE(autoencoder,faces,1000)
#ae.saveAutoencoder(autoencoder,encoder,decoder)

datatest=np.array([faces[20],faces[220]])

encoded_imgs = encoder.predict(datatest)

ae.saveLVect("vector.txt",encoded_imgs)

input_vectors = ae.getLVect('vector.txt')
n = input_vectors.shape[0]+1  # it works with the +1, don't ask
t = input_vectors.shape[1]
noisy_vectors = ga.gaussian_noise(input_vectors, t)
output_vectors = ga.crossing_over(noisy_vectors, T=t, n=n)
ae.saveLVect('new_vectors.txt', output_vectors)


decoded_imgs = decoder.predict(ae.getLVect("new_vectors.txt"))

# We make a new shape of the images vector to allow matplotlib to display our brand new images
datatest=np.array([faces[20],faces[220],faces[10]])
datatest=np.reshape(datatest,(3,64,64,1))
decoded_imgs=np.reshape(decoded_imgs,(3,64,64,1))

ae.testShow(datatest,decoded_imgs)
