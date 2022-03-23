import numpy as np
import AE_package as ae
import sklearn.datasets as dt
import genetic_algorithm as ga


global faces
faces=dt.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)['data']
f=[]
for i in range(len(faces)):
    if i%10==0:
        f.append(faces[i])
faces=np.array(f)
faces=np.reshape(faces,(len(faces),4096))


def runApp(data):
    global faces

    #autoencoder,encoder,decoder=ae.initAutoencoder()

    autoencoder,encoder,decoder=ae.getAutoencoder()

    #ae.trainAE(autoencoder,faces,1000)
    #ae.saveAutoencoder(autoencoder,encoder,decoder)

    encoded_imgs = encoder.predict(data)

    n = encoded_imgs.shape[0]  # it works with the +1, don't ask
    t = encoded_imgs.shape[1]
    m = 20
    noisy_vectors = ga.gaussian_noise(encoded_imgs, t)
    output_vectors = ga.crossing_over(noisy_vectors, T=t, n=n,m=m)

    decoded_imgs = decoder.predict(output_vectors)

    faces=decoded_imgs


###Currently 3000 cumulated epochs
    
#autoencoder,encoder,decoder=ae.getAutoencoder()
#ae.trainAE(autoencoder,faces,1000)
#ae.saveAutoencoder(autoencoder,encoder,decoder)
