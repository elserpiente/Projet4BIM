import numpy as np
import AE_package as ae
#import sklearn.datasets as dt
import genetic_algorithm as ga
from PIL import Image



global faces
"""
faces=dt.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)['data']
f=[]
for i in range(len(faces)):
    if i%10==0:
        f.append(faces[i])
faces=np.array(f)
faces=np.reshape(faces,(len(faces),4096))
"""



def runApp(data):
    global faces

    #autoencoder,encoder,decoder=ae.initAutoencoder()

    autoencoder,encoder,decoder=ae.getAutoencoder()

    #ae.trainAE(autoencoder,faces,1000)
    #ae.saveAutoencoder(autoencoder,encoder,decoder)

    encoded_imgs = encoder.predict(data)

    m = 20

    # if we allow the witness to choose only 1 image, we should only apply the gaussian noise on the face and loop it to have m resulting images
    noisy_vectors = ga.gaussian_noise(encoded_imgs, m=m)
    output_vectors = ga.crossing_over(noisy_vectors)

    decoded_imgs = decoder.predict(output_vectors)

    faces=decoded_imgs


###Currently 3000 cumulated epochs
print("start")
j=1
p=10
n="00000"
faces=[]
for i in range(1000):
    if j%100==0:
        print(j)
    if j==p:
        p*=10
        n=n[0:len(n)-1]
    faces.append(np.array(Image.open("img_align_celeba/"+n+str(j)+".jpg")))
    j+=1

faces=np.reshape(np.array(faces),(1000,178*218*3))
print("end")

    
    
autoencoder,encoder,decoder=ae.initAutoencoder()
ae.trainAE(autoencoder,faces,10)
ae.saveAutoencoder(autoencoder,encoder,decoder)
