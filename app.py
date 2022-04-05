import numpy as np
import AE_package as ae
import sklearn.datasets as dt
import genetic_algorithm as ga
from PIL import Image
from matplotlib import pyplot


"""
global faces

faces=dt.fetch_olivetti_faces(data_home=None, shuffle=False, random_state=0, download_if_missing=True)['data']
f=[]
for i in range(len(faces)):
    if i%10==0:
        f.append(faces[i])
faces=np.array(f)
faces=np.reshape(faces,(len(faces),4096))
"""

global faces,decoder,encoder
encoder,decoder=ae.getAutoencoder()
print("AE done")
n="00"
faces=["0020","0109","0119","0136","0229","0300","0307","0311","0377","0398","0450","0604","0619","0620","0689","0797","0902","0908","0971","1027","1094","2624","2668","2671","2725","2736","2743","2766","2768","2816","2900","2931","2935","3020","3036","3061","3066","3071","3077","3147","3364","3687","3690","3747","3773","3806","3864","3868","3886","3890","4018"]
for i in range(len(faces)):
    faces[i]=np.array(Image.open("Images/Beard/"+n+str(faces[i])+".jpg"))/255

def choice(characteristics):
    global faces
    #faces=[]
    for c in characteristics:
        print(c)
        #faces.append(np.array(Image.open("Images/"+c+"/"+n+".jpg")))
        

def runApp(data,var):
    global faces,encoder,decoder

    data=np.reshape(data,(len(data),218*178*3))

    encoded_imgs = encoder.predict(data)

    m = 20

    # if we allow the witness to choose only 1 image, we should only apply the gaussian noise on the face and loop it to have m resulting images

    
    noisy_vectors = ga.gaussian_noise(encoded_imgs, m=m,var=0.05+var.get()/100)
    if len(data)>1:
        output_vectors = ga.crossing_over(noisy_vectors,m)
    else:
        output_vectors = noisy_vectors

    decoded_imgs = decoder.predict(output_vectors)

    faces=np.reshape(decoded_imgs,(len(decoded_imgs),218,178,3))


"""
###Currently 3000 cumulated epochs
print("start")
j=1
p=10
n="00"
faces=["0020","0109","0119","0136","0229","0300","0307","0311","0377","0398","0450","0604","0619","0620","0689","0797","0902","0908","0971","1027","1094","2624","2668","2671","2725","2736","2743","2766","2768","2816","2900","2931","2935","3020","3036","3061","3066","3071","3077","3147","3364","3687","3690","3747","3773","3806","3864","3868","3886","3890","4018"]
for i in range(len(faces)):
    faces[i]=np.array(Image.open("Images/Beard/"+n+str(faces[i])+".jpg"))

faces=np.reshape(np.array(faces),(51,178*218*3))
faces=faces/255
print("end")

#autoencoder,encoder,decoder=ae.initAutoencoder()
#ae.trainAE(autoencoder,faces,10)
#ae.saveAutoencoder(autoencoder,encoder,decoder)

autoencoder,encoder,decoder=ae.getAutoencoder()
#ae.trainAE(autoencoder,faces,200)
#ae.saveAutoencoder(autoencoder,encoder,decoder)

print("deb")
encoded_img = encoder.predict(np.array([faces[0],faces[1]]))
decoded_img = decoder.predict(encoded_img)
faces=np.reshape(np.array([faces[0],faces[1]]),(2,218,178,3))
decoded_img=np.reshape(decoded_img,(2,218,178,3))
print("fin")
ae.testShow(faces,decoded_img)
"""

