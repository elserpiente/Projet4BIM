import numpy as np
import AE_package as ae
import genetic_algorithm as ga
from PIL import Image
from matplotlib import pyplot
from os import walk


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
#encoder,decoder=ae.getAutoencoder()
n="00"
faces=["0020","0109","0119","0136","0229","0300","0307","0311","0377","0398","0450","0604","0619","0620","0689","0797","0902","0908","0971","1027","1094","2624","2668","2671","2725","2736","2743","2766","2768","2816","2900","2931","2935","3020","3036","3061","3066","3071","3077","3147","3364","3687","3690","3747","3773","3806","3864","3868","3886","3890","4018"]
for i in range(len(faces)):
    faces[i]=np.array(Image.open("Images/Beard/"+n+str(faces[i])+".jpg"))/255

def choice(characteristics):
    global faces
    faces=[]
    stock=[]

    f = []
    for c in characteristics:
        for (dirpath, dirnames, filenames) in walk("Images/"+c):
            f.append(filenames)

    for i in range(len(f)):
        if i ==len(f)-1:
            j=0
        else:
            j=i+1
        it_i=0
        it_j=0
        while it_i!=len(f[i]) and it_j!=len(f[j]):
            if f[i][it_i]==f[j][it_j]:
                faces.append("Images/"+characteristics[i]+"/"+f[i][it_i])
                it_i+=1
                it_j+=1
            elif f[i][it_i]<f[j][it_j]:
                it_i+=1
            else:
                it_j+=1
        it_f=0
        for im in f[i]:
            n=0
            for it_f in range(len(faces)):
                if "Images/"+characteristics[i]+"/"+im!=faces[it_f]:
                    n+=1
            if n==len(faces):
                stock.append("Images/"+characteristics[i]+"/"+im)

    n=50-len(faces)
    for i in range(n):
        pick=np.random.randint(0,len(stock))
        faces.append(stock[pick])
        
    for i in range(len(faces)):
        faces[i]=np.array(Image.open(faces[i]))/255
        
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
savefaces=[]
print("start")
char=["Mustache","Beard","Beardless","Pale_Skin","Bald","Fair_Hair","Dark_Hair"]
for c in char:
    choice([c])
    faces=np.reshape(np.array(faces),(len(faces),178*218*3))
    print(len(faces))
    for f in faces:
        savefaces.append(f)
faces=savefaces
print(len(faces))
print("end")

autoencoder,encoder,decoder=ae.initAutoencoder()
ae.trainAE(autoencoder,faces,25)
ae.saveAutoencoder(autoencoder,encoder,decoder)

#autoencoder,encoder,decoder=ae.getAutoencoder()
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
