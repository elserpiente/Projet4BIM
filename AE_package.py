import numpy as np

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

