import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random as rd


#########################################################################################################################################
### Only for tests ###
#########################################################################################################################################

def genome_generation (length):         # generates a np.array-genome randomly full of number between 0 and 1
    vec = []
    while len(vec) < length:
        vec.append(rd.random())
    return np.array(vec)

# myGenome = genome_generation(100)
# print(myGenome)



def initialisation (N, T):               # creates a population of N genomes of size T
    pop = []
    while len(pop) < N:
        pop.append(genome_generation(T))
    return np.array(pop)

# myPop = initialisation(3)
# print(myPop)

###########################################################################################################################################



# Generating a gaussian noise around each point (= vector) with a variance var

def gaussian_noise (P, T, var = 0.05):
    cov = var*np.identity(T)
    vec_with_noise = []
    for i in range (0, P.shape[0]):
        vec_with_noise.append(np.random.multivariate_normal(P[i], cov))
    array_with_noise = np.array(vec_with_noise) 
    positive_array_with_noise = np.abs(vec_with_noise)
    return positive_array_with_noise           # make sure that values are > 0 !!!


# Crossing-over function
# only if P has more than 1 vector

def crossing_over (P, T, n, Tc=1):   # n has to be even
    new_P = np.zeros((n,T))
    
    i = 0
    while i < len(new_P)-1:
        if rd.random() < Tc:                                                    # only works with a certain probability Tc --> here we want as many crossings as there are genomes
            indc1 = 0
            indc2 = 0
            while indc1 == indc2 :                                              # to get sure to cross with another genome
                indc1 = rd.randint(0, P.shape[0]-1)                             # index in population of another random genome 
                indc2 = rd.randint(0, P.shape[0]-1)    
            posc = rd.randint(0, P.shape[1]-1)                                  # position of the first gene of the swapped sequences 

            tmp1 = P[indc1,:]
            tmp2 = P[indc2,:]
            tmp1_tail = tmp1[posc:P.shape[1]]                                   # we store the first swapped sequence (end of genome i)
            tmp1[posc:P.shape[1]] = tmp2[posc:P.shape[1]]                       # we replace the end of i by the end of indc
            tmp2[posc:P.shape[1]] = tmp1_tail                                   # we replace the end of indc by the end of i

            new_P[i] = tmp1
            new_P[i+1] = tmp2

            i += 2
          
    return new_P

# print(crossing_over(P=myPop, T=10))




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



# main 

input_vectors = getLVect('vector.txt')
n = input_vectors.shape[0]+1  # it works with the +1, don't ask
t = input_vectors.shape[1]
noisy_vectors = gaussian_noise(input_vectors, t)
output_vectors = crossing_over(noisy_vectors, T=t, n=n)
print(output_vectors)
saveLVect('new_vectors.txt', output_vectors)


