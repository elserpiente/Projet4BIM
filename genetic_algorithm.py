import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random as rd

def genome_generation (length):         # generates a np.array-genome randomly full of number between 0 and 1
    vec = []
    while len(vec) < length:
        vec.append(rd.random())
    return np.array(vec)

# myGenome = genome_generation(100)
# print(myGenome)


t=10

def initialisation (N, T=t):               # creates a population of N genomes of size T
    pop = []
    while len(pop) < N:
        pop.append(genome_generation(T))
    return np.array(pop)

myPop = initialisation(3)
print(myPop)


# Generating a gaussian noise around each point (= vector) with a variance var

var = 0.1
cov = var*np.identity(t)
vec_with_noise = []
for i in range (0, myPop.shape[0]):
    vec_with_noise.append(np.random.multivariate_normal(myPop[i], cov))
print(vec_with_noise)             # make sure that values are in [0,1] !!!


# Crossing-over function
# only if P has more than 1 vector

def crossing_over(P, T, n=12, Tc=1):   # n has to be even
    new_P = np.zeros((n,T))
    
    i = 0
    while i < len(new_P):
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

