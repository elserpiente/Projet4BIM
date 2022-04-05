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
        vec.append(rd.randint(1,10))
    return np.array(vec)




def initialisation (N, T):               # creates a population of N genomes of size T
    pop = []
    while len(pop) < N:
        pop.append(genome_generation(T))
    return np.array(pop)



###########################################################################################################################################



# Generating a gaussian noise around each point (= vector) with a variance var

def gaussian_noise (P, m, var):
    """ Apply a gaussian noise on the values of one or several vectors.

    Parameters
    ----------
	P : numpy array
		Numpy array of vector(s) on which the noise will be applied 
	m : int
		Number of noisy vectors returned if only one vector is presented in the input array 
	var : float
		Variance of the gaussian noise, in [0,1]

	Returns
	-------
	numpy array
		Numpy array of noisy vectors of positive values (either m vectors or the same number as in the input array)
    
    Example
    -------
    >>> P = np.array([[0.1, 0.5, 0.4, 0.3],
                      [0.8, 0.2, 0.2, 0.9]])
    >>> var = 0.05
	>>> m = 2 # useless here, becaus P contains more than 1 vector
	>>> gaussian_noise (P, m, var)
	array([[0.14, 0.47, 0.46, 0.23],
           [0.85, 0.19, 0.22, 0.88]])
    """

    cov = var*np.identity(P.shape[1])
    vec_with_noise = []
    if P.shape[0] == 1:
        for i in range (m):
            vec_with_noise.append(np.random.multivariate_normal(P[0], cov))
    else:
        for i in range (0, P.shape[0]):
            vec_with_noise.append(np.random.multivariate_normal(P[i], cov))
            
    array_with_noise = np.array(vec_with_noise) 
    positive_array_with_noise = np.abs(vec_with_noise)   # makes sure that values are > 0 
    return positive_array_with_noise           




# Crossing-over function
# only if P has more than 1 vector

def crossing_over (P, m, Tc=1):   
    """ Recombinate all the vectors of a numpy array, by exchanging heads and tails of random length.

    Parameters
    ----------
	P : numpy array
		Numpy array of at least 2 vectors on which the noise will be applied 
	m : int
		Number of recombined vectors returned
	Tc : float
		Probability of recombination, in [0,1], set to 1 by default

	Returns
	-------
	numpy array
		Numpy array of m recombined vectors

	Raises
	------
	ValueError
		If P contains less than 2 vectors.
    
    Example
    -------
    >>> P = np.array([[0.1, 0.5, 0.4, 0.3],
                      [0.8, 0.2, 0.2, 0.9]])
	>>> m = 2
	>>> crossing_over (P, m)
	array([[0.1, 0.5, 0.2, 0.9],
           [0.8, 0.2, 0.4, 0.3]])
    """
    
    if P.shape[0] <= 1:
    	raise ValueError("The input array P has to contain at least 2 vectors.")

    new_P = np.zeros((m, P.shape[1]))

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
            if i+1 < new_P.shape[0]:
                new_P[i+1] = tmp2

            i += 2
          
    return new_P


#########################################################################################################################################
### Unitary tests ###
#########################################################################################################################################

def unitary_test_1_vector_cross ():

	try:
		myPop = initialisation(1, 10)
		print(myPop)
		myNoisyPop = gaussian_noise(P=myPop, m=6, var=0.05)
		print(myNoisyPop)
		print(crossing_over(P=myNoisyPop, m=6))
		return True
	except ValueError:
		return False

print(unitary_test_1_vector_cross())


def unitary_test_1_vector_noise():
	myPop = initialisation(1, 10)
	print(myPop)
	m = 3
	myNoisyPop = gaussian_noise(P=myPop, m=m, var=0) # we set a nil variance so that the output should be 3 times the same vector
	print(myNoisyPop)
	if myNoisyPop.shape[0] == m:
		if myNoisyPop[0] == myNoisyPop[1] and myNoisyPop[1] == myNoisyPop[2]:
			return True
	else:
		return False

print(unitary_test_1_vector_noise())


"""
# Main loop
for i in range(50):
    print(rd.randint(0, 3))

#input_vectors = getLVect('vector.txt')
input_vectors=[[0.3,0.1,0.5],[0.2,0.6,0.2],[0.2,0.8,0.2]]
input_vectors=np.array(input_vectors)
print(input_vectors)
n = input_vectors.shape[0]  
t = input_vectors.shape[1]
noisy_vectors = gaussian_noise(input_vectors)
print(noisy_vectors)
output_vectors = crossing_over(noisy_vectors)
print(output_vectors)

"""
