#By: Martin Clovén , 2026-09-06

###-------- Imports -----------
import time
import random
import numpy as np
import math
from csv_functions import *

###-------- Functions -----------

##--- Functions for max likelilood -----

def pred(x, beta_0, beta_1):
    val = np.exp(beta_0 + beta_1 * x)
    return val

# Math to calculate the likelihood
def likelihood(x,y,b_0, b_1):
    t_1 = y*(b_0 + b_1*x)
    t_2 = -np.exp(b_0 + b_1*x)
    t_3 = -math.log(math.factorial(int(y)))
    return t_1 + t_2 + t_3

# Sum matrix over all presented values for beta_0 and beta_1
def likelihood_sum_matrix(x, y, beta):
    n = len(beta[0])
    sum_matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            for k in range(len(y)):
                sum_matrix[i][j] += likelihood(x[k], y[k],beta[0][i], beta[1][j])
    return sum_matrix


def max_likelihood(x, y, beta):
    sum_matrix = likelihood_sum_matrix(x, y, beta)
    idx = np.unravel_index(np.argmax(sum_matrix), sum_matrix.shape)
    return idx

##--- Functions for response Y from poisson cumulative probabilities

def Poisson_probability_function(lam, y):
    prob = (math.exp(-lam) * lam**y)/(math.factorial(y))
    return prob
    
# does not accept negative lambda
def Poisson_GLM_to_response(beta_0, beta_1, x):

    lam = math.exp(beta_0 + beta_1 * x) # rate parameter calculation
    if(lam < 0):    
        print("Error in: Poisson_GLM_to_response(...) ")

    first_y = int(np.round(lam)) # a good value to start at for probability is at its mean: lambda
    first_prob = Poisson_probability_function(lam, first_y)

    response_arr = [first_y]
    total_prob_arr = [first_prob]
    
    is_prob_low = True
    
    lowest_y = first_y
    highest_y = first_y
    loop = 0
    while(is_prob_low == True):

        if (lowest_y > 0):
            lowest_y = lowest_y - 1
            lowest_y_prob = Poisson_probability_function(lam, lowest_y)
            response_arr.append(lowest_y)
            total_prob_arr.append(lowest_y_prob + total_prob_arr[-1])

        highest_y = highest_y + 1
        #print("lam**Y:", lam**highest_y)
        highest_y_prob = Poisson_probability_function(lam, highest_y)
        response_arr.append(highest_y)
        total_prob_arr.append(highest_y_prob + total_prob_arr[-1])


        if(total_prob_arr[-1] > 0.995):  #Stop when the commuluative probability is over 99.5 %
            is_prob_low = False
        if(loop > 100):
            is_prob_low = False
        loop += 1
    chosen_prob = random.uniform(0, total_prob_arr[-1])
    #print("total_prob_arr[-1]: ", total_prob_arr[-1], " , chosen_prob:", chosen_prob)
    idx = np.abs(np.array(total_prob_arr) - chosen_prob).argmin()
    # getting the index closest to 

    new_response = response_arr[idx]
    return new_response

##--- Functions to manipulate data -----

# To get a double array containing all tested betas
def initial_betas(a_L, a_U, b_L, b_U):
    double_array = np.array([
        np.linspace(a_L, a_U, 250),
        np.linspace(b_L, b_U, 250)
    ])
    return double_array

# Need small x predictors when calculatin np.exp(b_0 + b_1*x)
def minimize_x_inputs(x_predictor, x_ask):
    x_ask = x_ask - np.median(x_predictor)  # using the median is prefered since it results in the smallest values |x|
    return x_ask


##--- Main function -----

# Reason behind "nr_of_samples" is to reuse beta_0 & beta_1, no need to recalculate
def main(x, y, nr_of_samples):
    start_time = time.perf_counter()
    preds = np.zeros((nr_of_samples*len(y), len(x)), dtype=int) # allows us to store all of our responses at one place
    for i in range(len(y)):

        beta = initial_betas(5, -5, 5, -5)
        inx = max_likelihood(x, y[i], beta)
        
        beta = initial_betas(beta[0][inx[0]-1], beta[0][inx[0]+1], beta[1][inx[1]-1], beta[1][inx[1]+1])
        inx = max_likelihood(x, y[i], beta)
        
        print("b_0: ", beta[0][inx[0]], " | b_1: ", beta[1][inx[1]])



            #print("lamda: ", math.exp(beta[0][inx[0]] + beta[1][inx[1]] * x))
        for n in range(nr_of_samples):
            for j in range(len(x)):
                preds[2*n + i][j] = int(Poisson_GLM_to_response(beta[0][inx[0]], beta[1][inx[1]], x[j]))
                #print("x =", x[j], " , y =" , preds[2*n + i][j])
                
    main_calc_time = time.perf_counter() - start_time
    print("Main calculation time: ", main_calc_time , "/s")
    return preds


###-------- Calling functions -----------

data = read_csv('/Users/martincloven/Documents/University/Computational Physics/BERN02/Repo/Martin-Clov-n---BERN02/Exercise02 - Generalised Linear Models/bird_count.csv')
x = data[:,1]
y = [data[:,0], data[:,2]]
x_minimized = minimize_x_inputs(x,x)    # needs small x-values smaller than 50 for "np.exp(b_0 + b_1*x)" in: likelihood(x,y,b_0, b_1): 
nr_of_samples = 3

preds = main(x_minimized, y, nr_of_samples)

for i in range(len(x)):
    print("x =", x[i], " , y =" , y[0][i])

output_data = np.vstack((x,preds))
###-------- Printing data -----------
'''
for i in range(len(x_ask)):
    print("Count: ",output_data[0][i], 
        " , Yr: ",output_data[1][i], 
        " , ObserverAge: ", output_data[2][i]
    )
'''

###-------- Saving data -----------

header = nr_samples_savetxt_header(nr_of_samples)   # dynamically changing header size depending on nr_of_samples

np.savetxt(
    "output.csv",
    np.transpose(output_data),  # needs to be transposed to be saved in the asked for format 
    fmt="%d",   # without this format the values will be in eg: 2.0000000000000e+03 instead of 2003
    delimiter=",",
    header=header,
    comments=""

)
