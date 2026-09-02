#By: Martin Clovén , 2026-09-02
import numpy as np


def Weights_around_x0(x0, x, k):
    C = -(np.log(0.005)/(k**2))
    weights=np.zeros(len(x))
    for i in range(len(x)):
        if(np.abs(x[i]-x0) < k):
            weights[i] = np.e ** (-C * (x[i]-x0)**2)
        else:
            weights[i] = 0
        
    return weights

def calculating_local_regression_one_predictor(y, x, k, weights, beta_hat_0, beta_hat_1):
    Q_matrix = np.zeros((len(beta_hat_0),len(beta_hat_1)))
    
    for i in range(len(beta_hat_1)):
        for j in range(len(beta_hat_1)):
            for k in range(len(x)):
                Q_matrix[i][j] += weights[k]*(y[k] - beta_hat_0[i] - beta_hat_1[j]*x[k])**2
    
    idx = np.unravel_index(np.argmin(Q_matrix), Q_matrix.shape)
    return idx


def y_predictions(x0, x, y, k, beta_hat_0, beta_hat_1):
    y_pred = np.zeros(len(y))
    for i in range(len(x)):
        if(np.abs(x[i]-x0) < k):
            y_pred[i] = beta_hat_0 + beta_hat_1*x[i]
    return y_pred


def standard_error(x0, x, y, y_pred, k):
    e = []
    for i in range(len(x)):
        if(np.abs(x[i]-x0) < k):
            e.append(y[i]-y_pred[i])

    sum = 0
    n = len(e)
    e_avg = np.sum(e) / n
    for i in range(n):
        sum += (-e[i]+e_avg)**2

    se=np.sqrt(sum/(n-2))

    return se
    

def local_regression_one_predictor(y, x, k, x0):
    n = len(x0)
    se = np.zeros(n)
    pred = np.zeros(n)
    Beta_hat_0_examples = np.linspace(1000,-1000,400)
    Beta_hat_1_examples = np.linspace(1000,-1000,400)

    for i in range(n):
        weights = Weights_around_x0(x0[i],x,k)
        idx = calculating_local_regression_one_predictor(y, x, k, weights, Beta_hat_0_examples, Beta_hat_1_examples)
        y_all_pred = y_predictions(x0[i], x, y, k, Beta_hat_0_examples[idx[0]], Beta_hat_1_examples[idx[1]])
        se[i] = standard_error(x0[i], x, y, y_all_pred, k)
        pred[i] = Beta_hat_0_examples[idx[0]] + Beta_hat_1_examples[idx[1]]*x0[i]


    return pred, se



data = np.loadtxt(open('pollution_cleaneddata.csv', "rb"), delimiter=",", skiprows=1)

x = data[:,10]
y = data[:,15]
x0 = [10, 18, 25]
k = 2   # units in x that are regarded for the local_regression

pred, se = local_regression_one_predictor(y, x, k, x0)

for i in range(len(pred)):
    print("at x=",x0[i]," , pred = ", pred[i]," , se: ", se[i], sep="")

