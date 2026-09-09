import numpy as np

def read_csv(file_name):
    data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)
    return data

def nr_samples_savetxt_header(nr_of_samples):
    header = "yr"
    for i in range(nr_of_samples):
        header +=f",Count:{i+1},ObserverAge:{i+1}"
    return header