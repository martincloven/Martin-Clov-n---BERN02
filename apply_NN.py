import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_exercise_template import *
from train_MLP import feature_vec



#------------------
# plot masses spectra
#-----------------
def mass_hist(sel_masses_nn, epochs, eps):
    bins = 40
    rng  = (60, 140)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(all_masses,      bins=bins, range=rng, density=True, histtype="step", label="All data")
    ax.hist(sel_masses_cuts, bins=bins, range=rng, density=True, histtype="step", label=f"Cuts")
    ax.hist(sel_masses_nn, bins=bins, range=rng, density=True, histtype="step", label=f"Neural Network")
    ax.set_xlabel("Large-R jet mass [GeV]"); ax.set_ylabel("Density")
    ax.set_title("Data: jet mass spectra"); ax.legend(); plt.tight_layout()
    
    fname = f"plots/Mass_histogram_{epochs}epochs.png"
    fig.savefig(fname)
    print("saved", fname)


def apply_nn(MLP_model, performance):   
    thr = performance['threshold']
    eps = performance['eps_lim']
    ATLAS_events = load_events_csv("jets.csv")

    #------------------
    # Scores of data
    # -----------------

    x_real = np.array([feature_vec(e) for e in ATLAS_events])
    x_real_std = MLP_model['scaler'].transform(x_real)  # standardize? -> otherwise not probabilities
    # do I need to extract the leading jets?
    #leading_jets = np.zeros(len(ATLAS_events))


    MLP_model['model'].eval()
    with torch.no_grad():
        probs = torch.sigmoid(MLP_model['model'](torch.from_numpy(x_real_std).float())).numpy().squeeze()
        #are these probabilities? ReLu function and standardization
        # do we have to apply the sigmoid here as well? 

    # ------------------
    # Find mass values (all, cuts, nn)
    # ------------------

    all_masses = []
    sel_masses_cuts = []
    sel_masses_nn = []

    for i,e in enumerate(ATLAS_events):
        j = e.leading_jet(); l = e.leading_lepton()
        if (j is None) or (l is None): #selects events that have both a jet and a lepton 
            continue 
        all_masses.append(j.m)
        if pass_cuts(e):
            sel_masses_cuts.append(j.m)
        if probs[i] >= thr:
            sel_masses_nn.append(j.m)

    #------------------
    # plot masses spectra
    #-----------------

    mass_hist(sel_masses_nn, MLP_model['epoch_no'], eps)