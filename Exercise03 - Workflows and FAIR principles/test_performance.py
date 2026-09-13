import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_exercise_template import *


#------------------
# Histograms 
#------------------

def plot_histograms(probs, y_valid):
    scores_background = np.array([p for p,y in zip(probs, y_valid) if y == 0])
    scores_signal = np.array([p for p,y in zip(probs, y_valid) if y == 1])

    fig, ax = plt.subplots(figsize=(5,4))
    plt.hist(scores_background, density=True, log=False)
    ax.set_title('Background')
    ax.set_xlabel(r'score')
    ax.set_ylabel(r'probability density')
    fname = f"plots/Background_scores.png"
    fig.savefig(fname)
    print("saved", fname)

    fig, ax = plt.subplots(figsize=(5,4))
    ax.hist(scores_signal, density=True, log=False)
    ax.set_title('Signal')
    ax.set_xlabel(r'score')
    ax.set_ylabel(r'probability density')
    fname = f"plots/Signal_scores.png"
    fig.savefig(fname)
    print("saved", fname)

#------------------
# ROC curve 
#------------------

def roc_curve_manual(y, s):
    """
    Manual ROC construction.
    y: (n,) labels in {0,1}
    s: (n,) scores (higher = more 'signal-like')
    Returns: fpr, tpr, thresholds, auc
    """
    # sort by score descending
    order = np.argsort(-s)
    y = y[order]
    s = s[order]

    # cumulative counts as we sweep the threshold down from +inf
    tp = np.cumsum(y == 1).astype(float)
    fp = np.cumsum(y == 0).astype(float)
    P  = max(1.0, (y == 1).sum())
    N  = max(1.0, (y == 0).sum())
    tpr = tp / P
    fpr = fp / N

    # prepend (0,0) at threshold above max, and append (1,1) at threshold below min
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]

    # AUC via trapezoid rule
    auc = np.trapz(tpr, fpr)
    return fpr, tpr, auc

def plot_ROC(y_true, probs, epochs):
    fpr_m, tpr_m, auc_m = roc_curve_manual(y_true, probs)
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(fpr_m, tpr_m, label=f"Manual AUC = {auc_m:.3f}")
    ax.plot([0,1],[0,1],'--',lw=1)
    ax.set_xlabel("False Positive Rate (FPR)")
    ax.set_ylabel("True Positive Rate (TPR)")
    ax.set_title("ROC (manual)")
    ax.legend()
    ax.grid()

    fname = f"plots/ROC_{epochs}epochs.png"
    fig.savefig(fname)
    print("saved", fname)

#------------------
# purity vs threshold , plotted against (1 - t) ----
# -----------------

def plot_purity_vs_thr(t, purity, prevalence, epochs):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.step(t, purity, where="pre", label="purity S/(S+B)")
    ax.scatter(t, purity, s=18)

    # Random-guess baseline (horizontal line at prevalence)
    ax.plot([0,1], [prevalence, prevalence], "--", lw=1, color="gray",
            label=f"random baseline (prevalence={prevalence:.2f})")

    ax.set_xlim(0,1); plt.ylim(0,1.05)
    ax.set_xlabel(r"$t$  (predict positive if score $>= t$)")
    ax.set_ylabel(r"Purity  $S/(S+B)=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$")
    ax.set_title("Purity vs threshold")
    ax.grid(True)
    ax.legend() 

    fname = f"plots/Purity_vs_thr_{epochs}epochs.png"
    fig.savefig(fname)
    print("saved", fname)


# --------------------
# TPR (epsilon_S)
# -------------------

def plot_epyslon_S(t, epsilon_S):
    fig, ax = plt.subplots(figsize=(5,4))
    ax.scatter(t, epsilon_S, s=18)
    ax.set_xlim(0,1)
    ax.set_xlabel(r"$t$  (predict positive if score $>= t$)")
    ax.set_ylabel(r"TPR, epsilon_S")
    ax.grid(True)

    fname = f"plots/TPR_vs_thr.png"
    fig.savefig(fname)
    print("saved", fname)



def run_test_performance(MLP_model):
    #------------------
    # Get model scores (probabilities) of the validation set
    #------------------
    MLP_model['model'].eval()
    with torch.no_grad():
        probs = torch.sigmoid(MLP_model['model'](torch.from_numpy(MLP_model['x_valid_std']).float())).numpy().squeeze()
        # evtl. make it 2-3 lines 

    #------------------
    # Plot probability histograms  
    #------------------
    plot_histograms(probs, MLP_model['y_valid'])

    #------------------
    # Plot ROC curve 
    #------------------
    y_true = MLP_model['y_valid'].astype(int)  # ensure {0,1}
    plot_ROC(y_true, probs, MLP_model['epoch_no'])


    #------------------
    # purity vs threshold , plotted against (1 - t) ----
    # -----------------
    # Purity = S/(S+B) = TP/(TP+FP)

    # Sort by score descending (sweep threshold downward)
    order   = np.argsort(-probs)
    y_sorted= y_true[order]
    s_sorted= probs[order]

    # Cumulative TP/FP as we include items one by one
    tp = np.cumsum(y_sorted == 1).astype(float)
    fp = np.cumsum(y_sorted == 0).astype(float)

    # Purity at each step after including that item
    purity = tp / np.maximum(1.0, tp + fp)

    # Threshold used at each step is t = score
    t = s_sorted

    # Class prevalence = expected purity for random scores (baseline)
    prevalence = y_true.mean()

    plot_purity_vs_thr(t, purity, prevalence, MLP_model['epoch_no'])

    # --------------------
    # TPR (epsilon_S)
    # -------------------
    S0_valid = B0_valid = 0 
    valid_events = np.array([e for e in MLP_model['MC_events'] if e.event_id not in MLP_model['train_ids']])
    for e in valid_events:
        j = e.leading_jet()
        if j is None: 
            continue
        S0_valid += int(j.truth)
        B0_valid += int(not j.truth)


    epsilon_S = tp / S0_valid

    plot_epyslon_S(t, epsilon_S)

    # -----------------
    # Find best threshold & corresponding purity
    # -----------------
    epsilon_limit = 0.7
    mask = (epsilon_S >= epsilon_limit)
    index = np.where(mask == True)
    thr_selected = t[index][0]
    
    print("___________________________")
    print(f"Baseline purity (no cuts) = {purity0:.3f}  with S0={S0}, B0={B0}")
    print("")
    print(f"MC (cuts): purity S/(S+B) = {purity_mc:.3f}  "
    f"with S={S}, B={B}, N={N_sel}  |  eps_S={eps_S:.3f}, eps_B={eps_B:.3f}")
    print("___________________________")
    print(f"Baseline purity (validation split) = {S0_valid/(S0_valid + B0_valid):.3f}  with S0_valid = {S0_valid}, B_valid = {B0_valid}")
    print("")
    print(f"MC (neural network): purity S/(S+B) = {purity[index][0]:.3f} with S = {int(tp[index][0])}, B = {int(fp[index][0])}, N = {int(tp[index][0] + fp[index][0])} | eps_S = {epsilon_limit}, threshold = {thr_selected:.3f}")
    print("___________________________")


    return {
        'threshold': thr_selected,
        'eps_lim': epsilon_limit
    }
