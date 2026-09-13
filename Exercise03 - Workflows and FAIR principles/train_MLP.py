import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_exercise_template import *
import copy

#-----------------
#Helpers 
#-----------------

def dR(a_eta,b_eta, a_phi, b_phi):
    eta_diff = a_eta-b_eta
    return eta_diff**2 + dphi(a_phi, b_phi)**2

#-----------------
# Feature vector 
#-----------------

def feature_vec(e: Event):
    j = e.leading_jet() #just the leading ones?
    l = e.leading_lepton()
    dphi_jl = dphi(j.phi, l.phi)
    dR_jl = dR(j.eta, l.eta, j.phi, l.phi)
    frac_pT = j.pt/l.pt
    frac_pT_diff = (j.pt - l.pt)/(j.pt + l.pt)

    x = np.array([j.pt, j.eta, j.phi, l.pt, l.eta, l.phi, dphi_jl, dR_jl, frac_pT, frac_pT_diff])
    return x 

#---------------------------------
# Create Dataset and DataLoader
#---------------------------------

class MC_Data(torch.utils.data.Dataset):
    def __init__(self, x_in, y_in):
        self.x = torch.from_numpy(x_in)
        self.y = torch.from_numpy(y_in)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return [self.x[idx], self.y[idx]]


#---------------------------------
# Run the actual MLP 
#---------------------------------

def run_MLP1():
    #-----------------
    # load data & shuffle & split data (60/40)
    #-----------------
    #shuffle before or after splitting?
    MC_events = load_events_csv("pythia.csv")

    event_ids = []

    for e in MC_events:
        event_ids.append(e.event_id)

    np.random.seed(42) #shuffles always in the same way -> reproducible results 
    np.random.shuffle(event_ids)

    split = int(0.6 * len(event_ids))
    train_ids = event_ids[:split]

    x_train = np.array([feature_vec(e) for e in MC_events if e.event_id in train_ids])
    x_valid = np.array([feature_vec(e) for e in MC_events if e.event_id not in train_ids])

    #extract truth value 
    y_train = np.array([e.leading_jet().truth for e in MC_events if e.event_id in train_ids])
    y_valid = np.array([e.leading_jet().truth for e in MC_events if e.event_id not in train_ids])

    #------------------
    # Standardizing 
    #------------------
    #mu = mean of that feature in the training set 
    #sigma = standard deviation of the traning set 

    scaler = StandardScaler() # function from sklearn that does this
    x_train_std = scaler.fit_transform(x_train)
    x_valid_std = scaler.transform(x_valid)

    #------------------
    # small MLP
    #------------------

    # Create the training data as a DataLoader Object
    train_dataset = MC_Data(x_train_std, y_train)
    train_data = torch.utils.data.DataLoader(train_dataset, batch_size=100)

    # Create the validation data as a DataLoader Object
    valid_dataset = MC_Data(x_valid_std, y_valid)
    valid_data = torch.utils.data.DataLoader(valid_dataset, batch_size=100)

    #define MLP:
    in_dim  = 10 # Number of input dimensions
    h1, h2  = 32, 16  # Two hidden layers with h1 and h2 neurons

    # Construct the MLP
    model = nn.Sequential(
        nn.Linear(in_dim, h1),
        nn.ReLU(),
        nn.Dropout(0.1), #to avoid over-training 
        nn.Linear(h1, h2),
        nn.ReLU(),
        nn.Linear(h2, 1),
        #nn.Sigmoid()  # output prob
    )

    print(model)
    print("Number of trainable parameters:", sum(p.numel()
          for p in model.parameters() if p.requires_grad))

    loss_function = nn.BCEWithLogitsLoss() #nn.MSELoss() # Mean-squared error
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9) # Stochastic gradient descent, lr=1e-1, learning rate

    #-----------------
    # initial predictions
    #-----------------
    torch.manual_seed(42) # Set the model random seed for reproducible results

    # If we are re-running this frame, the model will already be pre-fitted, so re-init its parameters
    #NEEDED?
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    initial_preds = model(train_dataset.x.float())
    initial_loss = loss_function(initial_preds, train_dataset.y.unsqueeze(1).float())
    initial_acc = torch.eq(initial_preds.squeeze(1).round().bool(), train_dataset.y.bool()).sum()

    print(f"Initial Loss: {initial_loss:.3f}")
    print(f"Initial Accuracy: {initial_acc/len(x_train_std):.3f} \n")

    #----------------
    # Monitor loss and accuracy on validation sample (& training sample)
    #----------------
    epochs = 500

    losses_valid = np.zeros((epochs))
    accuracies_valid = np.zeros((epochs))

    lowest_loss_val = float('inf')
    epochs_w_o_improvement = 0
    break_point = epochs

    losses_train = np.zeros((epochs))
    accuracies_train = np.zeros((epochs))

    #-----------------
    # Training & evaluation
    #-----------------
    for epoch in range(epochs):
        # --- training ---
        model.train() #training mode 
        running_loss_train = 0
        num_correct_train = 0
        total_samples_train = 0
        for i, (inputs, targets) in enumerate(train_data):
            optimizer.zero_grad()
            predictions = model(inputs.float())
            pred_class = (predictions.squeeze(1) > 0)
            loss = loss_function(predictions, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

            running_loss_train += loss.item()
            num_correct_train += (pred_class == targets.bool()).sum().item()
            #num_correct_train += torch.eq(predictions.squeeze(1).round().bool(), targets.bool()).sum()
            total_samples_train += targets.size(0)
        
        losses_train[epoch] = running_loss_train/len(train_data)
        accuracies_train[epoch] = num_correct_train/total_samples_train

        # --- evaluation ---
        model.eval() #evaluation mode 

        running_loss_valid = 0
        num_correct_valid = 0
        total_samples_valid = 0
        with torch.no_grad(): #saves memory & speeds up 
            for i, (inputs, targets) in enumerate(valid_data):
                predictions = model(inputs.float())
                pred_class = (predictions.squeeze(1) > 0)
                loss = loss_function(predictions, targets.unsqueeze(1).float())
                running_loss_valid += loss.item()
                num_correct_valid += (pred_class == targets.bool()).sum().item()
                #num_correct_valid += torch.eq(predictions.squeeze(1).round().bool(), targets.bool()).sum()
                total_samples_valid += targets.size(0)

        loss_val_epoch = running_loss_valid/len(valid_data)
        losses_valid[epoch] = loss_val_epoch
        accuracies_valid[epoch] = num_correct_valid/total_samples_valid

        if epoch%50 == 0:
            print(f"Epoch: {epoch+1:02d}/{epochs} \t| Loss valid: {losses_valid[epoch]:.3f} \t | Accuracy valid: {accuracies_valid[epoch]:.3f}")
            print(f"\t\t| Loss train: {losses_train[epoch]:.3f} \t | Accuracy train: {accuracies_train[epoch]:.3f}")

        if loss_val_epoch < lowest_loss_val:
                lowest_loss_val = loss_val_epoch
                epochs_w_o_improvement = 0
                #best_epoch = copy.deepcopy(model.state_dict())     # to keep the best epoch 
        else: 
                epochs_w_o_improvement += 1    

        if epochs_w_o_improvement >= 10:    
                print(f"Early stopping at epoch {epoch}\t | Final loss valid: {losses_valid[epoch]:.3f} \t | Final accuracy valid: {accuracies_valid[epoch]:.3f}")
                print(f"\t                         | Final loss train: {losses_train[epoch]:.3f} \t | Final accuracy train: {accuracies_train[epoch]:.3f}")
                break_point = epoch 
                break

    #model.load_state_dict(best_epoch)         # we want to set the state to the one of the best epoch 

    #----------------
    # Plot loss and accuracy of validation sample 
    #----------------
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(9, 4))
    ax1.plot(range(break_point), losses_valid[:break_point], label="Loss validation dataset")
    ax1.plot(range(break_point), losses_train[:break_point], label="Loss training dataset")
    ax2.plot(range(break_point), accuracies_valid[:break_point], label="Accuracy validation dataset")
    ax2.plot(range(break_point), accuracies_train[:break_point], label="Accuracy training dataset")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("BCEWithLogitsLoss") # BCEWithLogitsLoss, MSE Loss
    ax2.set_ylabel("Accuracy")
    ax1.legend()
    ax2.legend()
    fig.tight_layout()
    fname = f"plots/loss_acc_{epochs}epochs.png"
    fig.savefig(fname)
    print("saved", fname)

    return {
        "model": model,
        "x_valid_std": x_valid_std,
        "y_valid": y_valid,
        "epoch_no": break_point,
        "MC_events": MC_events,
        "train_ids": train_ids,
        "scaler": scaler 
    }