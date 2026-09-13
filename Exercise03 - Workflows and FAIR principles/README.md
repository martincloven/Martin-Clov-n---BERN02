# Neural-Network Tagger for Hadronic V→qq̄ Jets

This is the repository from Lena Emily Linsner, Martin Cloven & Marie-Philine Hebel.

A tiny neural network is implemented to tag boosted **W/Z→qq̄** jets and compared to a cut-based baseline. The model is then applied to real ATLAS Open Data.  
Learn more about the data source here: https://opendata.atlas.cern/

---

## What’s in this repo

- `assignment.pdf` — the project brief and deliverables.
- `data_exercise_template.py` — starter code:
  - `Particle`/`Event` classes
  - CSV loading
  - Cut-based selection + purity scaffold
- `train_MLP.py` — Setup of the Neural Network as suggested in the assignment.
  - Neural Network model (1)
  - training  
- `train_MLP_2.py` — Alternative Neural Network structure .
  - Neural Network model (2)
  - training 
- `test_performance.py` — Testing Neural Network on validation sample.
  - Histograms of scores for background and signal 
  - ROC curve (& AUC)
  - purity
  - True positive rate (TPR/epsilon_S, $\epsilon_S$)
  - threshold 
- `apply_NN.py` — Neural Network applied to real ATLAS open data.
  - mass histogram (Baseline, cuts and Neural network).
- `main.py` — Running the project.
- `Project_documentation.md` — Report with background information and discription of results.
- `/plots` — all generated plots are saved here
- `jets.csv` — **ATLAS Open Data** (flattened; leading lepton + large-R jets).
- `pythia.csv` — **MC** in the same format plus `v_true` (truth label for large-R jets).
- `requirements.txt` — minimal Python deps (numpy, matplotlib, torch, etc.).

---
## Set up the python environment
In the terminal run:
```bash
python -m venv .venv 
source .venv/bin/activate 
pip install -r requirements.txt
```
---

## Running the project
In the terminal run:
```bash
python main.py [--NeuralNetwork {1,2,both}]
```
If none of the options are added (`python main.py`), neural network 1 will be chosen automatically. 

To display the possible options run `python main.py -h`.

--- 
