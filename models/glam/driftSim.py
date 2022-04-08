# By PSD from BDM_lab based on GLAM components. (Nov 2018)
import numpy as np
import pandas as pd
from scipy.stats import invgauss


def simulate_trial_drift(parameters, values, gaze, boundary=1, error_weight=0.05, error_range=(0, 5000)):
    v, gamma, s, tau, t0 = parameters
    n_items = len(values)
    # we consider ´s´ paramter as the standard deviation. 
    #if np.random.uniform(0, 1) < error_weight:
    #    rt = int(np.random.uniform(*error_range))
    #    choice = np.random.choice(n_items)

    drifts = expdrift(v, tau, gamma, values, gaze)
    # initialize evidence accumlator for each item 
    # only 2 items case
    Evi0 = np.zeros(1) + t0 
    Evi1 = np.zeros(1) + t0
    time = 0
    while (Evi0[time] < boundary) and (Evi1[time] < boundary):
        
            #mu = boundary / drifts[i]
            #lam = (boundary / s)**2
            Evi0_new = Evi0[time] +  drifts[0] +  np.random.normal(0,s)
            Evi1_new = Evi1[time] +  drifts[1] +  np.random.normal(0,s)
            
            Evi0 = np.append(Evi0,Evi0_new)
            Evi1 = np.append(Evi1,Evi1_new)
            
            time += 1

    choice = np.argmax([Evi0[-1],Evi1[-1]])
    rt = int(np.round(time + t0))
    
    return Evi0, Evi1, choice, rt


def expdrift(v, tau, gamma, values, gaze):
    n_items = len(values)

    absolute = gaze * values + (1. - gaze) * gamma * values
    relative = np.zeros(n_items)

    for i in range(n_items):
        others = np.arange(n_items)[np.arange(n_items) != i].astype(int)
        relative[i] = absolute[i] - np.max(absolute[others])

    scaled = v * 10 / (1 + np.exp(-tau*relative))

    return scaled
