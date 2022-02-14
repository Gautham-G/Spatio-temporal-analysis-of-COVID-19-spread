import pickle
import os
import pandas as pd
import numpy as np

# Converting pkl to list

def predictions_from_pkl(res_path):

    a = [1, 2, 3, 4]
    path = []
    for i in a:
        path.append(res_path+ 'preds_' + str(i) +'.pkl')
    
    for i in path:
        if not os.path.exists(i):
            print(path)
            return None
    predictions = []

    for i in path:
        with open(i, 'rb') as f:
            data_pickle = pickle.load(f)

            predictions.append(data_pickle)
            print('hi')
    return predictions

pred = predictions_from_pkl('../pred_dir/')
