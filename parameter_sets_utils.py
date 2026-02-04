import numpy as np
import pickle
import pandas as pd

def load_base(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
        return data['params'], data['models'], data['multiband_list']
    
def vary_base_light(base_params, offsets_df):
    '''
    Generator that yields parameter sets by varying the base parameters according to the next row of the provided offsets dataframe. 
    '''

    for _, row in offsets_df.iterrows():
        new_params = base_params.copy()
        for col in offsets_df.columns:
            if col in new_params:
                new_params[col] += row[col]
        yield new_params