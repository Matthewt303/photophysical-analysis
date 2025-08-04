import numpy as np
import tifffile as tiff
import pandas as pd
import os

def load_stack(file_path: str) -> 'np.memarray':
    
    with tiff.TiffFile(file_path) as tif:
        
        memmap_stack = tif.asarray(out='memmap')
    
    return memmap_stack.astype(np.float32)

def conv_to_df(all_phsw_params: 'np.ndarray') -> 'pd.DataFrame':

    cols = ['Frame',
            'Duty cycle',
            'Photoswitching time (s)',
            'Number of cycles',
            'Photons per cycle',
            'Total photons']

    data = pd.DataFrame(all_phsw_params, columns=cols)

    return data

def save_data(all_phsw_params: 'pd.DataFrame', out: str) -> None:

    all_phsw_params.to_csv(os.path.join(out, 'pswitch_params.csv'), index=False)