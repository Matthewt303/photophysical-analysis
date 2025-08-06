import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
from numba import jit, prange
from analyse import rms_calc
import pandas as pd

def load_stack(file_path: str) -> 'np.memarray':
    
    with tiff.TiffFile(file_path) as tif:
        
        memmap_stack = tif.asarray(out='memmap')
    
    return memmap_stack.astype(np.float32)

@jit(nopython=True, nogil=True, cache=False)
def mean_im(im: 'np.ndarray') -> float:

    return np.mean(im)

#@jit(nopython=True, nogil=True, cache=False)
def sum_stack(stack: 'np.ndarray') -> 'np.ndarray':

    summed_intensities = np.zeros((stack.shape[0]))

    for i in range(stack.shape[0]):

        im = stack[i, :, :]

        summed_intensities[i] = mean_im(im)
    
    df = pd.DataFrame(summed_intensities)
    df.to_csv('C:/Users/mxq76232/Downloads/test.csv')
    
    return summed_intensities

def auto_cor(data: 'np.ndarray'):

    data_mean = np.mean(data)
    result = np.correlate(data - data_mean, data - data_mean, mode='full')
    print(result.size)
    result = result[(result.size + 1) // 2:]  # take only non-negative lags
    result /= np.max(result)  # normalize by lag 0
    return result

def main():

    file_name = 'C:/Users/mxq76232/Downloads/test_p/2.tif'

    stack = load_stack(file_name)

    data = sum_stack(stack)

    print(data[0:15])

    autocor = auto_cor(data)

    print(rms_calc(stack))
    print(np.mean(stack[29, :, :]))

    plt.plot(autocor)
    plt.show()

if __name__ == "__main__":

    main()