#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:33:16 2025

@author: kxtz813
"""

import numpy as np
from skimage.filters import difference_of_gaussians
from skimage.feature import peak_local_max
from numba import jit, prange
from numba.typed import List
import matplotlib.pyplot as plt
from more_itertools import consecutive_groups
from file_io import load_stack, conv_to_df, save_data

@jit(nopython=True, nogil=True, cache=False)
def convert_to_photons(data: 'np.ndarray', adc: float,
                       gain: int) -> 'np.ndarray':

    """
    Converts data (image, summed images etc.) from grayscale to photons
    """
    return data * adc * (1 / gain)

@jit(nopython=True, nogil=True, cache=False)
def rms_calc(data: 'np.ndarray') -> float:

    """
    Calculates RMS of an image
    """

    return np.sqrt(np.mean(data.flatten()**2))

def dif_of_gaussians_filt(image: 'np.ndarray') -> 'np.ndarray':
    
    im = image.copy()

    filt_im = difference_of_gaussians(im, 1, 6)

    return filt_im

def extract_local_maxima(image, threshold):

    coordinates = peak_local_max(
    image,
    min_distance=2,          # Minimum number of pixels between peaks
    threshold_abs=6 * threshold,      # Minimum intensity to be considered a spot
    num_peaks=np.inf,      # Maximum number of peaks to return
    )

    return coordinates.astype(np.int32)

#@jit(nopython=True, nogil=True, cache=False)
def get_spot_edges(x: int, y: int, width: int):

    # Width in pixels

    x_min, y_min = x - int(0.5 * width), y - int(0.5 * width)

    x_max, y_max = x + int(0.5 * width), y + int(0.5 * width)

    return np.array([x_min, x_max, y_min, y_max]).reshape(1, 4)

@jit(nopython=True, nogil=True, cache=False)
def extract_spot_3d(image_stack, edges):

    horizontal_filt = image_stack[:, :, edges[0, 0]:edges[0, 1]]

    vertical_filt = horizontal_filt[:, edges[0, 2]:edges[0, 3], :]

    return vertical_filt

@jit(nopython=True, nogil=True, cache=False)
def extract_spot_rois(image, image_stack, spot_centers):

    """
    Not in use
    """

    spots = List()

    for i in range(0, spot_centers.shape[0]):

        y, x = spot_centers[i, 0], spot_centers[i, 1]

        edge_coords = get_spot_edges(x, y, width=10)

        if np.any(edge_coords > image.shape[0] - 1) is np.True_:

            pass

        elif np.any(edge_coords < 0) is np.True_:

            pass

        else:
        
            spot = extract_spot_3d(image_stack, edge_coords)
            
            # Probably better to extract spot, calc photophysics then move to next spot #

            spots.append(spot)
            
            del spot
    
    return spots

#@jit(nopython=True, nogil=True, cache=False)
def extract_spot_roi(image, image_stack, spot_center):

    y, x = spot_center[0], spot_center[1]

    edge_coords = get_spot_edges(x, y, width=8)

    if np.any(edge_coords > image.shape[0] - 1) is np.True_:

        spot = np.array([])

    elif np.any(edge_coords < 0) is np.True_:

        spot = np.array([])

    else:
        
        spot = extract_spot_3d(image_stack, edge_coords)
    
    return spot

#@jit(nopython=True, nogil=True, cache=False)
def extract_on_off(spot: 'np.ndarray') -> tuple['np.ndarray', 'np.ndarray']:

    threshold = rms_calc(spot)

    mean_int = np.mean(np.mean(spot, axis=2), axis=1)

    on_frames = np.where(mean_int > (2 * threshold))[0]

    off_frames = np.where(mean_int < (2 * threshold))[0]

    return on_frames, off_frames

@jit(nopython=True, nogil=True, cache=False)
def calc_duty_cycle(on_frames: 'np.ndarray', off_frames: 'np.ndarray') -> float:

    return on_frames.shape[0] / off_frames.shape[0]

@jit(nopython=True, nogil=True, cache=False)
def calc_phsw_time(on_frames: 'np.ndarray', exp_time: float) -> float:

    # Convert from zero indexing to frame num
    phsw_time_frames = (np.max(on_frames) + 1) - (np.min(on_frames) - 1)

    # Time in seconds
    return phsw_time_frames * exp_time

def extract_phsw_cyc(on_frames: 'np.ndarray') -> list[list]:

    ph_sw_frame_seqs = [
        list(seq) for seq in consecutive_groups(np.sort(on_frames))
    ]

    return ph_sw_frame_seqs

#@jit(nopython=True, nogil=True, cache=False, parallel=True)
def ph_per_phsw_cyc(frame_seqs: list[list], spot: 'np.ndarray',
                    adc: float, gain: int=1) -> float:

    photons_per_cycle = np.zeros((len(frame_seqs), 1), dtype=np.int64)

    for i in prange(len(frame_seqs)):

        seq_indices = frame_seqs[i]

        seq = spot[seq_indices]

        photons_per_cycle[i, 0] = convert_to_photons(np.sum(seq), adc=adc, gain=gain)
    
    return np.int64(np.mean(photons_per_cycle))

def calc_num_phsw_cycs(phsw_cyc_frames: list) -> int:

    return len(phsw_cyc_frames)

def total_photons(num_sw_cyc: int, ph_per_phsw_cyc: int) -> int:

    return num_sw_cyc * ph_per_phsw_cyc

def main():

    ## Hyperparameters ##
    
    file_name = 'C:/Users/mxq76232/Downloads/test_p/1.tif'
    out = 'C:/Users/mxq76232/Downloads/test_p'
    exposure_time = 0.03
    adc = 0.59

    images = load_stack(file_name)

    print(images.shape)
    
    photophysical_params = list()
    
    for i, image in enumerate(images):

        print('Analysing frame ' + str(i + 1)) 
        
        filt_im = dif_of_gaussians_filt(image)

        threshold = rms_calc(filt_im)

        print(threshold)
        
        local_maxima = extract_local_maxima(filt_im, threshold)

        frame_params = np.zeros((local_maxima.shape[0], 5))

        for j, maximum in enumerate(local_maxima):
            
            spot = extract_spot_roi(image, images, maximum)

            if not spot.size:

                continue # Skip edge spots

            spot_no_bg = spot - np.min(spot)
            
            # Param order: duty cycle, psw time, num cyc, ph per cyc, tot ph #

            on, off = extract_on_off(spot_no_bg)

            if not on.size:

                continue # Skip empty arrays

            phsw_cycles = extract_phsw_cyc(on)

            frame_params[j, 0] = calc_duty_cycle(on, off)
            frame_params[j, 1] = calc_phsw_time(on, exposure_time)
            frame_params[j, 2] = calc_num_phsw_cycs(phsw_cycles)
            frame_params[j, 3] = ph_per_phsw_cyc(phsw_cycles, spot_no_bg, adc)
            frame_params[j, 4] = total_photons(frame_params[j, 2], frame_params[j, 3])

        frame_no = np.full(frame_params.shape[0], i + 1).reshape(-1, 1)

        frame_params = np.hstack((frame_no, frame_params))

        photophysical_params.append(frame_params)

        del frame_params

        if i == (images.shape[0] * 2) // 3:

            break

    all_params = np.vstack(photophysical_params)
    all_params_df = conv_to_df(all_params)

    save_data(all_params_df, out=out)

if __name__ == "__main__":

    main()