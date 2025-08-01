#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 21 11:33:16 2025

@author: kxtz813
"""

import numpy as np
import tifffile as tiff
from skimage.filters import difference_of_gaussians
from skimage.feature import peak_local_max
from numba import jit
from numba.typed import List

def load_stack(file_path: str) -> 'np.memarray':
    
    with tiff.TiffFile(file_path) as tif:
        
        memmap_stack = tif.asarray(out='memmap')
    
    return memmap_stack.astype(np.float32)

def convert_to_photons(image: 'np.ndarray', adc: float,
                       gain: int, qe: float) -> 'np.ndarray':

    """
    Converts an image from grayscale to photons
    """

    image_in_photons = image.copy()

    image_in_photons = image_in_photons - np.min(image_in_photons)

    return image_in_photons * adc * (1 / qe) * (1 / gain)

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
    threshold_abs=3 * threshold,      # Minimum intensity to be considered a spot
    num_peaks=np.inf,      # Maximum number of peaks to return
    )

    return coordinates.astype(np.int32)


@jit(nopython=True, nogil=True, cache=False)
def get_spot_edges(x: int, y: int, width: int):

    # Width in pixels

    x_min, y_min = x - int(0.5 * width), y - int(0.5 * width),

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

@jit(nopython=True, nogil=True, cache=False)
def extract_spot_roi(image, image_stack, spot_center):

    y, x = spot_center[0], spot_center[1]

    edge_coords = get_spot_edges(x, y, width=7)

    if np.any(edge_coords > image.shape[0] - 1) is np.True_:

        pass

    elif np.any(edge_coords < 0) is np.True_:

        pass

    else:
        
        spot = extract_spot_3d(image_stack, edge_coords)
    
    return spot

def extract_on_off(spot: 'np.ndarray') -> tuple['np.ndarray', 'np.ndarray']:

    threshold = rms_calc(spot)

    mean_int = np.mean(np.mean(spot, axis=2), axis=1)

    on_frames = np.where(mean_int > (3 * threshold))

    off_frames =np.where(mean_int < (3 * threshold))

    return on_frames, off_frame


def main():
    
    file_name = ''
    
    images = load_stack(file_name)
    
    photophysical_params = list()

    spot_params = np.zeros((1, 5), dtype=np.float32)
    
    for i, image in enumerate(images):
        
        threshold = rms_calc(image)
        
        filt_im = dif_of_gaussians_filt(image)
        
        local_maxima = extract_local_maxima(filt_im, threshold)

        frame_phphyiscal_params = list()
        
        for j, maximum in enumerate(local_maxima):
            
            spot = extract_spot_roi(image, images, maximum)

            spot_no_bg = spot - np.min(spot)
            
            ### Insert code for calculations here ###
        one_frame_params = np.vstack(frame_phphyiscal_params)

        frame_no = np.full(one_frame_params.shape[0], i + 1).reshape(-1, 1)

        frame_params = np.hstack((frame_no, one_frame_params))

        photophysical_params.append(frame_params)

        del frame_params

        