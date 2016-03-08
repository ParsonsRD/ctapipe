"""
Integrate sample-mode data (traces) Functions
and
Convert the integral pixel ADC count to photo-electrons
"""

import sys
import numpy as np
import scipy.ndimage as ndimage

from pyhessio import *
from ctapipe import io
from astropy import units as u
from time import time

__all__ = [
    'pixel_integration_mc',
    'calibrate_amplitude_mc',
    'calibrate_tpeak'
]

CALIB_SCALE = 0.92

def pixel_integration_mc(event, ped, telid, integration_type = "global",window = [7,2]):

    """Integrate the raw adc traces by using one of several algorithms (see functions for
    algorithm description)

    Parameters
    ----------
    geom: `ctapipe.io.CameraGeometry`
        Camera geometry information
    ped: list
        list of pedestals in all gain channels in all pixels
    pedvars: int
        Telescope ID number
    integration_type: string
        specify signal integration type
    window: list
        give dimensions of window to be read out in ADC bins (length, offset from peak)

    Returns
    -------

    numpy ndarray of integrated signals (in ADC counts) with the dimensions
    [Gain Channel][pixel number]
    To Convert this to p.e. this should be passed on to calibrate_pixel_amplitude

    """

    # Firstly copy everything to numpy array to make integrations easier/faster
    pixel_adc = pixels_to_array(telid, ped )
    if integration_type == "local":
        adc_sum = local_peak_integration(pixel_adc,window)
    elif integration_type == "global":
        adc_sum = global_peak_integration(pixel_adc,window)
    elif integration_type == "neighbour":
        adc_sum = neighbour_peak_integration(event,telid,pixel_adc,window)
    elif integration_type == "full":
        adc_sum = full_integration(pixel_adc)

    return adc_sum

def pixels_to_array(telid,ped):
    """
    Simple function to retrieve adc samples from a given event and copy them to a numpy array
    of dimensions [gain channel][pixel number][ADC bin]
    Pedestal subtracted binwise here, to make things easier later
    """
    camInfo = np.zeros((get_num_channel(telid),get_num_pixels(telid),get_num_samples(telid)))

    for igain in range(0, get_num_channel(telid)):
        samples_pix = get_adc_sample(telid, igain)
        camInfo[igain] = np.array(samples_pix) - (np.array(ped[igain])[:, np.newaxis]/np.float(samples_pix.shape[1]))

    return camInfo

def full_integration(pixel_adc):
    """
    Integrate all ADC samples, simply return the sum along the ADC channel axis

    Parameters
    ----------
    pixel_adc: numpy array
        pixel ADC traces with dimensions [gain channel][pixel number][ADC bin]

    Returns
    -------
    numpy ndarray of integrated signals (in ADC counts) with the dimensions
    [Gain Channel][pixel number]

    """
    return np.sum(pixel_adc,axis=2)

def local_peak_integration(pixel_adc, window):
    """
    Integrate ADC trace around the peak in the signal (specified by window)
    Currently this uses the scipy correlation function to perform this integration
    using FFT, as this should be extremely fast
    Likely quite unstable due to fluctuations in peak position and biased for small amplitudes

    Need to check window edges are dealt with correctly

    Parameters
    ----------
    pixel_adc: numpy array
        pixel ADC traces with dimensions [gain channel][pixel number][ADC bin]
    window: list
        give dimensions of window to be read out in ADC bins (length, offset from peak)

    Returns
    -------
    numpy ndarray of integrated signals (in ADC counts) with the dimensions
    [Gain Channel][pixel number]

    """
    nbins,offset = window

    # Correlate signal along ADC axis using ndimage (is constant correct here?)
    pixel_adc_corr = ndimage.correlate1d(pixel_adc,np.ones(nbins),origin=-1*offset,axis=2,mode="constant")
    return np.amax(pixel_adc_corr,axis=2) #Return peak of correlated signal

def global_peak_integration(pixel_adc, window):
    """
    Integrate signal based on the global peak of all ADC counts, should be more stabel than
    local integration, but peak likely non optimal for all pixels in large events

    Parameters
    ----------
    pixel_adc: numpy array
        pixel ADC traces with dimensions [gain channel][pixel number][ADC bin]
    window: list
        give dimensions of window to be read out in ADC bins (length, offset from peak)

    Returns
    -------
    numpy ndarray of integrated signals (in ADC counts) with the dimensions
    [Gain Channel][pixel number]

    """

    nbins,offset = window
    global_pix_adc = np.sum(pixel_adc,axis=1) # sum traces of all pixels
    peak = np.argmax(global_pix_adc,axis=1) #find peak bin

    mask = np.zeros(pixel_adc.shape[2])
    mask[peak-offset:peak+(nbins-offset)] = 1 #create mask of readout window

    pixel_adc *= mask #multiply ADC by mask to set all values outside window to 0

    return np.sum(pixel_adc,axis=2) #return sum of traces

def neighbour_peak_integration(event,telid,pixel_adc, window, add_central=False):
    """
    Integrate signal based on the local peak of all ADC counts of neighbouring pixels.
    Loop currently used which may slow things down (not sure how to get round this)

    Parameters
    ----------
    event: pyhessio event class
    telid: int
        telescope number
    pixel_adc: numpy array
        pixel ADC traces with dimensions [gain channel][pixel number][ADC bin]
    window: list
        give dimensions of window to be read out in ADC bins (length, offset from peak)
    add_central: bool
        include the pixel being integrated in the peak determination
    Returns
    -------
    numpy ndarray of integrated signals (in ADC counts) with the dimensions
    [Gain Channel][pixel number]
    """

    nbins,offset = window

    #correlate image first using ndimage
    pixel_adc_corr = ndimage.correlate1d(pixel_adc,np.ones(nbins),origin=-1*offset,axis=2,mode="constant")

    pix_x, pix_y = event.meta.pixel_pos[telid] # first get camera geometry (this could be passed in a nicer way)
    geom = io.CameraGeometry.guess(pix_x, pix_y) # most of time taken here, should think of better may to pass geometry

    signal = np.zeros(pixel_adc.shape[:-1])
    for gain in range(pixel_adc.shape[0]): # loop over gains and pixels (should be eliminated if possible)
        for pixel in range(pixel_adc.shape[1]):
            neighbours = geom.neighbors[pixel] # get list of neighbours
            if(add_central):
                neighbours.append(pixel)

            peakbin = np.argmax(np.sum(pixel_adc[gain][neighbours],axis=0)) # sum traces of neighbours and find peak bin
            signal[gain][pixel] = pixel_adc_corr[gain][pixel][peakbin] #fill signal with correlated values at peak

    return signal #return integrated charges

def calibrate_amplitude_mc(integrated_charge, calib, telid,linear_range=[-1000,10000]):
    """
    Convert ADC counts integrated by previous functions to photoelectrons. This is done
    by multiplying by ADC to p.e. ratio, then choosing the correct gain channel based
    on the linear range of ADC counts

    Parameters
    ----------
    integrated_charge: numpy array
        Array of integrated pixel charges in ADC counts with dimensions [gain channel][pixel number]
    calib: numpy array
        ADC to p.e. ratio fow all pixels iin telescopes [gain channel][pixel number]
    telid: int
        telescope number
    linear_range: list
        range on linearity of gain channels in ADC counts

    Returns
    -------
    numpy ndarray of integrated signals (in photoelectrons) with the dimensions
    [pixel number]
    """
    linear_min,linear_max = linear_range
    if integrated_charge is None:
        return None

    amplitude = np.zeros(integrated_charge.shape[1:])
    if get_num_channel(telid) == 1: # If only 1 gain, just multiply everything
        amplitude = integrated_charge[0] * calib[0] * CALIB_SCALE
        return amplitude # return 0th channel (as there is only 1)
    else:
        amplitude_final = np.zeros(integrated_charge.shape[1:])
        valid_last = np.zeros(integrated_charge.shape[1:])
        for i in range(amplitude.shape[0]):# loop over channels
            # first check that all elements in this array lie in the linear range
            valid = np.logical_and(np.greater(integrated_charge[i],linear_min),
                                np.less(integrated_charge[i],linear_max))
            #Then check that the amplitude of these pixels has not already been calculated
            valid = np.logical_and(valid,np.logical_not(valid_last))
            #Fill final amplitude for only pixels in the linear range
            amplitude += integrated_charge[i] * calib[i] * CALIB_SCALE * valid
            #Add pixels used in this channel to the total
            valid_last += valid

    return amplitude

def get_max_bin(integrated_charge):
    return np.argmax(integrated_charge,axis=2)

def calibrate_tpeak(integrated_charge, offsets, ns_per_bin, peak_method="bin_centre"):
    """
    Get peak time from pedestal subtracted traces, and subtract any offset in the
    peak position for each pixel

    Parameters
    ----------
    integrated_charge: numpy array
        Array of integrated pixel charges in ADC counts with dimensions [gain channel][pixel number]
    offsets: numpy array
        Offset of timing for this pixel readout
    ns_per_bin: float
        nanoseconds per ADC bin
    peak_method: string
        Method used for peak determination

    Returns
    -------
    numpy ndarray of peak times of ADC trace
    [gain channel][pixel number]
    """

    times = np.zeros(integrated_charge[:-1])
    if peak_method is "bin_centre":
        times = get_max_bin(integrated_charge)
    elif peak_method is "parabola_fit":
        times = get_max_bin(integrated_charge)
    elif peak_method is "cwt":
        times = get_max_bin(integrated_charge)

    return (times + offsets) * ns_per_bin
