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

__all__ = [
    'pixel_integration_mc',
    'calibrate_amplitude_mc',
]

CALIB_SCALE = 0.92

def pixel_integration_mc(ped, telid,):

    pixel_adc = pixels_to_array(telid, ped )
    adc_sum = local_peak_integration(pixel_adc,7,2)
    #adc_sum = global_peak_integration(pixel_adc,7,2)
    #adc_sum = full_integration_dan(pixel_adc)

    return adc_sum,None

def pixels_to_array(telid,ped):

    camInfo = np.zeros((get_num_channel(telid),get_num_pixels(telid),get_num_samples(telid)))

    for igain in range(0, get_num_channel(telid)):
        samples_pix = get_adc_sample(telid, igain)
        camInfo[igain] = np.array(samples_pix) - (np.array(ped[igain])[:, np.newaxis]/np.float(samples_pix.shape[1]))

    return camInfo

def full_integration(pixel_adc):
    return np.sum(pixel_adc,axis=2), None

def local_peak_integration(pixel_adc, nbins, offset):
    pixel_adc_corr = ndimage.correlate1d(pixel_adc,np.ones(nbins),origin=-1*offset,axis=2,mode="constant")
    return np.amax(pixel_adc_corr,axis=2)

def global_peak_integration(pixel_adc, nbins, offset):

    global_pix_adc = np.sum(pixel_adc,axis=1)
    peak = np.argmax(global_pix_adc,axis=1)
    mask = np.zeros(pixel_adc.shape[2])

    mask[peak-offset:peak+(nbins-offset)] = 1
    pixel_adc *= mask
    print (np.sum(pixel_adc,axis=2),np.sum(mask),peak)

    return np.sum(pixel_adc,axis=2)

def neighbour_peak_integration_dan(event,pixel_adc, nbins, offset):
    pix_x, pix_y = event.meta.pixel_pos[telid]
    geom = io.CameraGeometry.guess(pix_x, pix_y)

    return

def calibrate_amplitude_mc(integrated_charge, calib, telid, params):
    TAG = sys._getframe().f_code.co_name+">"
    """
    Parameters
    ----------
    integrated_charge     Array of pixels with integrated change [ADC cts],
                          pedestal substracted
    calib                 Array of double containing the single-pe events
    parameters['clip_amp']  Amplitude in p.e. above which the signal is
                            clipped.
    Returns
    ------
    Array of pixels with calibrate charge [photo-electrons]
    Returns None if event is None

    """

    if integrated_charge is None:
        return None
    amplitude = np.zeros(integrated_charge.shape)
    amplitude = integrated_charge * calib * CALIB_SCALE

    if get_num_channel(telid) == 1:
        print(integrated_charge,integrated_charge * calib)
        return amplitude[0]
    else:
        amplitude_final = np.zeros(integrated_charge.shape)

        for i in enumerate(integrated_charge.shape[0]):
            np.greater(integrated_charge[i],-1000)
            np.less(integrated_charge[i],10000)
            valid = (np.greater(integrated_charge[i],-1000)==np.less(integrated_charge[i],10000))
    #pe_pix_tel = []

    #for ipix in range(0, get_num_pixels(telid)):
    #    pe_pix = 0
    #    int_pix_hg = integrated_charge[get_num_channel(telid)-1][ipix]
    #    # If the integral charge is between -300,2000 ADC ts, we choose the HG
    #    # Otherwise the LG channel
    #    # If there is only one gain, it is the HG (default)
    #    print(int_pix_hg.shape)
    #    if (int_pix_hg > -1000 and int_pix_hg < 10000 or get_num_channel(telid) < 2):
    #        pe_pix = (integrated_charge[get_num_channel(telid)-1][ipix] *
    #        calib[get_num_channel(telid)-1][ipix])
    #    else:
    #        pe_pix = (integrated_charge[get_num_channel(telid)][ipix] *
    #        calib[get_num_channel(telid)][ipix])

    #    # pe_pix is in units of 'mean photo-electrons'
    #    # (unit = mean p.e. signal.).
    #    # We convert to experimentalist's 'peak photo-electrons'
    #    # now (unit = most probable p.e. signal after experimental resolution).
    #    # Keep in mind: peak(10 p.e.) != 10*peak(1 p.e.)
    #    pe_pix_tel.append(pe_pix*CALIB_SCALE)
    print(amplitude,amplitude.shape)
    return amplitude[0]
