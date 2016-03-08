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

def pixel_integration_mc(event,ped, telid,):

    pixel_adc = pixels_to_array(telid, ped )
#    adc_sum = local_peak_integration(pixel_adc,7,2)
    adc_sum = neighbour_peak_integration(event,telid,pixel_adc,7,2)

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

def neighbour_peak_integration(event,telid,pixel_adc, nbins, offset):
    pixel_adc_corr = ndimage.correlate1d(pixel_adc,np.ones(nbins),origin=-1*offset,axis=2,mode="constant")

    pix_x, pix_y = event.meta.pixel_pos[telid]
    geom = io.CameraGeometry.guess(pix_x, pix_y)
    signal = np.zeros(pixel_adc.shape[:-1])
    for gain in range(pixel_adc.shape[0]):
        for pixel in range(pixel_adc.shape[1]):
            neighbours = geom.neighbors[pixel]
            peakbin = np.argmax(np.sum(pixel_adc[gain][neighbours],axis=0))
            signal[gain][pixel] = pixel_adc_corr[gain][pixel][peakbin]

    return signal

def calibrate_amplitude_mc(integrated_charge, calib, telid, params):

    if integrated_charge is None:
        return None
    print ("Shape Charge",integrated_charge.shape,calib.shape)

    amplitude = np.zeros(integrated_charge.shape)
    if get_num_channel(telid) == 1:
        amplitude = integrated_charge * calib * CALIB_SCALE
        return amplitude[0]
    else:
        amplitude_final = np.zeros(integrated_charge.shape)
        valid_last = np.zeros(integrated_charge.shape)
        print ("valid",integrated_charge.shape)
        for i in range(amplitude.shape[0]):

            valid = np.logical_and(np.logical_and(np.greater(integrated_charge[i],-1000),
                                    np.less(integrated_charge[i],10000)),not valid_last)
            amplitude = integrated_charge * calib * CALIB_SCALE * valid
            valid_last += valid
    return amplitude[0]
