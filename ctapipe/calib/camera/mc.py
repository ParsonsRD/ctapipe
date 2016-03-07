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
    'nb_peak_integration_mc',
    'calibrate_amplitude_mc',
]

CALIB_SCALE = 0.92

"""

The function in this module are the same to their corresponding ones in
- read_hess.c
- reconstruct.c
in hessioxxx software package, just in some caes the suffix "_mc" has
been added here to the original name function.

Note: Input MC version = prod2. For future MC versions the calibration
function might be different for each camera type.
It has not been tested so far.

In general the integration functions corresponds one to one in name and
functionality with those in hessioxxx package.
The same the interpolation of the pulse shape and the adc2pe conversion.

"""


def pixel_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
    """
    Parameters
    ----------
    event  Data set container to the hess_io event ()
    ped    Array of double containing the pedestal
    telid  Telescope_id
    parameters
    integrator: pixel integration algorithm
       -"full_integration": full digitized range integrated amplitude-pedestal
       -"simple_integration": fixed integration region (window)
       -"global_peak_integration": integration region by global
       peak of significant pixels
       -"local_peak_integration": peak in each pixel determined independently
       -"nb_peak_integration":

    Returns
    -------
    Array of pixels with integrated change [ADC cts], pedestal substracted.
    Returns None if event is None
    """
    if __debug__:
        print(TAG, parameters['integrator'], end="\n")
    if event is None:
        return None
    pixel_adc = pixels_to_array(telid, ped )
    local_integration_dan(pixel_adc,7,2)
    global_integration_dan(pixel_adc,7,2)
    adc_sum = full_integration_dan(pixel_adc)

    return adc_sum,None

def pixels_to_array(telid,ped):
    print (telid)
#    if event is None or telid < 0:
#        return None

    camInfo = np.zeros((get_num_channel(telid),get_num_pixels(telid),get_num_samples(telid)))

    for igain in range(0, get_num_channel(telid)):
        samples_pix = get_adc_sample(telid, igain)
        camInfo[igain] = np.array(samples_pix) - (np.array(ped[igain])[:, np.newaxis]/np.float(samples_pix.shape[1]))

    return camInfo

def full_integration_dan(pixel_adc):
    return np.sum(pixel_adc,axis=2), None

def local_integration_dan(pixel_adc, nbins, offset):
    pixel_adc_corr = ndimage.correlate1d(pixel_adc,np.ones(nbins),origin=offset,axis=2,mode="constant")
    return np.amax(pixel_adc_corr,axis=2)

def global_integration_dan(pixel_adc, nbins, offset):

    global_pix_adc = np.sum(pixel_adc,axis=1)
    peak = np.argmax(global_pix_adc,axis=1)
    mask = np.zeros(pixel_adc.shape[2])

    mask[peak-offset:peak+(nbins-offset)] = 1
    pixel_adc *= mask
    print (np.sum(pixel_adc,axis=2))

    return np.sum(pixel_adc,axis=2)

def neighbour_peak_integration_dan(pixel_adc, nbins, offset):
    return


def nb_peak_integration_mc(event, ped, telid, parameters):
    TAG = sys._getframe().f_code.co_name+">"
    """
    Integrate sample-mode data (traces) around a peak in the signal sum of
    neighbouring pixels.

    The integration window can be anywhere in the available length
    of the traces.
    No weighting of individual samples is applied.

    Parameters
    ----------

    event                 Data set container to the hess_io event ()
    ped                   Array of double containing the pedestal
    telid                 Telescope_id
    parameters['nsum']    Number of samples to sum up
                          (is reduced if exceeding available length).
    parameters['nskip'] Start the integration a number of samples before
                          the peak, as long as it fits into the available data
                          range.
                          Note: for multiple gains, this results in identical
                          integration regions.
    parameters['sigamp']  Amplitude in ADC counts above pedestal at which
                          a signal is considered as significant (separate for
                          high gain/low gain).
    parameters['lwt']     Weight of the local pixel (0: peak from neighbours
                          only,1: local pixel counts as much as any neighbour).

    Returns
    -------
    array of pixels with integrated change [ADC cts], pedestal
    substracted per gain and peak slide
    """

    # The number of samples to sum up can not be larger than
    # the number of samples
    nsum = parameters['nsum']
    if nsum >= get_num_samples(telid):
        nsum = get_num_samples(telid)

    #  For this integration scheme we need the list of neighbours early on
    pix_x, pix_y = event.meta.pixel_pos[telid]
    geom = io.CameraGeometry.guess(pix_x, pix_y)

    sum_pix_tel = []
    time_pix_tel = []
    for igain in range(0, get_num_channel(telid)):
        sum_pix = []
        peak_pix = []
        for ipix in range(0, get_num_pixels(telid)):
            i = 0
            knb = 0
            # Loop over the neighbors of ipix
            ipix_nb = geom.neighbors[ipix]
            nb_samples = [0 for ii in range(get_num_samples(telid))]
            for inb in range(len(ipix_nb)):
                nb_samples += np.array(get_adc_sample(telid, igain)
                                       [ipix_nb[inb]])
                knb += 1
            if parameters['lwt'] > 0:
                for isamp in range(1, get_num_samples(telid)):
                    nb_samples += np.array(get_adc_sample(telid, igain)
                                           [ipix])*lwt
                knb += 1

            if knb == 0:
                continue
            ipeak = 0
            p = nb_samples[0]
            for isamp in range(1, get_num_samples(telid)):
                if nb_samples[isamp] > p:
                    p = nb_samples[isamp]
                    ipeak = isamp
            peakpos = peakpos_hg = ipeak
            start = peakpos - parameters['nskip']

            # Sanitity check?
            if start < 0:
                start = 0
            if start + nsum > get_num_samples(telid):
                start = get_num_samples(telid) - nsum

            int_corr = 1#set_integration_correction(telid, parameters)
            # Integrate pixel
            samples_pix_win = get_adc_sample(telid, igain)
            [ipix][start:(nsum+start)]
            ped_per_trace = ped[igain][ipix]/get_num_samples(telid)
            sum_pix.append(np.round(int_corr[igain]*(sum(samples_pix_win) -
                                                  ped_per_trace*nsum)))
            peak_pix.append(peakpos)

        sum_pix_tel.append(sum_pix)
        time_pix_tel.append(peak_pix)

    return sum_pix_tel, time_pix_tel


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
    amplitude = np.zeros(integrated_charge.shape[1])
    if get_num_channel(telid) == 1:
        amplitude = integrated_charge[0] * calib[0] * CALIB_SCALE

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

    #    if "climp_amp" in params and params["clip_amp"] > 0:
    #        if pe_pix > params["clip_amp"]:
    #            pe_pix = params["clip_amp"]

    #    # pe_pix is in units of 'mean photo-electrons'
    #    # (unit = mean p.e. signal.).
    #    # We convert to experimentalist's 'peak photo-electrons'
    #    # now (unit = most probable p.e. signal after experimental resolution).
    #    # Keep in mind: peak(10 p.e.) != 10*peak(1 p.e.)
    #    pe_pix_tel.append(pe_pix*CALIB_SCALE)
    print(amplitude,amplitude.shape)
    return amplitude*CALIB_SCALE
