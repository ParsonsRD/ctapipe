import sys
import argparse
from matplotlib import colors, pyplot as plt
import numpy as np
from ctapipe.io.hessio import hessio_event_source
from pyhessio import *
from ctapipe.core import Container
from ctapipe.io.containers import RawData, CalibratedCameraData
from ctapipe import visualization, io
from astropy import units as u
from ctapipe.calib.camera.gct import *
from ctapipe.reco import cleaning,hillas,hillasintersection
from time import time
import scipy.ndimage as nd
import scipy.signal as sig

fig = plt.figure(figsize=(16, 7))

def display_telescope(event, tel_id,hillas):
    global fig
    ntels = len(event.dl1.tels_with_data)
    fig.clear()

    plt.suptitle("EVENT {} {:.1e} TeV @({:.1f},{:.1f})deg @{:.1f} m".format(
            event.dl1.event_id, get_mc_shower_energy(),
            get_mc_shower_altitude(), get_mc_shower_azimuth(),
            np.sqrt(pow(get_mc_event_xcore(), 2) +
                    pow(get_mc_event_ycore(), 2))))
    print("\t draw cam {}...".format(tel_id))
    x, y = event.meta.pixel_pos[tel_id]
    geom = io.CameraGeometry.guess(x , y)

    npads = 1
    # Only create two pads if there is timing information extracted
    # from the calibration
    #if not event.dl1.tel[tel_id].tom is None:
    #    npads = 2

    ax = plt.subplot(1, npads, npads)
    disp = visualization.CameraDisplay(geom, ax=ax,
                                       title="CT{0}".format(tel_id))

    disp.pixels.set_antialiaseds(False)
    disp.autoupdate = False
    disp.pixels.set_cmap('cool')
    chan = 0
    signals = event.dl1.tel[tel_id].pe_charge
    disp.image = signals
    disp.overlay_moments(hillas)
    disp.add_colorbar()
    disp.show()

    if npads == 2:
        ax = plt.subplot(1, npads, npads)
        disp = visualization.CameraDisplay(geom,
                                           ax=ax,
                                           title="CT{0}".format(tel_id))
        disp.pixels.set_antialiaseds(False)
        disp.autoupdate = False
        disp.pixels.set_cmap('gnuplot')
        chan = 0
        disp.image = event.dl1.tel[tel_id].tom
        disp.add_colorbar()

    if __debug__:
        print("All sum = %.3f\n" % sum(event.dl1.tel[tel_id].pe_charge))


def camera_calibration(filename, parameters, disp_args, level):
    """
    Parameters
    ----------
    filename   MC filename with raw data (in ADC samples)
    parameters Parameters to be passed to the different calibration functions
               (described in each function separately inside mc.py)
    disp_args  Either: per telescope per event or
               all telescopes of the event (currently dissabled)
    level      Output information of the calibration level results
    Returns
    -------
    A display (see function display_telescope)

    """
    TAG = sys._getframe().f_code.co_name+">"

    # Load dl1 container
    container = Container("calibrated_hessio_container")
    container.add_item("dl1", RawData())
    container.meta.add_item('pixel_pos', dict())

    # loop over all events, all telescopes and all channels and call
    # the calc_peds function defined above to do some work:
    nt = 0

    for event in hessio_event_source(filename):
        hillas_parameters_list = list()
        hillas_parameters_list1 = list()

        nt = nt+1
        # Fill DL1 container headers information. Clear also telescope info.
        container.dl1.run_id = event.dl0.run_id
        container.dl1.event_id = event.dl0.event_id
        container.dl1.tel = dict()  # clear the previous telescopes
        container.dl1.tels_with_data = event.dl0.tels_with_data
        if __debug__:
            print(TAG, container.dl1.run_id, "#%d" % nt,
                  container.dl1.event_id,
                  container.dl1.tels_with_data,
                  "%.3e TeV @ (%.0f,%.0f)deg @ %.3f m" %
                  (get_mc_shower_energy(), get_mc_shower_altitude(),
                   get_mc_shower_azimuth(),
                   np.sqrt(pow(get_mc_event_xcore(), 2) +
                           pow(get_mc_event_ycore(), 2))))
        for telid in event.dl0.tels_with_data:
            print(TAG, "Calibrating.. CT%d\n" % telid)

            # Get per telescope the camera geometry
            x, y = event.meta.pixel_pos[telid]
            geom = io.CameraGeometry.guess(x , y)
            # Get the calibration data sets (pedestals and single-pe)
            ped = get_pedestal(telid)
            calib = get_calibration(telid)

            # Integrate pixels traces and substract pedestal
            # See pixel_integration_mc function documentation in mc.py
            # for the different algorithms options
            start = time()


            pix_adc = pyhessio_trace_array(telid,ped)

            t = np.arange(0., 25., 1)
            np.fft.fftfreq(25,d=1)
            dt_fft = np.fft.fft(pix_adc[0][0])
            print(dt_fft.shape)
            for i in range(25):
                if dt_fft[i] > 0.001: # cut off all frequencies higher than 0.005
                    dt_fft[i] = 0.0
                    #dt_fft[25/2 + i] = 0.0
            print(np.real(np.fft.ifft(dt_fft)), pix_adc[0][0])

            #plt.plot(t, pix_adc[0][0], 'r--',t,nd.gaussian_filter1d(pix_adc[0][0],1,axis=0),'b--',t,nd.median_filter(pix_adc[0][0],size=3),'g--',t, np.real(np.fft.ifft(dt_fft)),'y--')
           # plt.show()

            int_adc_pix,t_pix = pixel_integration(pix_adc,ped,integration_type="neighbour",geometry=geom)
            pe_pix,t_pix_g = calibrate_amplitude(int_adc_pix,t_pix, np.array(calib))

            int_adc_pix_local,t_pix = pixel_integration(pix_adc,ped,integration_type="local")
            pe_pix_local,t_pix_g = calibrate_amplitude(int_adc_pix_local,t_pix, np.array(calib))

            int_adc_pix_global,t_pix = pixel_integration(pix_adc,ped,integration_type="global")
            pe_pix_global,t_pix_g = calibrate_amplitude(int_adc_pix_global,t_pix, np.array(calib))

            int_adc_pix_full,t_pix = pixel_integration(pix_adc,ped,integration_type="full")
            pe_pix_full,t_pix_g = calibrate_amplitude(int_adc_pix_full,t_pix, np.array(calib))


            #start = time()
            #int_adc_pix = pixel_integration_mc(event,ped, telid,integration_type="local")
            #print ("Execution time",time()-start)

            # Convert integrated ADC counts into p.e.
            # selecting also the HG/LG channel (currently hard-coded)
            #pe_pix = calibrate_amplitude(int_adc_pix, np.array(calib),telid)
            # Including per telescope metadata in the DL1 container
            if telid not in container.meta.pixel_pos:
                container.meta.pixel_pos[telid] = event.meta.pixel_pos[telid]
            container.dl1.tels_with_data = event.dl0.tels_with_data
            container.dl1.tel[telid] = CalibratedCameraData(telid)
            container.dl1.tel[telid].pe_charge = np.array(pe_pix)
            container.dl1.tel[telid].tom = np.array(pe_pix)#np.array(peak_adc_pix[0])

            # FOR THE CTA USERS:
            # From here you can include your code.
            # It should take as input the last data level calculated here (DL1)
            # or call reconstruction algorithms (reco module) to be called.
            # For example: you could ask to calculate the tail cuts cleaning
            # using the tailcuts_clean in reco/cleaning.py module
            #
            # if 'tail_cuts' in parameters:
            clean_mask = cleaning.tailcuts_clean(geom,image=pe_pix,pedvars=1,picture_thresh=10,boundary_thresh=15)
            clean_mask_global = cleaning.tailcuts_clean(geom,image=pe_pix_global,pedvars=1,picture_thresh=5,boundary_thresh=10)
            clean_mask_local = cleaning.tailcuts_clean(geom,image=pe_pix_local,pedvars=1,picture_thresh=5,boundary_thresh=10)
            clean_mask_full = cleaning.tailcuts_clean(geom,image=pe_pix_full,pedvars=1,picture_thresh=5,boundary_thresh=10)

            pix_x, pix_y = event.meta.pixel_pos[telid] # first get camera geometry (this could be passed in a nicer way)

            hp = hillas.hillas_parameters(pix_x, pix_y,pe_pix*clean_mask)
            hillas_parameters_list.append(hp)

            if(hp.size>20):
                hillas_parameters_list1.append(hp)

            hp_local = hillas.hillas_parameters(pix_x, pix_y,pe_pix_local*clean_mask_local)
            hp_global = hillas.hillas_parameters(pix_x, pix_y,pe_pix_global*clean_mask_global)
            hp_full = hillas.hillas_parameters(pix_x, pix_y,pe_pix_full*clean_mask_full)
            pe_pix*=clean_mask
            #container.dl1.tel[telid].pe_charge = pe_pix

            print("Size, nb: ",hp.size,"local:",hp_local.size,"global:",hp_global.size,"full:",hp_full.size)
            #    container.dl1.tel[telid].pe_charge = np.array(pe_pix) *
            #    np.array(clean_mask)
            #    container.dl1.tel[telid].tom = np.array(peak_adc_pix[0]) *
            #    np.array(clean_mask)
            #
        start = time()

        hillasintersection.intersect_nominal(hillas_parameters_list1)
        print ("reco time",time()-start)
        
        sys.stdout.flush()
        # Display
        if 'event' in disp_args:
            ello = input("See evt. %d?<[n]/y/q> " % container.dl1.event_id)
            if ello == 'y':

                if 'telescope' in disp_args:
                    num=0
                    for telid in container.dl1.tels_with_data:
                        if hillas_parameters_list[num].size>20:

                            ello = input(
                                "See telescope/evt. %d?[CT%d]<[n]/y/q/e> " %
                                (container.dl1.event_id, telid))
                            if ello == 'y':
                                display_telescope(container, telid,hillas_parameters_list[num])
                                plt.pause(0.1)
                            elif ello == 'q':
                                break
                            elif ello == 'e':
                                return None
                            else:
                                continue
                        num+=1
                else:
                    plt.pause(0.1)
            elif ello == 'q':
                return None
        hillasintersection.intersect_nominal(hillas_parameters_list)

if __name__ == '__main__':
    TAG = sys._getframe().f_code.co_name+">"

    # Declare and parse command line option
    parser = argparse.ArgumentParser(
        description='Tel_id, pixel id and number of event to compute.')
    parser.add_argument('--f', dest='filename',
                        required=True, help='filename MC file name')
    args = parser.parse_args()

    plt.show(block=False)

    # Function description of camera_calibration options, given here
    # Integrator: samples integration algorithm (equivalent to hessioxxx
    # option --integration-sheme)
    #   -options: full_integration,
    #             simple_integration,
    #             global_peak_integration,
    #             local_peak_integration,
    #             nb_peak_integration
    # nsum: Number of samples to sum up (is reduced if exceeding available
    # length). (equivalent to first number in
    # hessioxxx option --integration-window)
    # nskip: Number of initial samples skipped (adapted such that interval
    # fits into what is available). Start the integration a number of
    # samples before the peak. (equivalent to second number in
    # hessioxxx option --integration-window)
    # sigamp: Amplitude in ADC counts [igain] above pedestal at which a
    # signal is considered as significant (separate for high gain/low gain).
    # (equivalent to hessioxxx option --integration-threshold)
    # clip_amp: Amplitude in p.e. above which the signal is clipped.
    # (equivalent to hessioxxx option --clip_pixel_amplitude (default 0))
    # lwt: Weight of the local pixel (0: peak from neighbours only,
    # 1: local pixel counts as much as any neighbour).
    # (option in pixel integration function in hessioxxx)
    # display: optionaly you can display events (all telescopes present on it)
    # or per telescope per event. By default the last one.
    # The first one is currently deprecated.
    # level: data level from which information is displayed.

    # The next call to camera_calibration would be equivalent of producing
    # DST0 MC file using:
    # hessioxxx/bin/read_hess -r 4 -u --integration-scheme 4
    # --integration-window 7, 3 --integration-threshold 2, 4
    # --dst-level 0 <MC_prod2_filename>

    calibrated_camera = camera_calibration(
        args.filename,
        parameters={"integrator": "full_integration",
                    "nsum": 7,
                    "nskip": 3,
                    "sigamp": [2, 4],
                    "clip_amp": 0,
                    "lwt": 0},
        disp_args={'event', 'telescope'}, level=1)

    sys.stdout.flush()

    print(TAG, "Closing file...")
    close_file()
