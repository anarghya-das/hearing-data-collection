import argparse
import autoreject
import glob
import matplotlib.pyplot as plt
import mne
from mne_icalabel import label_components
from mne.preprocessing import ICA
import numpy as np
import os
import pickle
import pyxdf
from utils import *

TIMINGS = ['prestim', 'stim']
DISPS= ['neutral', 'trigger'] # dispositions, may not be the best name

def setup_parser():
    parser = argparse.ArgumentParser(
                    prog='analyze_subject',
                    description='analyze a single control or experimental subject and dump the results in a .pkl file',
                    epilog='Text at the bottom of help')
    parser.add_argument('group', choices=['c', 'e'], help="c for control, e for experimental")
    parser.add_argument('index', help="index of subject, i.e. X for CTRLX or EXPX")
    parser.add_argument('-o', '--output', help="name of the .pkl output file")
    parser.add_argument('-n', '--no_ar_fit', action='store_true', help="do not fit autoreject to epochs, only for saving time debugging")
    parser.add_argument('-d', '--display_plots', action='store_true', help="display plots")
    return parser

def main():
    parser = setup_parser()
    cmd_args = parser.parse_args()

    exp_path = os.path.join("exp_data","02_Experimental")
    control_path = os.path.join("exp_data","01_Control")
    glob_pattern = os.path.join("**","*.xdf")

    if cmd_args.group == 'c':
        files = glob.glob(os.path.join(control_path,glob_pattern),recursive=True)
    else:
        files = glob.glob(os.path.join(exp_path,glob_pattern),recursive=True)
    sub_idx = int(cmd_args.index) - 1; # account for 1-indexing

    raw, events, mapping = read_data(
        files[sub_idx], bandpass={'low': 2, 'high': 50}, flat_voltage=0.1)
    montage = mne.channels.read_custom_montage(
        'ceegrid_coords.csv', coord_frame='head')
    raw.set_montage(montage)
    raw.interpolate_bads(reset_bads=True)
    event_type = 'ast'

    epochs = {timing: {disp: mne.Epochs(
        raw, events, event_id=mapping[event_type][timing][disp],
        tmin=-0.5, tmax=4, preload=True)
        for disp in DISPS}
        for timing in TIMINGS}
    
    ar = autoreject.AutoReject(n_interpolate=[1, 2, 3, 4], random_state=11,
                           n_jobs=1, verbose=True)

    if cmd_args.no_ar_fit:
        ar.fit(epochs[TIMINGS[0]][DISPS[0]])
    
    epochs_ar = {}
    for timing in TIMINGS:
        epochs_ar[timing] = {}
        for disp in DISPS:
            if not cmd_args.no_ar_fit:
                ar.fit(epochs[timing][disp])
            epochs_ar[timing][disp] = ar.transform(epochs[timing][disp])

    psd_power = {}
    psd_freqs = {}
    for timing in TIMINGS:
        psd_power[timing] = {}
        psd_freqs[timing] = {}
        for disp in DISPS:
            psd_power[timing][disp], psd_freqs[timing][disp] = calculate_power_spectrum(
                epochs[timing][disp], fmin=2, fmax=50, mean=True)

    # implicit assumption for dividing power spectra
    for disp in DISPS:
        assert np.array_equal(psd_freqs['prestim'][disp], psd_freqs['stim'][disp])

    psd_normalized_power = {}
    for disp in DISPS:
        psd_normalized_power[disp] = psd_power['stim'][disp] / psd_power['prestim'][disp]

    if cmd_args.display_plots:
        for disp in DISPS:
            plot_power_spectrum(
                psd_normalized_power[disp], psd_freqs['prestim'][disp], already_mean=True)
            plt.show()
    
    # TODO: add beta and gamma band frequencies
    tfr_alpha_freqs = np.logspace(*np.log10([1, 20]), num=8) # alpha band frequencies

    tfr_bands = {'alpha': tfr_alpha_freqs}

    tfr_power = {}
    tfr_itc = {}
    for band in tfr_bands:
        tfr_power[band] = {}
        tfr_itc[band] = {}
        for timing in TIMINGS:
            tfr_power[band][timing] = {}
            tfr_itc[band][timing] = {}
            for disp in DISPS:
                tfr_power[band][timing][disp], tfr_itc[band][timing][disp] = compute_tf_analysis(
                    epochs[timing][disp], tfr_bands[band], tfr_bands[band] / 2.0)

    if cmd_args.display_plots:
        for band in tfr_bands:
            for timing in TIMINGS:
                for disp in DISPS:
                    plot_tf_analysis(tfr_power[band][timing][disp])
                    plt.show()
    
    output = {
        'epochs': {},
        'psd': {},
        'tfr': {}
    }

    output['epochs'] = epochs
    output['psd'] = {
        'power': psd_power,
        'normalized_power': psd_normalized_power,
        'freqs': psd_freqs
    }
    output['tfr'] = {
        band: {
            'power': tfr_power[band],
            'itc': tfr_itc[band]
        } for band in tfr_bands
    }

    out_file = cmd_args.output
    if not out_file:
        if cmd_args.group == 'c':
            out_file = "CTRL"
        else:
            out_file = "EXP"
        out_file += str(cmd_args.index).zfill(2)
        out_file += ".pkl"

    with open(out_file, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"pickled output written to {out_file}")

main()