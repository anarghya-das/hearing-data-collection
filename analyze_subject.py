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
DISPS = ['neutral', 'trigger'] # dispositions, may not be the best name

def setup_parser():
    parser = argparse.ArgumentParser(
                    prog='analyze_subject',
                    description='analyze a single control or experimental subject and dump the results in a .pkl file',
                    epilog='Text at the bottom of help')
    parser.add_argument('group', choices=['c', 'e'], help="c for control, e for experimental")
    parser.add_argument('index', help="index of subject, i.e. X for CTRLX or EXPX")
    parser.add_argument('-o', '--output', help="name of the .pkl output file")
    parser.add_argument('-n', '--no_ar_fit', action='store_true', help="do not fit autoreject to epochs, only for saving time debugging")
    parser.add_argument('-p', '--plot_save', action='store_true', help="save plots")
    return parser

def main():
    parser = setup_parser()
    cmd_args = parser.parse_args()

    exp_path = os.path.join("exp_data","02_Experimental")
    control_path = os.path.join("exp_data","01_Control")
    glob_pattern = os.path.join("**","*.xdf")

    out_dir = cmd_args.output
    if not out_dir:
        out_dir = "processed"
        if not os.path.isdir("processed"):
            os.mkdir("processed")
        if cmd_args.group == 'c':
            out_dir = os.path.join(out_dir, "CTRL")
        else:
            out_dir = os.path.join(out_dir, "EXP")
        out_dir += str(cmd_args.index).zfill(2)
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

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

    if cmd_args.plot_save:
        for disp in DISPS:
            plot_power_spectrum(psd_normalized_power[disp], psd_freqs['prestim'][disp], already_mean=True)
            plt.savefig(os.path.join(out_dir, f"psd_{disp}.png"))
    
    tfr_delta_freqs = np.logspace(*np.log10([1, 4]), num=8) # alpha band frequencies
    tfr_theta_freqs = np.logspace(*np.log10([4, 8]), num=8) # alpha band frequencies
    tfr_alpha_freqs = np.logspace(*np.log10([8, 13]), num=8) # alpha band frequencies
    tfr_beta_freqs = np.logspace(*np.log10([13, 30]), num=8) # alpha band frequencies
    tfr_gamma_freqs = np.logspace(*np.log10([30, 50]), num=8) # alpha band frequencies

    tfr_bands = {'delta': tfr_delta_freqs, 'theta': tfr_theta_freqs, 'alpha': tfr_alpha_freqs, 'beta': tfr_beta_freqs, 'gamma': tfr_gamma_freqs}

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

    if cmd_args.plot_save:
        for band in tfr_bands:
            for timing in TIMINGS:
                for disp in DISPS:
                    plot_tf_analysis(tfr_power[band][timing][disp])
                    plt.savefig(os.path.join(out_dir, f"tfr_band_{band}_{timing}_{disp}.png"))
    
    if cmd_args.plot_save:
        for band in tfr_bands:
            for disp in DISPS:
                plot_tf_difference(tfr_power[band]['stim'][disp], tfr_power[band]['prestim'][disp])
                plt.savefig(os.path.join(out_dir, f"tfr_diff_{band}_{disp}.png"))
                pass

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

    out_pkl = os.path.join(out_dir, "data.pkl")

    with open(out_pkl, 'wb') as f:
        pickle.dump(output, f)
    
    print(f"pickled output written to {out_dir}")

main()