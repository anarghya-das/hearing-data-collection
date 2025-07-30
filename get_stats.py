import argparse
import numpy as np
import os
import pickle

TIMINGS = ['prestim', 'stim']
DISPS = ['neutral', 'trigger'] # dispositions, may not be the best name

def setup_parser():
    parser = argparse.ArgumentParser(
                    prog='get_stats',
                    description='TODO',
                    epilog='Text at the bottom of help')
    parser.add_argument('group', choices=['c', 'e'], help="c for control, e for experimental")
    parser.add_argument('index', help="index of subject, i.e. X for CTRLX or EXPX")
    parser.add_argument('-o', '--output', help="name of the .pkl output file")
    return parser

def main():
    parser = setup_parser()
    cmd_args = parser.parse_args()

    in_dir = "processed"
    if cmd_args.group == 'c':
        in_dir = os.path.join(in_dir, "CTRL")
    else:
        in_dir = os.path.join(in_dir, "EXP")
    in_dir += str(cmd_args.index).zfill(2)

    in_pkl = os.path.join(in_dir, "data.pkl")
    with open(in_pkl, 'rb') as f:
        data = pickle.load(f)

    out_dir = cmd_args.output
    if not out_dir:
        out_dir = "stats"
        if not os.path.isdir("stats"):
            os.mkdir("stats")
        if cmd_args.group == 'c':
            out_dir = os.path.join(out_dir, "CTRL")
        else:
            out_dir = os.path.join(out_dir, "EXP")
        out_dir += str(cmd_args.index).zfill(2)

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    
    tfr_bands = {'delta', 'theta', 'alpha', 'beta', 'gamma'}
    avg_negative = {}
    for disp in DISPS:
        avg_negative[disp] = {}
        for band in tfr_bands:
            prestim_power = data['tfr'][band]['power']['prestim'][disp].data
            stim_power = data['tfr'][band]['power']['stim'][disp].data
            diff_power = stim_power - prestim_power
            avg_negative[disp][band] = np.where(diff_power > 0, 0, diff_power).mean()
    
    avg_negative_ratio = {
        band: float(avg_negative['trigger'][band] / avg_negative['neutral'][band])
        for band in tfr_bands
    }
    print(sorted(avg_negative_ratio.items()))
        
main()
