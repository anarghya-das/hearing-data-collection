import matplotlib.pyplot as plt
import glob
import numpy as np
import os
import mne
import pyxdf


def closest_points_vector(eeg_timestamps, marker_timestamps):
    # Get the insertion indices for each marker timestamp
    indices = np.searchsorted(eeg_timestamps, marker_timestamps)

    # Preallocate the output array as a copy of indices.
    closest_eeg_indices = indices.copy()

    # Create a mask for markers where the insertion index equals 0 (marker before first EEG timestamp)
    mask_begin = (indices == 0)
    # For these, the closest EEG index is 0 (they cannot use a previous value)
    closest_eeg_indices[mask_begin] = 0

    # Create a mask for markers where the insertion index equals the length of the EEG timestamps
    mask_end = (indices == len(eeg_timestamps))
    # For these markers, set the closest EEG index to the last index
    closest_eeg_indices[mask_end] = len(eeg_timestamps) - 1

    # Create a mask for the "middle" markers, i.e., not at the very beginning or end
    mask_middle = (indices > 0) & (indices < len(eeg_timestamps))

    # For markers in the middle, compute the distance to the previous and next EEG timestamps:
    prev_times = eeg_timestamps[indices[mask_middle] - 1]
    next_times = eeg_timestamps[indices[mask_middle]]
    marker_times_middle = marker_timestamps[mask_middle]

    # Calculate the differences
    diff_prev = marker_times_middle - prev_times
    diff_next = next_times - marker_times_middle

    # For each marker in the middle, choose the index of the EEG timestamp that is closer:
    # If the distance to the previous timestamp is less or equal than the distance to the next,
    # then we pick indices[mask_middle]-1; otherwise, we pick indices[mask_middle].
    closest_eeg_indices[mask_middle] = np.where(diff_prev <= diff_next,
                                                indices[mask_middle] - 1,
                                                indices[mask_middle])

    return closest_eeg_indices


def create_mappings(event_names, prefix):
    marker_dict = {p: i for i, p in enumerate(np.unique(event_names))}
    id_binding = {v: k for k, v in marker_dict.items()}
    category_mapping = {}
    for p in prefix:
        # All keys for this prefix
        sub_map = {k: v for k, v in marker_dict.items() if k.startswith(p)}
        # Special handling for 'ast'
        if p == 'ast':
            # Separate keys containing 'control' from others, store as dicts
            ast_prefix = {'prestim', 'stim', 'poststim'}
            ast_map = {}
            for ap in ast_prefix:
                sub_map = {k: v for k, v in marker_dict.items() if k.startswith(p + "_" + ap)}
                ast_keys = list(sub_map.keys())
                ast_map[ap] = {
                    'neutral': {},
                    'trigger': {},
                    'all': {}
                }
                for key in ast_keys:
                    ast_map[ap]['all'][key] = sub_map[key]
                    if 'control' in key.lower():
                        ast_map[ap]['neutral'][key] = sub_map[key]
                    else:
                        ast_map[ap]['trigger'][key] = sub_map[key]
            category_mapping[p] = ast_map
        else:
            category_mapping[p] = sub_map
    return marker_dict, id_binding, category_mapping


def create_events(time_points, event_mapping, event_names):
    label_id_func = np.vectorize(event_mapping.get)
    events = np.zeros((len(time_points), 3), dtype=int)
    events[:, 0] = time_points
    events[:, 2] = label_id_func(event_names)
    return events


def create_mne(eeg_stream, events, id_binding,
               flat_voltage=0.1, bandpass={'low': 1, 'high': 50}, notch_freq=60):
    ch_labels = ['L1', 'L2', 'L4', 'L5', 'L7', 'L8', 'L9', 'L10',
                 'R1', 'R2', 'R4', 'R5', 'R7', 'R8', 'R9', 'R10']
    sampling_rate = float(eeg_stream['info']['nominal_srate'][0])
    if sampling_rate != 125:
        raise ValueError(
            f"Expected sampling rate of 125 Hz, got {sampling_rate} Hz")
    # Openbci EEG data is in microvolts, convert to volts for MNE
    eeg_data = eeg_stream['time_series'].T * 1e-6
    info = mne.create_info(
        ch_names=ch_labels, sfreq=sampling_rate, ch_types='eeg')
    raw = mne.io.RawArray(eeg_data, info)

    if flat_voltage != None:
        flat_voltage *= 1e-6  # Flat voltage threshold in Volts
        _, bads = mne.preprocessing.annotate_amplitude(
            raw, flat=dict(eeg=flat_voltage))
        raw.info['bads'] = bads
        print(f"Bad channels: {bads}")

    # raw.interpolate_bads()
    annot = mne.annotations_from_events(
        events, raw.info['sfreq'], id_binding)
    raw.set_annotations(annot)

    if notch_freq is not None:
        raw = raw.notch_filter(
            np.arange(notch_freq, sampling_rate/2, notch_freq), picks='eeg')

    # raw.set_montage(montage)  # Set the montage to the raw object
    if bandpass != None:
        raw = raw.filter(l_freq=bandpass['low'], h_freq=bandpass['high'])
        raw = raw.set_eeg_reference('average')
    return raw


def parse_xdf(file_path, eeg_stream_name='obci_eeg1'):
    data, header = pyxdf.load_xdf(file_path)
    # print([stream['info']['type'][0] for stream in data])
    # Extract the EEG and marker streams
    marker_stream = next(
        stream for stream in data if stream['info']['type'][0] == 'Markers')
    eeg_stream = next(
        stream for stream in data
        if stream['info']['type'][0] == 'EEG' and stream['info']['name'][0] == eeg_stream_name)
    marker_timestamps = marker_stream['time_stamps']
    marker_data = np.array(marker_stream['time_series']).squeeze()
    eeg_timestamps = eeg_stream['time_stamps']
    eeg_insert_points = closest_points_vector(
        eeg_timestamps, marker_timestamps)
    return marker_data, eeg_stream, eeg_insert_points


def get_event_names(files, prefix='ast_stim', exclude_participants=[]):
    """
    Extracts event names from marker data that start with a given prefix.
    Returns:
        name_mapping: dict mapping participant to their event names
        common_names: set of event names present for every participant
    """
    name_mapping = {}
    all_names = []
    for file in files:
        participant_number = file.split(os.sep)[2]
        if participant_number in exclude_participants:
            continue
        participant_id = file.split(os.sep)[-1].split('_')[0]
        marker_data, _, _ = parse_xdf(file)
        names = {str(f) for f in np.unique(marker_data)
                 if str(f).startswith(prefix)}
        name_mapping[f"{participant_number}_{participant_id}"] = names
        all_names.append(names)
    # Intersection: names present for every participant
    if all_names:
        common_names = set.intersection(*all_names)
    else:
        common_names = set()
    return name_mapping, common_names


def read_data(file_path, eeg_stream_name='obci_eeg1', bindings=None,
              bandpass={'low': 1, 'high': 50}, flat_voltage=0.1):
    marker_data, eeg_stream, eeg_insert_points = parse_xdf(
        file_path, eeg_stream_name)
    # Create MNE events from the marker data
    if bindings is None:
        bindings = ['pmt', 'hlt', 'let', 'ast']
    marker_dict, id_binding, category_mapping = create_mappings(
        marker_data, bindings)
    events = create_events(eeg_insert_points, marker_dict, marker_data)
    raw = create_mne(eeg_stream, events, id_binding,
                     bandpass=bandpass, flat_voltage=flat_voltage)
    return raw, events, category_mapping
    # return eeg_stream, events, id_binding, category_mapping


def calculate_power_spectrum(epoch, method='multitaper', fmin=1, fmax=20, mean=False):
    """Calculate power spectrum for given epochs."""
    psd = epoch.compute_psd(method=method, fmin=fmin, fmax=fmax)
    power, freqs = psd.get_data(return_freqs=True)
    # if compute_method == 'wavelet':
    #     freqs = np.logspace(*np.log10([fmin, fmax]), num=8)
    #     n_cycles = freqs / 2.0  # different number of cycle per frequency
    #     power, itc = epoch.compute_tfr(
    #         method="morlet",
    #         freqs=freqs,
    #         n_cycles=n_cycles,
    #         average=True,
    #         return_itc=True,
    #         decim=3,
    #     )
    if mean:
        power = power.mean(axis=(0, 1))
    return power, freqs


def plot_power_spectrum(power, freq, average_axis=(0, 1), title='Power Spectrum', already_mean=False):
    print(f"Power shape: {power.shape}, Frequency shape: {freq.shape}")
    bands = {
        'delta': (freq[0], 4, 'blue'),
        'theta': (4, 8, 'green'),
        'alpha': (8, 13, 'orange'),
        'beta': (13, 30, 'red'),
        'gamma': (30, freq[-1], 'purple')
    }

    if not already_mean:
        mean_power = power.mean(axis=average_axis)
    else:
        mean_power = power

    fig = plt.figure(figsize=(10, 6))
    for band, (low, high, color) in bands.items():
        idx = np.where((freq >= low) & (freq < high))[0]
        if len(idx) > 0:
            plt.fill_between(freq[idx], mean_power[idx],
                             color=color, alpha=0.5, label=band.capitalize())

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Power Spectrum by Frequency Band')
    plt.xlim(0, 50)
    plt.grid()
    plt.legend()
    plt.tight_layout()

    return fig

# probably redundant but fits the pattern of having a compute and plot function for each step
def compute_tf_analysis(epochs, freqs, n_cycles):
    return epochs.compute_tfr(
        method="morlet",
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=True,
        decim=3,
    )

def plot_tf_analysis(power):
    channels = power.ch_names
    n_channels = len(channels)
    n_cols = 4
    n_rows = int(np.ceil(n_channels / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 3 * n_rows))
    axes = axes.flatten()

    # Normalize each channel's power between 0 and 1
    for idx, ch in enumerate(channels):
        data = power.data[idx]
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
        im = axes[idx].imshow(
            data_norm,
            aspect='auto',
            origin='lower',
            extent=[power.times[0], power.times[-1], power.freqs[0], power.freqs[-1]],
            vmin=0,
            vmax=1,
            cmap='Reds'
        )
        axes[idx].set_title(ch)
        axes[idx].set_ylabel('Freq (Hz)')
        axes[idx].set_xlabel('Time (s)')

    # Hide any unused subplots
    for ax in axes[n_channels:]:
        ax.axis('off')

    # Add a single colorbar to the right
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    plt.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.88, 1])

def compute_band_ratios(power, freqs, bands={
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 50)
}):
    """
    Compute normalized average power per band per channel for PSD data.
    power: shape (n_epochs, n_channels, n_freqs)
    freqs: shape (n_freqs,)
    Returns: dict of band -> array of shape (n_channels,)
    """

    # Get frequency indices for each band
    band_indices = {
        band: np.where((freqs >= low) & (freqs < high))[0]
        for band, (low, high) in bands.items()
    }

    # Average power over epochs: shape (n_channels, n_freqs)
    mean_power = power.mean(axis=0)

    # Calculate normalized average power per band per channel
    band_power_norm = {}
    for band, idxs in band_indices.items():
        band_power_norm[band] = []
        for ch in range(mean_power.shape[0]):
            data = mean_power[ch]
            # Normalize the PSD for this channel
            data_norm = (data - data.min()) / (data.max() - data.min() + 1e-12)
            # Average over band freqs
            band_avg = data_norm[idxs].mean()
            band_power_norm[band].append(band_avg)
        band_power_norm[band] = np.array(band_power_norm[band])

    return band_power_norm


if __name__ == "__main__":
    exp_path = os.path.join("exp_data", "02_Experimental")
    control_path = os.path.join("exp_data", "01_Control")
    glob_pattern = os.path.join("**", "*.xdf")

    exp_files = glob.glob(os.path.join(exp_path, glob_pattern), recursive=True)
    control_files = glob.glob(os.path.join(
        control_path, glob_pattern), recursive=True)[:6]
# CTRL03-sub-129059-old error
    read_data(exp_files[0])
