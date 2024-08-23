import numpy as np
from scipy import interpolate
from wfdb import rdsamp, rdann
from wfdb.processing import (
    resample_singlechan,
)
from dataclasses import field
import copy


def interpolate_(x: np.ndarray, new_x: np.ndarray, y: np.ndarray, fill_value: str ='nan') -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolates y based on x, by creating n point in x and preserving end points.
    """
    ip = interpolate.interp1d(x, y, kind='linear', fill_value=fill_value)
    new_y = ip(new_x)

    return new_y, new_x

def min_max_normalize(signal_: np.ndarray, min_val: int =0, max_val: int=1) -> np.ndarray:
    """
    Min-max normalizes the signal.
    """
    s_norm = min_val + (signal_ - np.nanmin(signal_)) * (max_val - min_val) / \
        (np.nanmax(signal_) - np.nanmin(signal_))

    return s_norm

def zero_mean(x: np.ndarray) -> np.ndarray:
    """
    Zero-mean of x.
    """
    x = x - np.mean(x)

    return x

def find_corresponding(arr, locs1, w, sym = True):
    locs2 = []
    if sym:
        w1, w2 = int(w/2), int(w/2)
    else:
        w1, w2 = int(0), int(w)
    for loc in locs1:
        l1, l2 = max(loc - w1, 0), min(loc + w2, len(arr))
        locs2.append(l1 + np.argmax(arr[l1:l2]))
        
    return np.array(locs2)


def get_beats(annotation):
    """
    Extract beat indices and types of the beats.
    Beat indices indicate location of the beat as samples from the
    beg of the signal. Beat types are standard character
    annotations used by the PhysioNet.
    Parameters
    ----------
    annotation : wfdb.io.annotation.Annotation
        wfdb annotation object
    Returns
    -------
    beats : array
        beat locations (samples from the beg of signal)
    symbols : array
        beat symbols (types of beats)
    """
    # All beat annotations
    beat_annotations = ['N', 'L', 'R', 'B', 'A',
                        'a', 'e', 'J', 'V', 'r',
                        'F', 'S', 'j', 'n', 'E',
                        '/', 'Q', 'f', '?']

    # Get indices and symbols of the beat annotations
    indices = np.isin(annotation.symbol, beat_annotations)
    symbols = np.asarray(annotation.symbol)[indices]
    beats = annotation.sample[indices]

    return beats, symbols


def data_from_records(records, channel, db):
    """
    Extract ECG, beat locations and beat types from Physionet database.
    Takes a list of record names, ECG channel index and name of the
    PhysioNet data base. Tested only with db == 'mitdb'.
    Parameters
    ----------
    records : list
        list of file paths to the wfdb-records
    channel : int
        ECG channel that is wanted from each record
    db : string
        Name of the PhysioNet ECG database
    Returns
    -------
    signals : list
        list of single channel ECG records stored as numpy arrays
    beat_locations : list
        list of numpy arrays where each array stores beat locations as
        samples from the beg of one resampled single channel
        ECG recording
    beat_types : list
        list of numpy arrays where each array stores the information of
        the beat types for the corresponding array in beat_locations
    """
    signals = []
    beat_locations = []
    beat_types = []

    for record in records:
        print('processing record: ', record)
        signal = (rdsamp(record, pn_dir=db))
        signal_fs = signal[1]['fs']
        annotation = rdann(record, 'atr', pn_dir=db)

        # resample to 250 Hz
        signal, annotation = resample_singlechan(
                                signal[0][:, channel],
                                annotation,
                                fs=signal_fs,
                                fs_target=250)

        beat_loc, beat_type = get_beats(annotation)

        signals.append(signal)
        beat_locations.append(beat_loc)
        beat_types.append(beat_type)

    return signals, beat_locations, beat_types

def default_field(obj):
    """Setting a default value
    for a dataclass attribute."""
    return field(default_factory=lambda: copy.deepcopy(obj))

def create_label(arr, locations, label_number, label_width=5):
    for l in locations:
        i1 = max(0, l - label_width // 2)
        i2 = min(len(arr), l + label_width // 2 + 1)
        arr[i1:i2] = label_number
    return arr


