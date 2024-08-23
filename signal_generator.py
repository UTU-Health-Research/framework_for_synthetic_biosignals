import numpy as np
from scipy import integrate

import utils

class SignalGenerator:

    def __init__(self, distance: list[float], 
                 width: list[float], 
                 amplitude: list[float],
                 symmetry: list[float] = [1.0, 1.0]):
        self.distance = distance
        self.width = width
        self.amplitude = amplitude
        self.symmetry = symmetry

    def generate(self, beat_intervals, fs) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates clean biosignal.

        Parameters
        ----------
        beat_intervals
            Beat intervals for the signal
        fs
            Sampling frequency
        Returns
        ----------
        synt
            Synthetic biosignal
        beat_intervals
            Beat intervals multiplied with fs

        """
        beat_intervals = np.array(beat_intervals*fs + 0.5, dtype=int)

        # Cumulative sum of pulse widths.
        pulse_widths_cumsum = np.zeros(len(beat_intervals) + 1, dtype=int)
        pulse_widths_cumsum[1:] = np.cumsum(beat_intervals)

        # Derivatives.
        n_dist = len(self.distance)
        s_len = np.sum(beat_intervals)
        ders = np.zeros((n_dist, s_len))
        for i in range(n_dist):
            # Skip if the amplitude is zero.
            if self.amplitude[i] == 0:
                continue

            # Phase signal.
            phase = np.zeros(s_len)
            for j in range(1, pulse_widths_cumsum.size):
                phase[pulse_widths_cumsum[j - 1]:pulse_widths_cumsum[j]] = \
                    np.roll(np.linspace(-np.pi, np.pi, beat_intervals[j - 1]),
                            int(self.distance[i]*beat_intervals[j - 1]))
            # Derivative.
            neg_ind = np.where(phase<=0)
            pos_ind = np.where(phase>0)  
            ders[i, neg_ind] = (-phase[neg_ind])  / (self.width[i]**2) * self.amplitude[i] * 2*np.pi* \
                np.exp((- phase[neg_ind]**2) / (2 * self.width[i] ** 2))
            ders[i, pos_ind] = (-phase[pos_ind] * self.symmetry[i])  / (self.width[i]**2) * self.amplitude[i]* 2*np.pi * \
                np.exp((-self.symmetry[i] * phase[pos_ind]**2) / (2 * self.width[i] ** 2))

        # The final derivative of the signal is the sum of the derivatives.
        der_raw = np.sum(ders, axis=0)

        # Compute the final synthetic signal with numerical integration.
        synt = integrate.cumtrapz(der_raw, dx=1/fs, initial=0)

        # Normalize to range [0, 1].
        synt = utils.min_max_normalize(synt)

        return synt, beat_intervals
