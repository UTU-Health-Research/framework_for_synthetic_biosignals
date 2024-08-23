from dataclasses import dataclass
import numpy as np
from signal_generator import SignalGenerator
from noise_generator import NoiseGenerator
from beat_interval_generator import BeatIntervalGenerator
from scipy import signal
from utils import create_label, default_field, min_max_normalize
import os
import concurrent.futures
from timeit import default_timer as timer

@dataclass
class PPGWavePrms:
    fst: float=0.0
    snd: float=0.0
    
    def to_list(self):
        return [self.fst, self.snd]
    
@dataclass
class PPGGenerator(SignalGenerator):

    noise_generator: NoiseGenerator = default_field(NoiseGenerator())
    beat_interval_generator: BeatIntervalGenerator = default_field(BeatIntervalGenerator())
    fs: int = 200
    number_of_beats: int = 30
    ppg_amplitude: PPGWavePrms = default_field(PPGWavePrms(0.8, 0.8))
    ppg_amplitude_low: PPGWavePrms = default_field(PPGWavePrms(0.5, 0.5))
    ppg_amplitude_high: PPGWavePrms = default_field(PPGWavePrms(1.0, 0.9))
    ppg_width: PPGWavePrms = default_field(PPGWavePrms(0.7, 1.9))
    ppg_width_low: PPGWavePrms = default_field(PPGWavePrms(0.5, 1.7))
    ppg_width_high: PPGWavePrms = default_field(PPGWavePrms(0.9, 2.1))
    ppg_distance: PPGWavePrms = default_field(PPGWavePrms(-0.26, 0.11))
    ppg_distance_low: PPGWavePrms = default_field(PPGWavePrms(-0.32, 0.06))
    ppg_distance_high: PPGWavePrms = default_field(PPGWavePrms(-0.22, 0.16))
    peak_label_width: int = 5

    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates clean or noisy PPG signal. Also returns labels for peaks and feet and for noise.

        Returns
        ----------
        signal
            Clean or noisy PPG signal.
        peak_inds
            1D array of labels for peaks and feet.
        noise_labels
            Noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last item in the list is the artifact label (array).
        beat_intervals
            Beat intervals in seconds
        """

        self.distance = self.ppg_distance.to_list()
        self.width = self.ppg_width.to_list()
        self.amplitude = self.ppg_amplitude.to_list()
        self.symmetry = [1, 1]
        noise_labels = []

        if self.noise_generator is not None:
            self.beat_interval_generator.duration = None
            self.noise_generator.fs = self.fs
            noise_signal, noise_labels = self.noise_generator.generate()           

            if self.noise_generator.noise_list:
                dur = 0
                for item in self.noise_generator.noise_list:
                    dur += item.duration
                self.beat_interval_generator.duration = dur
            else:
                self.beat_interval_generator.duration = self.noise_generator.noise_type.duration

        self.beat_interval_generator.n = self.number_of_beats
        beat_intervals = self.beat_interval_generator.generate()
        if self.beat_interval_generator.duration is None:
            self.beat_interval_generator.duration = np.sum(beat_intervals)
        signal_, beat_intervals = super().generate(beat_intervals, self.fs)

        # Find peaks.
        peaks, feet = [], []
        for idx in np.cumsum(beat_intervals):
            # Index range for finding the foot.
            i1 = max(0, int(idx - 0.1 * self.fs))
            i2 = min(np.sum(beat_intervals), int(idx + 0.1 * self.fs))
            f = i1 + np.argmin(signal_[i1:i2])
            feet.append(f)
            # End index of the range for finding the peak.
            i2 = min(np.sum(beat_intervals), int(f + 0.6 * self.fs))
            # Find the relative maxima.
            arg_rel_maxs = signal.argrelmax(signal_[f:i2])[0]
            if len(arg_rel_maxs) > 0:
                peaks.append(f + arg_rel_maxs[0])

        peaks.insert(0, np.argmax(signal_[0:feet[0]]))
        peak_inds = np.zeros(len(signal_))
        peak_inds = create_label(peak_inds, np.array(peaks), 2, self.peak_label_width)
        peak_inds = create_label(peak_inds, np.array(feet), 1, self.peak_label_width)

        #randomize the beginning of the signal
        random_start = int(np.random.uniform(0,0.25)*self.fs)
        signal_ = signal_[random_start:]
        peak_inds = peak_inds[random_start:]
        
        if self.noise_generator is not None:
            signal_, peak_inds, noise_labels = self.noise_generator.combine_signal_noise(signal_, noise_signal, peak_inds, noise_labels)       

        signal_ = signal_[:int(self.fs*self.beat_interval_generator.duration)]
        peak_inds = peak_inds[:int(self.fs*self.beat_interval_generator.duration)]

        return signal_, peak_inds, noise_labels, beat_intervals/self.fs
    
    
    def generate_random_set(self, number_of_signals, duration):
        """
        Generates set of random PPG signals.
        
        Parameters
        ----------
        number_of_signals
            Number of generated signals.
        duration
            Duration of each signal.
        Returns
        ----------
        signal
            List of clean or noisy PPG signal.
        peak_inds
            List of 1D array of labels for peaks and feet.
        noise_labels
            List of noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last item in the list is the artifact label (array).
        beat_list
            List of beat intervals.
        """
        
        signals, peak_inds, noise_labels, beats_list = [], [], [], []
        
        self.beat_interval_generator.duration = duration
        for _ in range(number_of_signals):
            x = np.random.uniform(0, 1)
            self.ppg_distance = self._randomize_prms(self.ppg_distance_low, self.ppg_distance_high, x)
            self.ppg_width = self._randomize_prms(self.ppg_width_low, self.ppg_width_high, x)
            self.ppg_amplitude = self._randomize_prms(self.ppg_amplitude_low, self.ppg_amplitude_high, x)
            self.beat_interval_generator.randomize()
        
            if self.noise_generator is not None:
                self.noise_generator.noise_type.duration = duration
                self.noise_generator.randomize()

            signal, peaks, label, beats = self.generate()
            signals.append(signal)
            peak_inds.append(peaks)
            noise_labels.append(label)
            beats_list.append(beats)
        
        return signals, peak_inds, noise_labels, beats_list

    
    def _randomize_prms(self, low_prms: PPGWavePrms, 
                          high_prms: PPGWavePrms, x: float) -> np.ndarray:
        """
        Randomizes waveform parameters.
        
        Parameters
        ----------
        low_prms
            Lower limit
        high_prms
            Upper limit
        x
            Random coefficient between 0 and 1.

        Returns
        ----------
        randomized wave parameters.
        """
        arr = np.zeros(len(low_prms.__dataclass_fields__))
        for i, (f_low, f_high) in enumerate(zip(low_prms.__dataclass_fields__, 
                                                high_prms.__dataclass_fields__)):
            arr[i] = getattr(low_prms, f_low) + x*(getattr(high_prms, f_high)-getattr(low_prms, f_low))
        
        return PPGWavePrms(arr[0], arr[1])
    
    def gen_ppg_with_ptt(self, beat_intervals: np.ndarray, offset: float, change_offset: float, peaks_ecg: np.ndarray, idx: float) -> np.ndarray:
        """Sets pulse transit time.

        Parameters
        ----------
        beat_intervals
            Beat intervals
        offset
            The time difference between PPG and ECG.
        change_offset
            The change of the time difference.
        peaks_ecg
            The array of the labels of the ECG peaks
        idx
            Index where the change happens in relation to the signal.
        Returns
        ----------
        ppg
            PPG signal
        peaks_ppg
            Peaks of the PPG signal
        noise_labels
            Noise label
        beat_intervals_ppg
            Beat intervals
        """
        x = np.linspace(-5, 5, 10)
        mu = 0
        var = 2
        gaussian = 1./(np.sqrt(2.*np.pi)*var)*np.exp(-np.power((x - mu)/var, 2.)/2)
        gaussian_cumsum = min_max_normalize(np.cumsum(gaussian), 0, change_offset)
        gaussian = np.diff(gaussian_cumsum)
        gaussian = gaussian + 1
        before = np.ones(int((len(beat_intervals)-9)*idx))
        after = np.ones(int((len(beat_intervals)-9)*(1-idx)))
        if len(before) + len(after) + len(gaussian) != len(beat_intervals):
            after = np.concatenate((after, np.ones(len(beat_intervals)-(len(before) + len(after) + len(gaussian)))))
        gaussian = np.concatenate((before, gaussian, after))
        beat_intervals = beat_intervals * gaussian
        self.beat_interval_generator.beat_intervals = beat_intervals
        ppg, peaks_ppg, noise_labels, beat_intervals_ppg = self.generate()
        first_r_peak = np.argwhere(peaks_ecg == 3)[0]
        first_fst_peak = np.argwhere(peaks_ppg == 2)[0]
        ppg = np.roll(ppg, (first_r_peak-first_fst_peak)+(int(offset*self.fs)))
        peaks_ppg = np.roll(peaks_ppg, (first_r_peak-first_fst_peak)+(int(offset*self.fs)))

        return ppg, peaks_ppg, noise_labels, beat_intervals_ppg


    def generate_random_set_parallel(self, number_of_signals, duration):
        """Generates random PPG signals using parallel computing.
        Parameters
        ----------
        number_of_signals
            Number of generated signals.
        duration
            Duration of each signal.
        Returns
        -------
        results : (list, list, list, list)
            Tuple of lists: signals, peaks, noise labels, beat intervals
        """
        # Record the time it takes to run the function.
        start_time = timer()
        # Use all logical processors.
        workers_count = os.cpu_count()
        # Create an array of batch sizes for each processor.
        batches = [number_of_signals // workers_count] * workers_count
        batches[-1] += number_of_signals - np.sum(batches)
        # Initialize empty lists to hold the results.
        results = [[], [], [], []]
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers_count) \
                as executor:
            # Store the futures into a dict --> Completed futures can then be
            # deleted to release RAM.
            futures = {}
            for b in batches:
                f = executor.submit(self.generate_random_set, b, duration)
                futures[f] = f
            futures_count = len(futures)
            futures_completed_count = 0
            for i, f in enumerate(concurrent.futures.as_completed(futures)):
                futures_completed_count += 1
                if futures_completed_count == futures_count:
                    print('All futures completed.')
                else:
                    print(
                        f'Future {futures_completed_count}/{futures_count} completed...      ', end='\r', flush=True)
                try:
                    # Get results.
                    res = f.result()
                except Exception as e:
                    print(f'''Exception occurred while trying to get future\'s 
                        result: {e}''')
                else:
                    for i in range(len(results)):
                        for j in range(number_of_signals//workers_count):
                            results[i].append(res[i][j])
                del futures[f]
        time_elapsed = timer() - start_time
        print(f'Time elapsed: {int(time_elapsed // 60)} m '
            f'{round(time_elapsed % 60, 3)} s')
        return results