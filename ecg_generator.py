from dataclasses import dataclass
import numpy as np
from signal_generator import SignalGenerator
from noise_generator import NoiseGenerator
from beat_interval_generator import BeatIntervalGenerator
from utils import create_label, default_field
import os
import concurrent.futures
from timeit import default_timer as timer

@dataclass
class ECGWavePrms:
    p: float=0.0
    q: float=0.0
    r: float=0.0
    s: float=0.0
    t: float=0.0
    
    def to_list(self):
        return [self.p, self.q, self.r, 
                self.s, self.t]

@dataclass
class ECGGenerator(SignalGenerator):

    noise_generator: NoiseGenerator = default_field(NoiseGenerator())
    beat_interval_generator: BeatIntervalGenerator = default_field(BeatIntervalGenerator())
    fs: int = 200
    number_of_beats: int = 30
    ecg_amplitude: ECGWavePrms = default_field(ECGWavePrms(0.1, -0.08, 1.0, -0.08, 0.3))
    ecg_amplitude_low: ECGWavePrms = default_field(ECGWavePrms(0.05, -0.05, 0.8, -0.05, 0.1))
    ecg_amplitude_high: ECGWavePrms = default_field(ECGWavePrms(0.2, -0.2, 1.2, -0.2, 0.6))
    ecg_width: ECGWavePrms = default_field(ECGWavePrms(0.15, 0.1, 0.1, 0.1, 0.5))
    ecg_width_low: ECGWavePrms = default_field(ECGWavePrms(0.065, 0.03, 0.06, 0.03, 0.085))
    ecg_width_high: ECGWavePrms = default_field(ECGWavePrms(0.085, 0.08, 0.085, 0.08, 0.21))
    ecg_distance: ECGWavePrms = default_field(ECGWavePrms(-0.12, -0.04, 0.0, 0.03, 0.25))
    ecg_distance_low: ECGWavePrms = default_field(ECGWavePrms(-0.12, -0.03, 0.0, 0.03, 0.2))
    ecg_distance_high: ECGWavePrms = default_field(ECGWavePrms(-0.18, -0.05, 0.0, 0.05, 0.25))
    ecg_symmetry: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 3.0))
    ecg_symmetry_low: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 1.0))
    ecg_symmetry_high: ECGWavePrms = default_field(ECGWavePrms(1.0, 1.0, 1.0, 1.0, 5.0))
    peak_label_width: int = 5


    def generate(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates clean or noisy ECG signals. Also returns indeces for P and T waves and R peaks and labels for noise.
        
        Returns
        ----------
        signal
            Clean or noisy ECG signal.
        peak_inds
            1D array of labels for P, R and T waves.
        noise_labels
            Noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last tuple in the list is the artifact label (array).
        beat_intervals
            Beat intervals in seconds
        """
        
        self.distance = self.ecg_distance.to_list()
        self.width = self.ecg_width.to_list()
        self.amplitude = self.ecg_amplitude.to_list()
        self.symmetry = self.ecg_symmetry.to_list()
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
        signal, beat_intervals = super().generate(beat_intervals, self.fs)

        # Find R peaks.
        r_peaks = np.zeros(len(beat_intervals), dtype=int)
        for i, nn in enumerate(beat_intervals):
            r_peaks[i] = nn // 2 if i == 0 else nn // 2 + np.sum(beat_intervals[:i])
        
        p_waves = r_peaks + np.array(self.ecg_distance.p*beat_intervals, dtype=int)
        t_waves = r_peaks + np.array(self.ecg_distance.t*beat_intervals, dtype=int)
        peak_inds = np.zeros(len(signal))
        peak_inds = create_label(peak_inds, p_waves, 1, self.peak_label_width)
        peak_inds = create_label(peak_inds, r_peaks, 3, self.peak_label_width)
        peak_inds = create_label(peak_inds, t_waves, 2, self.peak_label_width)

        random_start = int(np.random.uniform(0,0.25)*self.fs)
        signal = signal[random_start:]
        peak_inds = peak_inds[random_start:]
        
        if self.noise_generator is not None:
            signal, peak_inds, noise_labels = self.noise_generator.combine_signal_noise(signal, noise_signal, peak_inds, noise_labels)  

        signal = signal[:int(self.fs*self.beat_interval_generator.duration)]
        peak_inds = peak_inds[:int(self.fs*self.beat_interval_generator.duration)]

        return signal, peak_inds, noise_labels, beat_intervals/self.fs
    
    
    def generate_random_set(self, number_of_signals:int, duration:float):
        """
        Generates set of random ECG signals.
        
        Parameters
        ----------
        number_of_signals
            Number of generated signals.
        duration
            Duration of each signal.
        Returns
        ----------
        signal
            List of clean or noisy ECG signal.
        peak_inds
            List of 1D array of labels for P, R and T waves.
        noise_labels
            List of noise labels: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last item in the list is the artifact label (array).
        beat_list
            List of beat intervals.
        """
        signals, peak_inds, noise_labels, beats_list = [], [], [], []
        self.beat_interval_generator.duration = duration
        for _ in range(number_of_signals):
            self.beat_interval_generator.beat_intervals = None
            self.ecg_distance = self._randomize_prms(self.ecg_distance_low, self.ecg_distance_high)
            self.ecg_width = self._randomize_prms(self.ecg_width_low, self.ecg_width_high)
            self.ecg_amplitude = self._randomize_prms(self.ecg_amplitude_low, self.ecg_amplitude_high)
            self.ecg_symmetry = self._randomize_prms(self.ecg_symmetry_low, self.ecg_symmetry_high)
            self.beat_interval_generator.randomize()
            
            if self.noise_generator is not None:
                self.noise_generator.noise_type.duration = duration
                self.noise_generator.randomize()
            
            signal, r_peaks, label, beats = self.generate()
            signals.append(signal[:int(duration*self.fs)])
            peak_inds.append(r_peaks)
            noise_labels.append(label)
            beats_list.append(beats)
        
        return signals, peak_inds, noise_labels, beats_list

    
    def _randomize_prms(self, low_prms: ECGWavePrms, 
                          high_prms: ECGWavePrms) -> np.ndarray:
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
            arr[i] = np.random.uniform(getattr(low_prms, f_low), getattr(high_prms, f_high))
        
        return ECGWavePrms(arr[0], arr[1], arr[2], arr[3], arr[4])
    
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
