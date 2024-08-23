import numpy as np
from utils import interpolate_, min_max_normalize, zero_mean
from dataclasses import dataclass
import random
from scipy.signal import resample_poly
from wfdb import rdsamp
from utils import default_field

@dataclass
class NoiseType:
    name: str = 'model'
    duration: int = 30
    amplitude: float = 0.1
    alpha: float = 1.0
    c: float = 0.04
    wn: float = 0.01
    point_bool: bool = False
    point_freq: float = 0.5
    point_value: int = 1

    def _get_label(self, fs):
        """
        Returns NoiseType as string.

        Parameters
        ----------
        fs
            sample frequency

        Returns
        ----------
        label
            noise label as string
        """
        label = self.name + ','
        if self.point_bool:
            label = label + ' ' + str(round(self.point_freq*(fs/2))) + ' Hz,'
        if self.name == 'model':
            label = label + f' ({self.c:.1e})/f^'+str(np.round(self.alpha, 3))+f' + ({self.wn:.1e}),'
        label = label + f' {self.amplitude:.1e}'
        
        return label
                

@dataclass
class NoiseGenerator():

    available_noise_types: list[str] = default_field(['walking', 'hand_movement', 
                            'muscle_artifact', 'baseline_wander'])          
    fs: int = 200
    noise_list: list[NoiseType] = default_field([])
    noise_psds = {}
    amplitude_rng: list[float] = default_field([0.005, 0.25])
    noise_type: NoiseType = default_field(NoiseType())
    alpha_rng: list = default_field([0, 1])
    c_rng: list  = default_field([0.0, 0.15])
    wn_rng: list = default_field([0.0, 0.1])
    point_prob: float = 0.25
    point_freq_rng: list = default_field([0.0001, 1])
    point_value_rng: list = default_field([0.1, 1])
    noise_types = [('walking', 0.15), ('hand_movement', 0.15), ('model', 0.4), ('baseline_wander', 0.15), ('muscle_artifact', 0.15)]
    artifact_bool: bool = False
    artifact_prob: float = 0.5
    artifact_amp: float = 1
    artifact_amp_rng: list = default_field([0.1, 1])
    artifact_type: str = 'ma'
    artifact_types = [('ma', 0.5), ('bw', 0.5)]
    artifact_start: float = 0.5
    artifact_start_rng: list = default_field([0, 1])
    artifact_length: float = 3
    artifact_length_rng: list = default_field([1, 10])
    artifact_idx: float = 0.5
    artifact_idx_rng: list = default_field([0, 1])
    __artifact_arrs = None
    __tap_len:int = 20

    def __post_init__(self):
        self.__artifact_arrs = (self._load_noise())
        for nt in self.available_noise_types:
            self.noise_psds[nt] = np.loadtxt(f'./measurements/{nt}.csv', delimiter=',')
        # even tap_len
        self.__tap_len = self.__tap_len - self.__tap_len%2

    
    def generate(self):
        """
        Generates time domain noise stream from measured PSDs or creates first a model PSD. Also optionally adds point frequency to the PSD and an artifact to the noise stream.

        Returns
        ----------
        concat_time_real
            Concatenated time domain noise stream. 
        labels
            Labels for the noise: list of tuples of noise type (str) and signal-length array of amplitude of noise. Last tuple in the list is the artifact label (string, array).
        """

        empty = False
        # If noise_list is empty, the current noise_type is added to noise_list and removed after the noise is generated
        if not self.noise_list:
            empty = True
            self.noise_list.append(self.noise_type)
            
        noises, noise_labels = self._generate_noise()

        concat_time_real, label_indeces = self._concatenate_time_realisations(noises)

        art_label = ('no artefact', np.zeros(len(concat_time_real)))
        if self.artifact_bool:
            concat_time_real, art_label = self._add_artifact(concat_time_real)

        if len(noise_labels) == 1:
            labels = [(noise_labels[0][0], np.ones(len(concat_time_real))* noise_labels[0][1])]
        else:
            labels = self._get_labels(list(zip(*noise_labels))[0], label_indeces, list(zip(*noise_labels))[1])
        
        labels.append(art_label)

        # If the noise_list was empty in the beginning, it is now restored
        if empty:
            self.noise_list = []

        return concat_time_real, labels
    
    def _generate_noise(self):
        """
        Generates the noise or noises given by the noise_type objects in the noise_list and labels them accordingly.
        Raises
        ----------
            ValueError: If an unavailable noise type is given.

        Returns
        ----------
        noises
            Noise or noises in frequency domain.
        noise_labels   
            The labels of the noise.
        """

        noises, noise_labels = [], []
        
        # If multiple noises are given, the overlapping length is added to snippets
        if len(self.noise_list) > 1:
            for i in range(len(self.noise_list)):
                if i == 0 or i == len(self.noise_list)-1:
                    self.noise_list[i].duration += self.__tap_len/2/self.fs
                else:
                    self.noise_list[i].duration += self.__tap_len/self.fs

        for item in self.noise_list:
            self.noise_type = item
            if item.name not in self.available_noise_types:
                if item.name != 'model':
                    raise ValueError("'%s' is not a valid noise type." % item.name)

            if item.name == 'model':
                psd, freq = self._model_psd()
            else:
                psd = self.noise_psds[item.name][0]
                freq = self.noise_psds[item.name][1]

            if item.point_bool:
                psd, freq = self._add_point_frequency(psd, freq)
            time, y = self._get_time_realisation(psd, freq)
            new_fs = self.fs
            new_time = np.linspace(0, (len(time)*(new_fs/(freq[-1]*2)))/new_fs, int((len(time))*(new_fs/(freq[-1]*2))))
            y, time = interpolate_(time, new_time, y, fill_value='extrapolate')
            label = self.noise_type._get_label(fs=self.fs)
            label = (label, item.amplitude)
            y = y[:int(item.duration*self.fs + 0.5)]
            noises.append(y)
            noise_labels.append(label)

        return noises, noise_labels
    

    def randomize(self):
          """
          Randomizes noise parameters.
          """
          
          x = lambda l1, l2: np.random.uniform(l1, l2)
          
          self.noise_type.alpha = x(self.alpha_rng[0], self.alpha_rng[1])
          self.noise_type.c = x(self.c_rng[0], self.c_rng[1])
          self.noise_type.wn = x(self.wn_rng[0], self.wn_rng[1])
          self.noise_type.amplitude = x(self.amplitude_rng[0], self.amplitude_rng[1])
          self.noise_type.point_freq = x(self.point_freq_rng[0], self.point_freq_rng[1])
          self.noise_type.point_value = x(self.point_value_rng[0], self.point_value_rng[1])
          self.noise_type.name = random.choices(list(zip(*self.noise_types))[0], weights=list(zip(*self.noise_types))[1])[0]
          self.noise_type.point_bool = random.choices([True, False], weights=[self.point_prob, 1-self.point_prob])[0]
          self.artifact_bool = random.choices([True, False], weights=[self.artifact_prob, 1-self.artifact_prob])[0]
          self.artifact_type = random.choices(list(zip(*self.artifact_types))[0], weights=list(zip(*self.artifact_types))[1])[0]
          self.artifact_start = x(self.artifact_start_rng[0], self.artifact_start_rng[1])
          self.artifact_length = x(self.artifact_length_rng[0], self.artifact_length_rng[1])
          self.artifact_idx = x(self.artifact_idx_rng[0], self.artifact_idx_rng[1])
          self.artifact_amp = x(self.artifact_amp_rng[0], self.artifact_amp_rng[1])
        
    def _get_time_realisation(self, psd: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]: 
        """
        Interpolates psd and freq and creates the time realisation of given duration (in secs) from the psd.

        Parameters
        ----------
        psd
            PSD of the noise
        freq
            Frequency vector of the PSD

        Returns
        ----------
        time
            Time vector of the signal
        y   
            The generated time realisation
        """
        new_freq = np.linspace(freq[0], freq[-1], int(self.noise_type.duration*self.fs + 0.5), endpoint=True)
        psd, freq = interpolate_(freq, new_freq, psd, fill_value='extrapolate')
        time, y = self._psd2time(freq, psd)
        if self.noise_type.name != 'model':
            y = self.noise_type.amplitude*y/np.std(y)
            
        return time, y

    def _psd2time(self, freq: np.ndarray, psd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Creates time realisation from the psd. Last value of the freq vector is the Nyquist frequency. 

        Parameters
        ----------
        freq
            Frequency vector of the PSD
        psd 
            PSD of the noise
        
        Returns
        ----------
        time
            Time vector of the signal
        y
            The generated time realisation
        """
        if np.isin(freq, 0)[0]:
            freq=np.delete(freq, 0)
            psd=np.delete(psd, 0)
            
        fs = freq[-1]*2
            
        n = len(freq)*2
        w0 = np.zeros(1)

        x = np.random.randn(len(freq)) + 1j*np.random.randn(len(freq))
        w1 = np.sqrt(psd)/2*x
        w = np.concatenate((w0, w1, np.conj(w1)[::-1])).astype(complex)
        
        y = np.sqrt(fs)*np.sqrt(n)*np.real(np.fft.ifft(w))
        time = np.arange(0,n)*(1/fs)
        
        return time, y[1:]



    def _model_psd(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates the PSD based on a mathematical model and given parameters. 

        Returns
        ----------
        psd
            Created PSD for the model noise
        freq
            Frequency vector of the created PSD
        """
        f2 = self.fs/2
        df = self.fs/(self.noise_type.duration*self.fs)
        freq = np.linspace(df, f2, int(self.noise_type.duration*self.fs/2))
        
        psd = 1/freq**self.noise_type.alpha
        if np.mean(psd) != 0:
            psd = self.noise_type.c*psd/(np.mean(psd)) + self.noise_type.wn
        else:
            psd = self.noise_type.wn
        
        return psd/1000, freq 


    def _add_point_frequency(self, psd: np.ndarray, freq: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Adds spike at given frequency and amplitude (point_value) to given PSD. PSD and freq are first interpolated to given duration.
        Parameters
        ----------
        psd
            PSD to add the point frequency
        freq
            Frequency vector of the PSD
        Returns
        ----------
        psd
            PSD with the added power at the point frequency
        freq
            Frequency vector of the PSD
        """

        #Check that the given frequency is in range
        point_frequency = self.noise_type.point_freq*freq[-1]
        if point_frequency>freq[-1] or point_frequency<freq[0]:
            print(f'Given point frequency {point_frequency} out of range [{freq[0]}, {freq[-1]}]')
            point_frequency = np.random.uniform(freq[0], freq[-1])
            print(f'A new point frequency {point_frequency} was randomly selected.')
            

        new_freq = np.linspace(freq[0], freq[-1], int(self.noise_type.duration*freq[-1]), endpoint=True)
        psd, freq = interpolate_(freq, new_freq, psd)
        freq_idx = np.argmin(np.abs(freq - point_frequency))
        psd[freq_idx] = self.noise_type.point_value
        
        return psd, freq

    def _add_artifact(self, arr: np.ndarray) -> np.ndarray:
        """
        Add artifact with amplitude to given location (in samples) of the signal. 
        Both signal and artifact are first normalized between 0 and 1.

        Parameters
        ----------
        arr
            Signal where the artifact is added

        Returns
        ----------
        arr
            Signal with the added artifact
        label   
            Labels for showing where the artifact is added
        """   
        if self.artifact_length*self.fs > len(arr):
            self.artifact_length = len(arr)/self.fs
        # ma, bw = self._load_noise()
        ma, bw = self.__artifact_arrs 
        start = int(self.artifact_start*(len(arr)-(self.artifact_length*self.fs)))
        if self.artifact_type == 'ma':
            artifact = ma[int(self.artifact_idx*(len(ma)-(self.artifact_length*self.fs))):int(self.artifact_idx*(len(ma)-(self.artifact_length*self.fs)))+int(self.artifact_length*self.fs)]
        if self.artifact_type == 'bw':
            artifact = bw[int(self.artifact_idx*(len(bw)-(self.artifact_length*self.fs))):int(self.artifact_idx*(len(bw)-(self.artifact_length*self.fs)))+int(self.artifact_length*self.fs)]       
        stop = start + len(artifact) 
        
        arr = zero_mean(arr)   
        artifact = min_max_normalize(artifact)
        artifact = zero_mean(artifact)*self.artifact_amp     
        arr[start:stop] = arr[start:stop] + artifact #-->add artifact    
        
        label = np.zeros(len(arr)) 
        label[start:stop] = self.artifact_amp
        label = (self.artifact_type + '_art, ' + str(self.artifact_amp), label)

        return arr, label
    
    def _load_noise(self):
        """
        Downloads muscle artifact or baseline wander noise from MIT-BIH Noise Stress Test Database.

        Returns
        ----------
        ma
            Muscle artifact
        bw
            Baseline wander
        """   
        # Load data
        baseline_wander = rdsamp('bw', pn_dir='nstdb')
        muscle_artifact = rdsamp('ma', pn_dir='nstdb')

        # Concatenate two channels to make one longer recording
        ma = np.concatenate((muscle_artifact[0][:,0], muscle_artifact[0][:,1]))
        bw = np.concatenate((baseline_wander[0][:,0], baseline_wander[0][:,1]))

        # Resample noise to wanted Hz
        ma = resample_poly(ma, up=int(self.fs), down=muscle_artifact[1]['fs'])
        bw = resample_poly(bw, up=int(self.fs), down=baseline_wander[1]['fs'])

        return ma, bw


    def _concatenate_two_time_realisations(self, arr1: np.ndarray, arr2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculates tapered part between two time realisations and appends the part to the list

        Parameters
        ----------
        arr1
            The first time realisation
        arr2
            The second time realisation

        Returns
        ----------
        concatenated
            Concatenated time realisation
        label_idx
            Index of the ending point of the first signal
        """   
        w1 = np.linspace(1/self.__tap_len, 1, self.__tap_len)
        w2 = w1[::-1]
        s1,s2a = np.split(arr1,[len(arr1)-self.__tap_len]) 
        s2b,s3 = np.split(arr2,[self.__tap_len]) 
        s2 = s2a*w2 + s2b*w1 #-->weighted average of overlapping arrays

        concatenated = np.concatenate([s1, s2, s3])
        label_idx = int(len(arr1) - (self.__tap_len/2))

        
        return concatenated, label_idx

    def _concatenate_time_realisations(self, time_realisations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Ending and beginning of time realisations are tapered (with given length) to handle discontinuity and non-zero mean.

        Parameters
        ----------
        time_realisations
            Array of the signals to concatenate

        Returns
        ----------
        time_realisation
            The concatenated signals
        label_indeces
            Indeces for starting and ending points of the original signals
        """
        time_realisation = time_realisations[0]
        label_indeces = []
        for real in time_realisations[1:]:
            time_realisation, label_idx = self._concatenate_two_time_realisations(time_realisation, real)
            label_indeces.append(label_idx)
        label_indeces.append(int(len(time_realisation)))

        return time_realisation, label_indeces
        

    def _get_labels(self, noise: list[str], label_indeces: np.ndarray, amplitude: float) -> np.ndarray:
        """
        Returns noise labels for the tapered time realisations.

        Parameters
        ----------
        noise
            Noise type as string
        label_indeces
            Indeces for starting and ending points of the original signals
        amplitude
            Amplitude of noise
        Returns
        ----------
        labels
            List of tuples: (string label, array with noise amplitude)

        """
        labels = [(noise[0], np.ones(label_indeces[0])*amplitude[0])]
        for i, [nt, l_idx, amp] in enumerate(zip(noise[1:], label_indeces, amplitude[1:])):
            labels.append((nt, np.ones(label_indeces[i+1]-l_idx)* amp))

        return labels

    def combine_signal_noise(self, synt_signal: np.ndarray, noise_signal: np.ndarray, peak_labels: np.ndarray, label: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Combines signal with noise and fixes the lengths.
        
        Parameters
        ----------
        synt_signal
            Pure (synthetic) signal
        noise_signal
            Noise as a time realisation
        peak_labels
            Peak labels of the pure signal
        label
            Label of noise
        Returns
        ----------
        signal_
            Noisy signal
        peak_labels
            Peak labels of the original signal
        label
            Label of noise
        """

        length = np.min([len(noise_signal), len(synt_signal)])
        signal_ = noise_signal[0:length] + synt_signal[0:length]
        peak_labels = peak_labels[0:length]
        label[-2] = (label[-2][0], np.ones(len(label[-2][1])-(len(np.concatenate(list(zip(*label[:-1]))[1]))-len(signal_)))*label[-2][1][0])
        label[-1] = label[-1][0:len(signal_)]

        return signal_, peak_labels, label

