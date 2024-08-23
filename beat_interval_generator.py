import numpy as np
from dataclasses import dataclass
import random
from utils import default_field
import warnings


@dataclass
class BeatIntervalGenerator():

    n: int = 30
    duration: float = None
    beat_intervals: list = None
    #mean beat interval
    mu: float = 1.0
    mu_rng: list = default_field([0.4, 1.2])
    a: float = 1.2
    std: float = 0.5
    std_rng: list = default_field([0.45, 0.55])
    b: float = 0.075
    bc: float = 0.1
    bf: float = 1/3.6
    #mean beat interval after step change
    mu_new: float = 0.75
    mu_new_rng: list = default_field([0.3, 2])
    step_i: float = 0.5
    step_i_rng: list = default_field([0, 1])
    step_f: float = 1
    step_f_rng: list = default_field([1, 10])
    step: bool = False
    step_prob: float = 0.5
    step_min: float = 0.3
    step_max: float = 2

    def generate(self):
        """
        Generates beat intervals for signal generator. 
        Returns
        ----------
        intervals
            Generated beat intervals.
        """
        
        if self.duration:
            self.n = int(8*self.duration/np.min([self.mu, self.mu_new]))

        if self.beat_intervals is None:
            breathing_gen = lambda bf, bc, br_prev: bc*np.sin(2*np.pi*br_prev*bf)
            y = self._stochastic(self.n, self.a, self.std, self.b)    
        
            z = np.zeros(self.n)
            if self.step:       
                z = self._gen_hr_step()
            intervals = np.zeros(self.n)
            for i in np.arange(self.n):  
                br_prev = np.sum(intervals)
                intervals[i] = self.mu*(1+z[i]) + breathing_gen(self.bf, self.bc, br_prev) + y[i]   
                # scales the effect of breathing for rr intervals < 0.35 to avoid negative values 
                if intervals[i] < 0.35:
                    intervals[i] = self.mu*(1+z[i]) * (1 + breathing_gen(self.bf, self.bc, br_prev))
                
                if self.duration:
                    if np.sum(intervals) >= self.duration:
                        intervals = intervals[:i+1]
                        self.n = len(intervals)
                        break

        else:
            intervals = np.array(self.beat_intervals)



        return intervals
    
    def _gen_hr_step(self) -> np.ndarray:
        """
        Generates the change in the heart rate.
        
        Returns
        ----------
        z
            Beat intervals with HR change
        
        """

        tau_constant = 3
        # distance = self.step_i*self.n
        x = np.linspace(1, self.n, int(self.n))
        distance = int(self.step_i*self.duration/self.mu)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            z = ((self.mu_new/self.mu)-1)/(1 + np.exp(-(x-distance)/self.step_f*tau_constant))
        
        return z


    def _stochastic(self, n: int, a: float, std: float, b: float) -> np.ndarray:
        """
        Generates long-term correlation. Source: Citation J. W. Kantelhardt et al 2003 EPL 62 147
        
        Parameters
        ----------
        n
            Number of beat intervals
        a
            Shape parameter of Pareto distribution
        std
            Standard deviation
        b
            Coefficient
        Returns
        ----------
        Beat intervals with transient correlations

        """
        k, x = self._rand_sequence(n, a, std)    
        y, y_tmp = np.zeros(n), np.zeros(n)
        for i in np.arange(n):       
            #eq. 3          
            if i - k[i] > 0:
                avg = np.mean(np.square(y_tmp[i-k[i]:i])) 
                y_tmp[i] = x[i]*np.sqrt(1+b*avg)            
            else: 
                y_tmp[i] = 0   
                
            m = np.zeros(i)
            m[(k[:i] + np.arange(i) - i) > 0] = 1
            y[i] = 0.05*np.sum(y_tmp[:i]*m)
        return y
    
    def _rand_sequence(self, n: int, a: float, std: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates sequences for funtion _stochastic().

        Parameters
        ----------
        n
            Number of beat intervals
        a
            Shape parameter of Pareto distribution
        std
            Standard deviation

        Returns
        ----------
        s: random stream drawn from Pareto(6, a) distribution
        x: random stream drawn from normal distribution
        """
        
        s = (np.random.pareto(a, n) + 1)*6
        x = np.random.randn(n)*std  

        return np.array(s, dtype=int), np.array(x)

    def randomize(self):
        """
        Randomizes the parameters of beat interval generator.
        """
        
        x = lambda l1, l2: np.random.uniform(l1, l2)
        
        self.mu = x(self.mu_rng[0], self.mu_rng[1])
        self.std = x(self.std_rng[0], self.std_rng[1])
        
        self.step_size = x(self.mu_new_rng[0], self.mu_new_rng[1])
        self.step_i = x(self.step_i_rng[0], self.step_i_rng[1])
        self.step_f = x(self.step_f_rng[0], self.step_f_rng[1])
        self.step = random.choices([True, False], weights=[self.step_prob, 1-self.step_prob])[0]
        
