import pickle as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from dfa import dfa
import scipy.signal as sig
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d
from scipy.integrate import romb
plt.rcParams.update({'font.size': 22})
from hsl_functions import *
import glob
from scipy.signal import medfilt,butter
import corner
import scipy

"""
Basics is a module consisting of basic statistic functions applicable to all signal types. 

"""

def signal_statistics(signal):
    """
    Returns an array containing basic statistics of the signal (mean, standard deviation, skewness, 
    kurtosis, maximum, minimum, interquartile range, variation, entropy, DFA).
    
    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic statistics are desired.     
    
    Returns
    -------
    ndarray
        Returns array of basic statistics [mean, std, skewness, kurtosis, maximum, minimum, iqr, 
        variation, entropy, dfa_exp].

        mean - mean of the signal
        std - standard deviation of the signal
        skewness - imbalance and asymmetry from the mean computed as the Fisher-Pearson coefficient of skewness
        kurtosis - sharpness of the peak of a frequency-distribution curve
        maximum - maximum value of the signal 
        minimum - minimum value of the signal 
        iqr - interquartile range, statistical dispersion as the diffrence betwen the upper and lower quartiles 
        variation - ratio of the biased standard deviation to the mean
        entropy - measure of uncertainty using Shannon entropy
        dfa_exp - discriminant function analysis, detrended fluctuation analysis, deterministic finite automaton??
    
    """
   	mean = np.mean(signal)
   	std = np.std(signal)
   	skewness = stats.skew(signal)
   	kurtosis = stats.kurtosis(signal) #sharpness of the peak of a freq-distrib curve
   	maximum = np.max(signal)
   	minimum = np.min(signal)
   	iqr = stats.iqr(signal)
   	variation = stats.variation(signal)
   	entropy = stats.entropy(np.abs(signal))
   	dfa_exp = 0  #dfa(signal) 
   	return np.asarray([mean, std, skewness, kurtosis, maximum, minimum, iqr, variation, entropy, dfa_exp])


def ss_dataframe(signal):
    """
    Returns signal_statistics of the signal in the form of a labeled data frame. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic statistics are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of basic statistics with column headings [mean, std, skewness, kurtosis, 
        maximum, minimum, iqr, variation, entropy, dfa_exp].
    
    """
	stat_names = ['mean', 'std', 'skewness', 'kurtosis', 'maximum', 'minimum', 'iqr', 'variation', 'entropy', 'dfa_exp']
	fdf = pd.DataFrame(columns = stat_names, data = basic_stats.T)
	return fdf
