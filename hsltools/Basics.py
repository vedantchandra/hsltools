import pickle as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from .dfa import dfa
import scipy.signal as sig
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d
from scipy.integrate import romb
plt.rcParams.update({'font.size': 22})
import glob
from scipy.signal import medfilt,butter
import scipy

"""
Basics is a module consisting of basic statistic functions applicable to all signal types. 

"""

def signal_statistics(signal):
    """
    Returns an array containing basic statistics of the signal (mean, standard deviation, skewness, 
    kurtosis, maximum, minimum, interquartile range, variation, entropy, scaled correlation time).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic statistics are desired.     

    Returns
    -------
    ndarray
        Returns array of basic statistics [mean, std, skewness, kurtosis, maximum, minimum, iqr, 
        variation, entropy, corrtime].

        mean - mean of the signal
        std - standard deviation of the signal
        skewness - imbalance and asymmetry from the mean computed as the Fisher-Pearson coefficient of skewness
        kurtosis - sharpness of the peak of a frequency-distribution curve
        maximum - maximum value of the signal 
        minimum - minimum value of the signal 
        iqr - interquartile range, statistical dispersion as the diffrence betwen the upper and lower quartiles 
        variation - ratio of the biased standard deviation to the mean
        entropy - measure of uncertainty using Shannon entropy
        corrtime - see scaled_correlation_time

    """
    mean = np.mean(signal)
    std = np.std(signal)
    skewness = stats.skew(signal)
    kurtosis = stats.kurtosis(signal) #sharpness of the peak of a freq-distrib curve
    maximum = np.max(signal)
    minimum = np.min(signal)
    iqr = stats.iqr(signal)
    entropy = stats.entropy(np.abs(signal))
    corrtime = scaled_correlation_time(signal,signal) 
    return np.asarray([mean, std, skewness, kurtosis, maximum, minimum, iqr, entropy, corrtime])


def basic_all_features(signal, name):
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
        maximum, minimum, iqr, variation, entropy, corrtime] (see signal_statistics).

    """
    basic_stats = signal_statistics(signal).reshape(-1, 1)
    stat_names = ['mean', 'std', 'skewness', 'kurtosis', 'maximum', 'minimum', 'iqr', 'entropy', 'corrtime']
    stat_names = [name + '_' + stat_names[ii] for ii in range(len(stat_names))]
    fdf = pd.DataFrame(columns = stat_names, data = basic_stats.T)
    return fdf

def scaled_correlation_time(signal1, signal2):
    """
    Returns the scaled correlation time of the signal.   

    Parameters
    ----------
    signal1 : array-like
        Array containing numbers whose correlation time is desired.
    signal2 : array-like
        Array containing numbers whose correlation time is desired.

    Returns
    -------
    ndarray of ints
        Returns the scaled correlation time of the signals. 
   
    """

    signal1 = (signal1 - np.mean(signal1))/np.std(signal1)
    signal2 = (signal2 - np.mean(signal2))/np.std(signal2)
    acorr = np.correlate(signal1, signal2, mode='full')
    acorr = acorr[(acorr.size // 2 ):] / np.max(acorr)
    tau = np.argmax([acorr < 1/np.exp(1)])
    return tau / len(acorr)