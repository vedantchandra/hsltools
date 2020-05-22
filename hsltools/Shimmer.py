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

from hsltools.basics import signal_statistics

"""
Shimmer is a module consisting of features designed to specifically analyze Shimmer sensors.

"""

def spectrum_statistics(signal):
    """
    Returns an array containing basic statistics of the signal (peak, peak magnitude, integral, 
    energy, shannon).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose spectrum statistics are desired.

    Returns
    -------
    array-like
        Returns array of signal statistics [peak, peakmag, integral, energy, shannon].

        peak - peak frequency of the signal
        peakmag - peak power of the signal 
        integral - integral of signal using the composite trapezoidal rule
        energy - spectral energy of the signal 
        shannon - The Shannon entropy, or self-information is a quantity that identifies the amount of information 
        associated with an event using the probabilities of the event. There is an inverse relationship 
        between probability and information, as low-probability events carry more information [3],[4].

        [3] https://arxiv.org/pdf/1405.2061.pdf
		[4] https://towardsdatascience.com/the-intuition-behind-shannons-entropy-e74820fe9800


    """
	fs,pxx = scipy.signal.periodogram(signal, fs = 50, nfft = 1000, scaling = 'density', detrend = 'constant')

    peak = fs[np.argmax(pxx)]
	peakmag = np.max(pxx)
	integral = np.trapz(pxx,fs)
	energy = np.dot(pxx,pxx)
	shannon = np.sum(pxx*np.log(1/pxx))

	 	# Add wavelet analysis

	return [peak, peakmag, integral, energy, shannon]

def bodyshimmer_features(signal):
	"""
    Returns signal_statistics and spectrum_statistics of the body signal in the form of a labeled data frame.

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic and spectrum statistics are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of basic and spectrum statistics with column headings [bodyshim_mean, 
        bodyshim_std, bodyshim_skewness, bodyshim_kurtosis, bodyshim_maximum, bodyshim_minimum, 
        bodyshim_iqr, bodyshim_variation, bodyshim_entropy, bodyshim_dfa, bodyshim_peakfreq, 
        bodyshim_peakpower, bodyshim_powerint, bodyshim_specenergy, bodyshim_shannon]
		
		bodyshim_mean, bodyshim_std, ..., bodyshim_skewness - see basics.signal_statistics
		bodyshim_peakfreq, bodyshim_peakpower, ..., bodyshim_shannon - see spectrum_statistics

    """
    shim_stats = signal_statistics(signal).reshape(-1,1)
    stat_names = ['bodyshim_mean', 'bodyshim_std', 'bodyshim_skewness', 'bodyshim_kurtosis', 'bodyshim_maximum', 'bodyshim_minimum', 'bodyshim_iqr', 'bodyshim_variation', 'bodyshim_entropy', 'bodyshim_dfa']
    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
    
    spec_stats = np.asarray(spectrum_statistics(signal)).reshape(-1,1)
    specstat_names = ['bodyshim_peakfreq','bodyshim_peakpower','bodyshim_powerint','bodyshim_specenergy', 'bodyshim_shannon']
    spec_fdf = pd.DataFrame(columns = specstat_names, data = spec_stats.T)
    
    return fdf.join(spec_fdf)

def headshimmer_features(signal):
	"""
    Returns signal_statistics and spectrum_statistics of the head signal in the form of a labeled data frame.

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic and spectrum statistics are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of basic and spectrum statistics with column headings [headshim_mean, 
        headshim_std, headshim_skewness, headshim_kurtosis, headshim_maximum, headshim_minimum, 
        headshim_iqr, headshim_variation, headshim_entropy, headshim_dfa, headshim_peakfreq, 
        headshim_peakpower, headshim_powerint, headshim_specenergy, headshim_shannon]
		
		headshim_mean, headshim_std, ..., headshim_skewness - see basics.signal_statistics
		headshim_peakfreq, headshim_peakpower, ..., headshim_shannon - see spectrum_statistics

    """
    shim_stats = signal_statistics(signal).reshape(-1,1)
    stat_names = ['headshim_mean', 'headshim_std', 'headshim_skewness', 'headshim_kurtosis', 'headshim_maximum', 'headshim_minimum', 'headshim_iqr', 'headshim_variation', 'headshim_entropy', 'headshim_dfa']
    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
    
    spec_stats = np.asarray(spectrum_statistics(signal)).reshape(-1,1)
    specstat_names = ['headshim_peakfreq','headshim_peakpower','headshim_powerint','headshim_specenergy', 'headshim_shannon']
    spec_fdf = pd.DataFrame(columns = specstat_names, data = spec_stats.T)
    
    return fdf.join(spec_fdf)
  
def hr_features(signal):
    """
    Returns the signal_statistics of the heart rate signal in the form of a labeled data frame. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic statistics are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of basic statistics with column headings [hr_mean, hr_std, hr_skewness, 
        hr_kurtosis, hr_maximum, hr_minimum, hr_iqr, hr_variation, hr_entropy, hr_dfa].

    """
    shim_stats = signal_statistics(signal).reshape(-1,1)
    stat_names = ['hr_mean', 'hr_std', 'hr_skewness', 'hr_kurtosis', 'hr_maximum', 'hr_minimum', 'hr_iqr', 'hr_variation', 'hr_entropy', 'hr_dfa']
    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
    return fdf	

def temp_features(signal):
    """
    Returns the signal_statistics of the temperature signal in the form of a labeled data frame. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose basic statistics are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of basic statistics with column headings [temp_mean, temp_std, temp_skewness, 
        temp_kurtosis, temp_maximum, temp_minimum, temp_iqr, temp_variation, temp_entropy, temp_dfa].

    """
    shim_stats = signal_statistics(signal).reshape(-1,1)
    stat_names = ['temp_mean', 'temp_std', 'temp_skewness', 'temp_kurtosis', 'temp_maximum', 'temp_minimum', 'temp_iqr', 'temp_variation', 'temp_entropy', 'temp_dfa']
    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
    return fdf
