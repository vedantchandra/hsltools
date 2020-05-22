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
IBI is a module consisting of features designed to specifically analyze interbeat interval.

"""

def rmssd(signal):
	"""
    Returns the Root Mean Square of the Successive Differences (RMSSD) of the IBI signal.  

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose RMSSD is desired.

    Returns
    -------
    ndarray
        Returns the RMSSD of the signal. 
   
    """
    return np.sqrt(np.mean(np.square(np.diff(signal))))

def ibi_passbands(signal):
	"""
    Returns an array containing the very low frequency, low frequency, and high frequency 
    passbands of the IBI signal (see eda.eda_passbands). 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose frequency passbands are desired.

    Returns
    -------
    array-like
        Returns array containing frequency passbands [vlf_integral, lf_integral, hf_integral]
        
        vlf_integral - very low frequency, 0.0033-0.04 Hz
        lf_integral - low frequency, 0.04-0.15 Hz
        hf_integral - high frequency, 0.15-0.4 Hz

    """
    fs,pxx = scipy.signal.periodogram(signal, nfft = 1000, scaling = 'density', detrend = 'constant')

    vlfband = (fs > 0.0033)*(fs < 0.04)
    lfband = (fs > 0.04)*(fs < 0.15)
    hfband = (fs > 0.15) * (fs < 0.4)

    vlf_integral = np.trapz(pxx[vlfband],fs[vlfband])
    lf_integral = np.trapz(pxx[lfband],fs[lfband])
    hf_integral = np.trapz(pxx[hfband],fs[hfband])

    return [vlf_integral,lf_integral,hf_integral]

def ibi_lfhf(signal):
    """
    Returns the ratio of low frequency passbands to high frequency passbands of the IBI signal 
    (see IBI_passbands).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose ratio between low and high frequency passbands is desired. 

    Returns
    -------
    float
        Returns the ratio of low frequency passbands to high frequency passbands of the signal.

    """
    return ibi_passbands(signal)[1]/ibi_passbands(signal)[2]

def ibi_all_features(signal): #returns all ibi features in data frame
   	"""
    Returns all of the IBI features of the IBI signal in the form of a labeled data frame (rmssd, ibi_lf/hf).  

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose IBI features are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of all the EDA features with the column headings [ibi_rmssd,
        ibi_lf/hf]

        ibi_rmssd - see rmssd
        ibi_lf/hf - see ibi_lfhf

    """
   	functions = [rmssd, ibi_lfhf]
   	measure_names = ['ibi_rmssd', 'ibi_lf/hf']
   	features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
   	fdf = pd.DataFrame(columns = measure_names, data = features.T)
   	return fdf