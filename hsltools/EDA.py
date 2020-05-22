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
EDA is a module consisting of features designed to specifically analyze electrodermal activity.

"""

def mean_firstdiff(signal):
	"""
    Returns the mean first difference of the EDA signal. 

    The first difference in perturbed epochs is expected to be higher in magnitude than those at 
    rest due to the orienting responses in active states [1]. Feature extraction from the first and 
    second differences of the EDA signal can be used to study the manifestation of psychological 
    stress [2].

	[1] 
	Blain et al., 2008 S. Blain, A. Mihailidis, T. Chau Assessing the potential of electrodermal 
	activity as an alternative access pathway Medical Engineering & Physics, 30 (4) (2008), 
	pp. 498-505
	[2]
	Liu Y (2018) Du S (2018) Psychological stress level detection based on electrodermal 
	activity. Behav Brain Res 341:50–53. https://doi.org/10.1016/j.bbr.2017.12.021


    Parameters
    ----------
    signal : array-like
        Array containing numbers whose mean first difference is desired. 
    
    Returns
    -------
    ndarray
        Returns the mean value of the signal's first differences. 

    """

    return np.mean(np.abs(np.diff(signal, n=1)))

def std_firstdiff(signal):
	"""
    Returns the standard deviation of the EDA signal's first difference (see mean_firstdiff). 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose standard deviation of the first differences is desired. 
    
    Returns
    -------
    ndarray
        Returns the standard deviation of the signal's first differences. 

    """
    return np.std(np.abs(np.diff(signal, n=1)))

def mean_seconddiff(signal):
	"""
    Returns the mean second difference of the EDA signal (see mean_firstdiff).

	Parameters
    ----------
    signal : array-like
        Array containing numbers whose mean second difference is desired. 
    
    Returns
    -------
    ndarray
        Returns the mean value of the signal's second differences.

    """
    return np.mean(np.abs(np.diff(signal, n=2)))

def std_seconddiff(signal):
    """
    Returns the standard deviation of the EDA signal's second difference (see mean_firstdiff).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose standard deviation of the second differences is desired. 
    
    Returns
    -------
    ndarray
        Returns the standard deviation of the signal's second differences. 

    """
    return np.std(np.abs(np.diff(signal, n=2)))

def find_peaks(signal):
    """
    Returns the peaks of the EDA signal and returns the minima between the peaks.  

    A median filter (with a window size of 31) is first applied to reduce noise. Peaks of the orienting 
    response are calculated using peak prominence and distance between responses as 
    constraints to further avoid noise. The minima between peaks is also calculated.

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose peaks are desired.  

    Returns
    -------
    (ndarray, ndarray)
        Returns the indices of peaks in the signal.
        Returns minima between the peaks.
    
    """
    #normalize function
    norm_signal = medfilt((signal-np.mean(signal))/np.std(signal), 31)
    #find peaks
    peaks, _ = sig.find_peaks(norm_signal, distance = len(signal)/12, prominence = 0.25)
    if len(peaks) == 0:
        peaks, _ = sig.find_peaks(norm_signal, distance = len(signal)/14, prominence = 0.1)
    if len(peaks) == 0:
        peaks, _ = sig.find_peaks(norm_signal, distance = len(signal)/14)
    min_0 = np.argmin(signal[0:peaks[0]])
    min_peaks = [min_0]
    for i in range(1,len(peaks)):
        min_peaks.append((peaks[i-1]+min_i))
    return peaks, min_peaks

def orienting_features(signal):
    """
    Returns an array containing features of orienting response of the EDA signal as calculated in 
    Healy and Picard (number of onsets, sum of onsets, sum of peaks, difference between sum of 
    onsets and peaks, sum of area under response curve, sum of orienting response duration, 
    mean of orienting response duration, mean amplitude of peaks, total amplitude of peaks). 

    An orienting response is a multisystem reaction evoked by exposure to a novel stimulus [1]. 
    One part of the physiological response is the ionic filling of sweat glands in the skin due to the 
    activation of the sympathetic nervous system [2]. This leads to a sudden rise in skin 
    conductance, as well as an automatic shift of attentional resources to the stimulus [3].  

    [1] 
    Friedman D, Goldman R, Stern Y, Brown TR. The brain's orienting response: 
    An event-related functional magnetic resonance imaging investigation. Hum Brain Mapp. 
    2009;30(4):1144–1154. doi:10.1002/hbm.20587
    [2] 
    Healey, J. A., & Picard, R. W. (January 01, 2005). Detecting stress during real-world 
    driving tasks using physiological sensors. Ieee Transactions on Intelligent Transportation 
    Systems, 6, 2, 156-166.
    [3]
    Ranganath C, Rainer G. Neural mechanisms for detecting and remembering novel 
    events. Nat Rev Neurosci. 2003;4(3):193–202. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose orienting response features are desired.

    Returns
    -------
    array-like
        Returns an array of orienting response features of the signal [num_onsets, sum_onsets, sum_peaks, 
        sum_difference, sum_areas, sum_od, mean_od, mean_amp, total_amp].

        num_onsets - number of onsets/minima (see find_peaks) 
        sum_onsets - sum of onsets/minima (see find_peaks)
        sum_peaks - sum of peaks (see find_peaks)
        sum_difference - difference between sum_peaks and sum_onsets
        sum_areas - estimated areas under the orienting response curves
        sum_od - sum of the orienting response durations
        mean_od - mean of the orienting response durations
        mean_amp - mean amplitude of the peaks calculated as (Σ(Xpeak+Xonset)/2)/num_peaks (see find_peaks)
        total_amp - the total amplitude of the peaks Σ(Xpeak+Xonset)/2 (see find_peaks)
        
    """
    peaks, onsets = find_peaks(signal)
    num_onsets = len(onsets)
    sum_peaks = sum(signal[peaks])
    sum_onsets = sum(signal[onsets])
    sum_difference = sum_peaks - sum_onsets
    
    orienting_duration = []
    for i in range(len(peaks)):
        orienting_duration.append(peaks[i]-onsets[i])
    mean_od = np.mean(orienting_duration)
    sum_od = sum(orienting_duration)
	    
    areas = []
    for i in range(len(peaks)):
        areas.append(0.5*orienting_duration[i]*peaks[i])
	    
    sum_areas = sum(areas)
	    
    mean_amplitude = []
    for i in range(len(peaks)):
        mean_amplitude.append((peaks[i]+onsets[i])/2)
	    
    mean_amp = np.mean(mean_amplitude)
    total_amp = sum(mean_amplitude)
    return [num_onsets, sum_onsets, sum_peaks, sum_difference, sum_areas, sum_od, mean_od, mean_amp, total_amp]

def eda_passbands(signal):
    """
    Returns an array containing the very low frequency, low frequency, and high frequency 
    passbands of the EDA signal. 

    EDA is strongly correlated to sweat production, so its power spectral density (PSD) can be used 
    to measure arousal of the sympathetic nervous system in isolation [1]. Low frequency (LF) 
    changes derive from the influence of the cardiac sympathetic nerves [3], so the same LF 
    passband was used for IBI and EDA. The PSD was approximated using a periodogram, and the 
    integral for each passband was calculated using composite trapezoidal integration.

    [1] 
    Critchley, H. D. (2002). Review: Electrodermal Responses: What Happens in the
    Brain. The Neuroscientist, 8(2), 132–142. https://doi.org/10.1177/107385840200800209
    [2]
    Posada-Quintero, Hugo & Florian, John & Orjuela-Cañón, Alvaro & Corrales, Tomás & 
    Charleston-Villalobos, Sonia & Chon, Kaye. (2016). Power Spectral Density Analysis of 
    Electrodermal Activity for Sympathetic Function Assessment. Annals of Biomedical 
    Engineering. 44. 10.1007/s10439-016-1606-6.
    [3]
    H. F. Posada-Quintero and K. H. Chon, "Frequency-domain electrodermal activity index
    of sympathetic function," 2016 IEEE-EMBS International Conference on Biomedical and 
    Health Informatics (BHI), Las Vegas, NV, 2016, pp. 497-500.

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose frequency passbands are desired.

    Returns
    -------
    array-like
        Returns array containing frequency passbands of the signal [vlf_integral, lf_integral, hf_integral]
        
        vlf_integral - very low frequency, 0.001-0.045 Hz
        lf_integral - low frequency, 0.045-0.15 Hz
        hf_integral - high frequency, 0.15-0.25 Hz

    """
    fs,pxx = scipy.signal.periodogram(signal, nfft = 1000, scaling = 'density', detrend = 'linear')

    vlfband = (fs > 0.001)*(fs < 0.045)
    lfband = (fs > 0.045)*(fs < 0.15)
    hfband = (fs > 0.15) * (fs < 0.25)

    vlf_integral = np.trapz(pxx[vlfband],fs[vlfband])
    lf_integral = np.trapz(pxx[lfband],fs[lfband])
    hf_integral = np.trapz(pxx[hfband],fs[hfband])
	return [vlf_integral,lf_integral,hf_integral]

def eda_lfhf(signal):
    """
    Returns the ratio of low frequency passbands to high frequency passbands of the EDA signal 
    (see eda_passbands).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose ratio between low and high frequency passbands is desired. 

    Returns
    -------
    float
        Returns the ratio of low frequency passbands to high frequency passbands of the signal.

    """
    return eda_passbands(signal)[1]/eda_passbands(signal)[2]

def eda_all_feat(signal): #returns all eda features in data frame
    """
    Returns all of the EDA features of the EDA signal in the form of a labeled data frame (mean_firstdiff, std_firstdiff, 
    mean_seconddiff, std_seconddiff, eda_lfhf, orienting_features).  

    Parameters
    ----------
    signal : array-like 
        Array containing numbers whose EDA features are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of all the EDA features with the column headings [eda_mean1diff, 
        eda_std1diff, eda_mean2diff, eda_std2diff, eda_lf/hf, eda_num_onsets, eda_sum_onsets, 
        eda_sum_peaks, eda_sum_difference, eda_sum_areas, eda_sum_od, eda_mean_od, 
        eda_mean_amp, eda_total_amp]

        eda_mean1diff - see mean_firstdiff
        eda_std1diff - see std_firstdiff
        eda_mean2diff - see mean_seconddiff
        eda_std2diff - see std_seconddiff
        eda_lf/hf - see eda_lfhf
        eda_num_onsets, eda_sum_onsets, ..., eda_total_amp - see orienting features

    """
    functions = [mean_firstdiff, std_firstdiff, mean_seconddiff, std_seconddiff, eda_lfhf]
    measure_names = ['eda_mean1diff', 'eda_std1diff', 'eda_mean2diff', 'eda_std2diff', 'eda_lf/hf']
    features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
    fdf = pd.DataFrame(columns = measure_names, data = features.T)
    
    orient_features = np.asarray(orienting_features(signal)).reshape(-1,1)
    orient_featurenames = ['eda_num_onsets', 'eda_sum_onsets', 'eda_sum_peaks', 'eda_sum_difference',\
                           'eda_sum_areas', 'eda_sum_od', 'eda_mean_od', 'eda_mean_amp', 'eda_total_amp']
    ofdf = pd.DataFrame(columns = orient_featurenames, data = orient_features.T)
	    
    return fdf.join(ofdf)
