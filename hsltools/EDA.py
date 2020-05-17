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

#EDA functions include: passbands, lfhf, features 

def mean_firstdiff(signal):
	"""
    In a person at rest, the EDA signal has a tendency to decay over time, leading to smaller, often 
    negative first differences. Orienting responses are more common in active states, and are 
    characterized by larger changes in the EDA signal. Thus, the first differences in perturbed 
    epochs would be expected to be higher in magnitude[1]. Feature extraction from the first and 
    second differences of the EDA signal have also been used to study the manifestation of 
    psychological[2] stress.

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
        Returns the mean value of the first differences. 

    """

    return np.mean(np.abs(np.diff(signal, n=1)))

def std_firstdiff(signal):
	"""
	
    description

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose standard deviation of the first differences is desired. 
    
    Returns
    -------
    ndarray
        Returns the standard deviation of the first differences. 

    """
    return np.std(np.abs(np.diff(signal, n=1)))

def mean_seconddiff(signal):
	"""
    description

	Parameters
    ----------
    signal : array-like
        Array containing numbers whose mean second difference is desired. 
    
    Returns
    -------
    ndarray
        Returns a new array containing the mean values of the second differences. 

    """
    return np.mean(np.abs(np.diff(signal, n=2)))

def std_seconddiff(signal):
    """
    Description of module level function. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose standard deviation of the second differences is desired. 
    
    Returns
    -------
    ndarray
        Returns the standard deviation of the second differences. 

    """
    return np.std(np.abs(np.diff(signal, n=2)))

def find_peaks(signal):
    """
    Description of module level function. 

    Parameters
    ----------
    signal : array-like
        The first parameter.

    Returns
    -------
    placeholder
        peaks, min_peaks
    
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
        min_i = np.argmin(signal[peaks[i-1]:peaks[i]])
        min_peaks.append((peaks[i-1]+min_i))
    return peaks, min_peaks

def orienting_features(signal):
    """
    The bulk of the features extracted from the EDA signal were predicated on the idea of orienting 
    responses. An orienting response is a multisystem reaction evoked by exposure to a novel 
    stimulus [1]. One part of the physiological response is the ionic filling of sweat glands in the skin 
    due to the activation of the sympathetic nervous system [2]. This leads to a sudden rise in skin 
    conductance, as well as an automatic shift of attentional resources to the stimulus [3]. To find 
    orienting responses within the EDA signal, a median filter (with a window size of 31) was first 
    applied to reduce noise. Using peak prominence and distance between responses as 
    constraints to further avoid noise, the peaks of the orienting responses were calculated, and the 
    minima between peaks was used as the onset of the orienting reflex. As calculated in Healy and 
    Picard, key features included the sum of the startle magnitudes, the sum of response durations, 
    and the estimated areas under the response curves. Additional features consist of the 
    respective sums of the peaks and onsets, the mean response duration, and the mean and total 
    amplitude of the peaks which were calculated as Σ(Xpeak + Xonset)/2 and 
    (Σ(Xpeak+Xonset)/2)/num_peaks. 

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
        The first parameter.

    Returns
    -------
    array-like
        [num_onsets, sum_onsets, sum_peaks, sum_difference, sum_areas, sum_od, mean_od, mean_amp, total_amp]

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
    Techniques involving power spectral density (PSD) similar to those used to study heart rate 
    variability can be used to analyze EDA. While heart rate variability is influenced by both the 
    sympathetic and parasympathetic nervous systems, because EDA is strongly correlated to 
    sweat production, its PSD can be used to measure arousal of the sympathetic nervous system 
    in isolation [1]. The very low frequency (VLF) , low frequency (LF), and high frequency (HF) 
    passbands are 0.001-0.045 Hz, 0.045-0.15 Hz, and 0.15-0.25 Hz respectively [2]. LF changes in 
    HRV derive from the influence of the cardiac sympathetic nerves [3], so the same LF passband 
    was used for IBI and EDA. The PSD was approximated using a periodogram, and the integral 
    for each passband was calculated using composite trapezoidal integration. 

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
        The first parameter.

    Returns
    -------
    array-like
        [vlf_integral,lf_integral,hf_integral]
        
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
    Description of module level function. 

    Parameters
    ----------
    signal : array-like
        The first parameter.

    Returns
    -------
    bool
        eda_passbands(signal)[1]/eda_passbands(signal)[2]

    """
    return eda_passbands(signal)[1]/eda_passbands(signal)[2]

def eda_all_feat(signal): #returns all eda features in data frame
    """
    Description of module level function. 

    Parameters
    ----------
    signal : array-like 
        The first parameter.

    Returns
    -------
    DataFrame
        ['eda_mean1diff', 'eda_std1diff', 'eda_mean2diff', 'eda_std2diff', 'eda_lf/hf', 'eda_num_onsets', 'eda_sum_onsets', 'eda_sum_peaks', 'eda_sum_difference',\
        'eda_sum_areas', 'eda_sum_od', 'eda_mean_od', 'eda_mean_amp', 'eda_total_amp']


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