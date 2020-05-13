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

class EDA:

  	def __init__(self): 
		pass

	def eda_median(signal):
	    return np.median(signal)

	def mean_firstdiff(signal):
	    return np.mean(np.abs(np.diff(signal, n=1)))

	def std_firstdiff(signal):
	    return np.std(np.abs(np.diff(signal, n=1)))

	def mean_seconddiff(signal):
	    return np.mean(np.abs(np.diff(signal, n=2)))

	def std_seconddiff(signal):
	    return np.std(np.abs(np.diff(signal, n=2)))

	def find_peaks(signal):
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

	def EDA_passbands(signal):
	    fs,pxx = scipy.signal.periodogram(signal, nfft = 1000, scaling = 'density', detrend = 'linear')

	    vlfband = (fs > 0.001)*(fs < 0.045)
	    lfband = (fs > 0.045)*(fs < 0.15)
	    hfband = (fs > 0.15) * (fs < 0.25)

	    vlf_integral = np.trapz(pxx[vlfband],fs[vlfband])
	    lf_integral = np.trapz(pxx[lfband],fs[lfband])
	    hf_integral = np.trapz(pxx[hfband],fs[hfband])
		return [vlf_integral,lf_integral,hf_integral]

	def eda_lfhf(signal):
	    return EDA_passbands(signal)[1]/EDA_passbands(signal)[2]

	def eda_all_features(signal): #returns all eda features in data frame
	    functions = [eda_median, mean_firstdiff, std_firstdiff, mean_seconddiff, std_seconddiff, eda_lfhf]
	    measure_names = ['eda_median', 'eda_mean1diff', 'eda_std1diff', 'eda_mean2diff', 'eda_std2diff', 'eda_lf/hf']
	    features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
	    fdf = pd.DataFrame(columns = measure_names, data = features.T)
	    
	    orient_features = np.asarray(orienting_features(signal)).reshape(-1,1)
	    orient_featurenames = ['eda_num_onsets', 'eda_sum_onsets', 'eda_sum_peaks', 'eda_sum_difference',\
	                           'eda_sum_areas', 'eda_sum_od', 'eda_mean_od', 'eda_mean_amp', 'eda_total_amp']
	    ofdf = pd.DataFrame(columns = orient_featurenames, data = orient_features.T)
	    
	    return fdf.join(ofdf)