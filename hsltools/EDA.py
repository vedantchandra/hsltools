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

class EDA:

  	def __init__(self): 
		pass

	def EDA_passbands(signal):
	    fs,pxx = scipy.signal.periodogram(signal, nfft = 1000, scaling = 'density', detrend = 'linear')
	    # plt.figure()
	    # plt.plot(fs,pxx)
	    # plt.xlim(0,0.05)
	    # plt.xlabel('Frequency (Hz.)')
	    # plt.ylabel('Spectral Density')

	    vlfband = (fs > 0.001)*(fs < 0.045)
	    lfband = (fs > 0.045)*(fs < 0.15)
	    hfband = (fs > 0.15) * (fs < 0.25)

	    # plt.figure(figsize = (13,7))
	    # plt.plot(fs,pxx,'k')
	    # #plt.xlim(0,0.05)
	    # plt.xlabel('Frequency (Hz.)')
	    # plt.ylabel('Spectral Density')
	    # plt.plot(fs,vlfband*np.max(pxx),label = 'VLF')
	    # plt.plot(fs,lfband*np.max(pxx),label = 'LF')
	    # plt.plot(fs,hfband*np.max(pxx),label = 'HF')
	    # plt.title('IBI Frequency Passbands')
	    # plt.legend()

	    vlf_integral = np.trapz(pxx[vlfband],fs[vlfband])
	    lf_integral = np.trapz(pxx[lfband],fs[lfband])
	    hf_integral = np.trapz(pxx[hfband],fs[hfband])
	    return [vlf_integral,lf_integral,hf_integral]

	def eda_lfhf(signal):
	    return EDA_passbands(signal)[1]/EDA_passbands(signal)[2]

	def eda_features(signal):
	    functions = [eda_median, mean_firstdiff, std_firstdiff, mean_seconddiff, std_seconddiff, eda_lfhf]
	    measure_names = ['eda_median', 'eda_mean1diff', 'eda_std1diff', 'eda_mean2diff', 'eda_std2diff', 'eda_lf/hf']
	    features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
	    fdf = pd.DataFrame(columns = measure_names, data = features.T)
	    
	    orient_features = np.asarray(orienting_features(signal)).reshape(-1,1)
	    orient_featurenames = ['eda_num_onsets', 'eda_sum_onsets', 'eda_sum_peaks', 'eda_sum_difference','eda_sum_areas', 'eda_sum_od', 'eda_mean_od', 'eda_mean_amp', 'eda_total_amp']
	    ofdf = pd.DataFrame(columns = orient_featurenames, data = orient_features.T)
	    
		return fdf.join(ofdf)	


		