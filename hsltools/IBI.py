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

class IBI:

  	def __init__(self): 
		pass

	def ibi_iqr(signal): #interquartile range
	    return scipy.stats.iqr(signal)

	def ibi_std(signal): 
	    return np.std(signal)

	def rmssd(signal):
	    return np.sqrt(np.mean(np.square(np.diff(signal))))

	def ibi_passbands(signal):
	    fs,pxx = scipy.signal.periodogram(signal, nfft = 1000, scaling = 'density', detrend = 'constant')

	    vlfband = (fs > 0.0033)*(fs < 0.04)
	    lfband = (fs > 0.04)*(fs < 0.15)
	    hfband = (fs > 0.15) * (fs < 0.4)

	    vlf_integral = np.trapz(pxx[vlfband],fs[vlfband]) #unsure what this does
	    lf_integral = np.trapz(pxx[lfband],fs[lfband])
	    hf_integral = np.trapz(pxx[hfband],fs[hfband])

	    return [vlf_integral,lf_integral,hf_integral]

	def ibi_lfhf(signal):
	    return ibi_passbands(signal)[1]/ibi_passbands(signal)[2]

	def ibi_all_features(signal): #returns all ibi features in data frame
    	functions = [ibi_iqr, ibi_std, rmssd, ibi_lfhf]
    	measure_names = ['ibi_iqr', 'ibi_std', 'ibi_rmssd', 'ibi_lf/hf']
    	features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
    	fdf = pd.DataFrame(columns = measure_names, data = features.T)
    	return fdf

