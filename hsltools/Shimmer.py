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

class Shimmer:

  	def __init__(self): 
		pass

	shimfiles = glob.glob('shimmerData/*/*')

	def get_shimmer(subjno, part, epochno):
    	epoch = num_to_epoch(epochno)
    	if epoch == 'REC':
        	epoch = 'Rec'
	    elif epoch == 'P1':
    	    epoch = '_P1'
   		elif epoch == 'P2':
        	epoch = '_P2'
    
    	for file in shimfiles:
        	if str(subjno) in file and part in file and epoch in file:
            	data = pd.read_csv(file, header = None)
    	try:
        	vectors = np.asarray(data[[1,2,3]])
    	except:
       		print('mising shimmer data. skipping...')
        	return np.repeat(np.nan, 1000)
    	mean = np.mean(vectors, axis = 0)
    	std = np.std(vectors, axis = 0)
    
	    vectors = (vectors - mean[np.newaxis,:])

    	#plt.plot(vectors)
    
   		z_vector = np.linalg.norm(vectors, axis = 1)

    	return z_vector

	def get_split_shimmer(subjno, part, epochno, splitno):

    	z_vector = get_shimmer(subjno,part,epochno)

    	return np.array_split(z_vector,n_splits)[splitno]

	def signal_statistics(signal):
    	mean = np.mean(signal)
    	std = np.std(signal)
    	skewness = stats.skew(signal)
    	kurtosis = stats.kurtosis(signal)
    	maximum = np.max(signal)
    	minimum = np.min(signal)
    	iqr = stats.iqr(signal)
    	variation = stats.variation(signal)
    	entropy = stats.entropy(np.abs(signal))
    	dfa_exp = 0#dfa(signal)
    	return np.asarray([mean, std, skewness, kurtosis, maximum, minimum, iqr, variation, entropy, dfa_exp])

	def spectrum_statistics(signal):
    
    	fs,pxx = scipy.signal.periodogram(signal, fs = 50, nfft = 1000, scaling = 'density', detrend = 'constant')

	    peak = fs[np.argmax(pxx)]
    	peakmag = np.max(pxx)
    	integral = np.trapz(pxx,fs)
    	energy = np.dot(pxx,pxx)
    	shannon = np.sum(pxx*np.log(1/pxx))

   	 	# Add wavelet analysis

    	return [peak, peakmag, integral, energy, shannon]

	def bodyshimmer_features(signal):
	    shim_stats = signal_statistics(signal).reshape(-1,1)
	    stat_names = ['bodyshim_mean', 'bodyshim_std', 'bodyshim_skewness', 'bodyshim_kurtosis', 'bodyshim_maximum', 'bodyshim_minimum', 'bodyshim_iqr', 'bodyshim_variation', 'bodyshim_entropy', 'bodyshim_dfa']
	    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
	    
	    spec_stats = np.asarray(spectrum_statistics(signal)).reshape(-1,1)
	    specstat_names = ['bodyshim_peakfreq','bodyshim_peakpower','bodyshim_powerint','bodyshim_specenergy', 'bodyshim_shannon']
	    spec_fdf = pd.DataFrame(columns = specstat_names, data = spec_stats.T)
	    
	    return fdf.join(spec_fdf)

	def headshimmer_features(signal):
	    shim_stats = signal_statistics(signal).reshape(-1,1)
	    stat_names = ['headshim_mean', 'headshim_std', 'headshim_skewness', 'headshim_kurtosis', 'headshim_maximum', 'headshim_minimum', 'headshim_iqr', 'headshim_variation', 'headshim_entropy', 'headshim_dfa']
	    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
	    
	    spec_stats = np.asarray(spectrum_statistics(signal)).reshape(-1,1)
	    specstat_names = ['headshim_peakfreq','headshim_peakpower','headshim_powerint','headshim_specenergy', 'headshim_shannon']
	    spec_fdf = pd.DataFrame(columns = specstat_names, data = spec_stats.T)
	    
	    return fdf.join(spec_fdf)
	  
	def hr_features(signal):
	    shim_stats = signal_statistics(signal).reshape(-1,1)
	    stat_names = ['hr_mean', 'hr_std', 'hr_skewness', 'hr_kurtosis', 'hr_maximum', 'hr_minimum', 'hr_iqr', 'hr_variation', 'hr_entropy', 'hr_dfa']
	    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
	    return fdf	

	def temp_features(signal):
	    shim_stats = signal_statistics(signal).reshape(-1,1)
	    stat_names = ['temp_mean', 'temp_std', 'temp_skewness', 'temp_kurtosis', 'temp_maximum', 'temp_minimum', 'temp_iqr', 'temp_variation', 'temp_entropy', 'temp_dfa']
	    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)
	    return fdf

	def shimmer_all_features(signal): #returns all shimmer features in data frame 
		bdshim_fun = bodyshimmer_features(signal)
		hdshim_fun = headshimmer_features(signal)
		hr_fun = hr_features(signal)
		temp_fun = temp_features(signal)
		return ((bdshim_fun.join(hdshim_fun)).join(hr_fun)).join(temp_fun)