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

class Multivar:

  	def __init__(self): 
		pass

	def resample(signal1, signal2):
    	#samples = np.max([len(signal1), len(signal2)])
	    ts = np.linspace(0, 1500, 1500)

    	f1 = scipy.interpolate.interp1d(np.linspace(0, 1500, len(signal1)), signal1, kind = 'linear')
    	signal1 = f1(ts)

    	f2 = scipy.interpolate.interp1d(np.linspace(0, 1500, len(signal2)), signal2, kind = 'linear')
    	signal2 = f2(ts)
    
	    return signal1, signal2

	def normalize(signal):
    	return (signal - np.mean(signal)) / np.std(signal)

	def detrend(signal):
	    line = np.polyfit(np.arange(len(signal)), signal, 2)
   		return signal - np.polyval(line, np.arange(len(signal)))

	def xcorr_lagtime(signal1, signal2, make_plot = False, sig1 = '', sig2 = ''):
    	signal1, signal2 = resample(signal1, signal2)
    
    	X = (signal1 - np.mean(signal1)) / np.std(signal1)
   		Y = (signal2 - np.mean(signal2)) / np.std(signal2)
    
		#     X = detrend(X)
		#     Y = detrend(Y)
    
    	xcorr = np.correlate(X, Y, mode='full')
    	xcorr = xcorr[(xcorr.size // 2 ):] / np.max(xcorr)
    
    	tau = np.argmax(xcorr)
    
    	if make_plot:
        	plt.figure(figsize = (12,10))
        	plt.subplot(211)
        
        	plt.plot(X, label = sig1)
        	plt.plot(Y, label = sig2)
        	plt.legend()
        	plt.xlabel('Time (s)')
        	plt.ylabel('Normalized Signal')
        
        	plt.subplot(212)
        
        	plt.plot(xcorr, 'k', label = 'Cross-Correlation')
        	plt.axvline(tau, color = 'k', linestyle = '--')
        	plt.xlabel('Lag (s)')
       		plt.ylabel('Coefficient')
        	plt.legend()
        
        	plt.tight_layout()
    
    	return tau

    def xbicorr(x, y):
    	    
        x,y = resample(x,y)
        x,y = normalize(x), normalize(y)
    	    
        N = len(x)
        L = int(np.floor(N**0.4))
    	    
   	    z = 0
   	    for s in range(2, L+1):
 		    for r in range(1, L+1):
    	        m = np.max([r, s])
    	        z += (N-m) * Cxyy(x, y, r, s, N)**2
        return z
	
	def Cxyy(x, y, r, s, N):
    	z = 0
    	m = np.max([r, s])
   		for i in range(0, N-m-1):
       		z += x[i] * y[i+r] * y[i+s]
    	z /= (N-m)
	    return z

	def xbicorr(x, y):
	        
        x,y = resample(x,y)
        x,y = normalize(x), normalize(y)
	        
        N = len(x)
        L = int(np.floor(N**0.4))
	        
        z = 0
        for s in range(2, L+1):
            for r in range(1, L+1):
                m = np.max([r, s])
                z += (N-m) * Cxyy(x, y, r, s, N)**2
        return z

	def multi_features(signal1, signal2, name):
    	functions = [xcorr_lagtime, xbicorr]
    	measure_names = [name+'_xcorr_lag', name+'_xbicorr']
    	features = np.asarray([func(signal1, signal2) for func in functions]).reshape(-1,1)
    	functionsdf = pd.DataFrame(columns = measure_names, data = features.T)
    	return fdf



