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

class Basics:
    #to do/questions: 
    #is self necessary in this instance, if not is the init function blank?
    #Is there any object, can I just have a list of functions without a class since they will all be static methods? 
    #Is the signal the object? 
    #will all inputted data have the same attributes?
    #I have been reading a lot about python packages and looking at the source code for data analysis packages, but I still need to do more research 

	def __init__(self): 


    #copied from github
	def signal_statistics(signal): #copied from github
    	mean = np.mean(signal)
    	std = np.std(signal)
    	skewness = stats.skew(signal)
    	kurtosis = stats.kurtosis(signal) #sharpness of the peak of a freq-distrib curve
    	maximum = np.max(signal)
    	minimum = np.min(signal)
    	iqr = stats.iqr(signal)
    	variation = stats.variation(signal)
    	entropy = stats.entropy(np.abs(signal))
    	dfa_exp = 0  #dfa(signal), discriminant function analysis
    	return np.asarray([mean, std, skewness, kurtosis, maximum, minimum, iqr, variation, entropy, dfa_exp])

    #make into data frame
	def ss_dataframe(signal):
		basic_stats = signal_statistics(signal).reshape(-1,1)
		stat_names = ['mean', 'std', 'skewness', 'kurtosis', 'maximum', 'minimum', 'iqr', 'variation', 'entropy', 'dfa_exp']
		fdf = pd.DataFrame(columns = stat_names, data = basic_stats.T)
		return fdf
