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