import pickle as p
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from tqdm import tqdm
from .dfa import dfa
import scipy.signal as sig
from sklearn.linear_model import LogisticRegression
from scipy.interpolate import interp1d
from scipy.integrate import romb
plt.rcParams.update({'font.size': 22})
import glob
from scipy.signal import medfilt,butter
import scipy
import math
from scipy.integrate import odeint

from .Basics import signal_statistics

"""
Shimmer is a module consisting of features designed to specifically analyze Shimmer sensors.

"""

def spectrum_statistics(signal):
    """
    Returns an array containing basic statistics of the signal (peak, peak magnitude, integral, 
    energy, shannon, spectral centroid).

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose spectrum statistics are desired.

    Returns
    -------
    array-like
        Returns array of spectrum statistics [peak, peakmag, integral, energy, shannon, spectral_centroid].

        peak - peak frequency of the signal
        peakmag - peak power of the signal 
        integral - integral of signal using the composite trapezoidal rule
        energy - spectral energy of the signal 
        shannon - The Shannon entropy, or self-information is a quantity that identifies the amount of information 
        associated with an event using the probabilities of the event. There is an inverse relationship 
        between probability and information, as low-probability events carry more information [3],[4].
        spectral_centroid - spectral centroid of the signal

        [3] https://arxiv.org/pdf/1405.2061.pdf
        [4] https://towardsdatascience.com/the-intuition-behind-shannons-entropy-e74820fe9800


    """
    fs,pxx = scipy.signal.periodogram(signal, fs = 50, nfft = 1000, scaling = 'density', detrend = 'constant')

    peak = fs[np.argmax(pxx)]
    peakmag = np.max(pxx)
    integral = np.trapz(pxx,fs)
    energy = np.dot(pxx,pxx)
    shannon = np.sum(pxx*np.log(1/pxx))
    normalized_spectrum = pxx / sum(pxx)  
    normalized_frequencies = np.linspace(0, 1, len(pxx))
    spectral_centroid = sum(normalized_frequencies * normalized_spectrum)/sum(normalized_spectrum)

        # Add wavelet analysis

    return [peak, peakmag, integral, energy, shannon, spectral_centroid]

def max_dists(signal, m):
    """
    Returns the maximum Chebyshev distances of the signal.   

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose maximum distances are desired.
    m : int
        A positive integer that indicates the length of each vector.

    Returns
    -------
    array-like
        Returns the maximum Chebyshev distances, type double, of the signal. 
   
    """
    count = 0
    N = len(signal)
    max_dists = []
    while count < N-m-1:
        x_i = signal[count:count+m-1]
        x_j = signal[count+1:count+m]
        max_dists.append(scipy.spatial.distance.chebyshev(x_i, x_j))
        count += 1
    return max_dists

def approx_entropy(signal):
    """
    Returns the approximate entropy of the signal.

    Approximate entropy (ApEn) is a method for determining system complexity of biological signals 
    [7]. ApEn is based on the sequence recurrence of the data. ApEn quantifies the unpredictability 
    of fluctuations in a time-domain dataset. The minimum value for ApEn is 0, indicating a 
    completely predictable dataset [8]. The algorithm for ApEn [9] relies on two parameters: m (see 
    max_dists) and r, a positive real number that indicates a “scaling range,” or level of filtering. In 
    this case, m = 2 and r = 0.25 [2].

    [2]
    Lee, C.-H.; Sun, T.-L.; Jiang, B.C.; Choi, V.H. Using Wearable Accelerometers in a 
    Community Service Context to Categorize Falling Behavior. Entropy 2016, 18, 257.
    [7]
    Pincus, S. Approximate entropy (apen) as a complexity measure. Chaos: An 
    Interdisciplinary Journal of Nonlinear Science 5, 110–117 (1995). 
    [8]
    G. M. Lee, S. Fattinger, A. L. Mouthon, Q. Noirhomme, and R. Huber, 
    “Electroencephalogram approximate entropy influenced by both age and sleep,” Front. 
    Neuroinf. 7, 33 (2013). https://doi.org/10.3389/fninf.2013.00033
    [9]
    Pincus SM. Approximate entropy as a measure of system complexity. Proc Natl Acad Sci 
    U S A. 1991;88(6):2297–2301. doi:10.1073/pnas.88.6.2297

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose approximate entropy is desired.

    Returns
    -------
    float
        Returns the approximate entropy of the signal. 
   
    """
    N = len(signal)
    m = 2 #or 3 -- dimensionality? 
    r = 0.3
        
    
    def phi(m):
        d_func = max_dists(signal,m)
        C = []
        for i in d_func:
            if i <= r and i > 0:
                C.append(i/(N-m+1)) 
        return (N - m + 1.0)**(-1) * sum(np.log10(C))
    return (abs(phi(m)-phi(m+1)))

def sample_entropy(signal):
    """
    Returns the sample entropy of the signal.   

    Sample entropy (SampEn) is an entropy measure that quantifies regularity and complexity. 
    SampEn is the negative natural logarithm of the probability that two uniform, random vectors of 
    m+1 consecutive data points have distance r given that the corresponding vectors of length m 
    have a distance less than or equal to r [11]. SampEn has been used with center of pressure 
    (COP) time series [10], heart rate, and other biological data series. SampEn addresses the bias 
    of approximate entropy, which increases asymptotically as N (the number of data points) 
    increases [12]. SampEn can measure postural control in the assessment of sensory integration 
    and vestibular function [11]. The algorithm relies on parameters m and r (see approx_entropy), 
    in this case, m = 4 and r = 0.3 [10]. 

    [10]
    Montesinos, L., Castaldo, R. & Pecchia, L. On the use of approximate entropy and 
    sample entropy with centre of pressure time-series. J NeuroEngineering Rehabil 15, 116 
    (2018). https://doi.org/10.1186/s12984-018-0465-9
    [11]
    Lubetzky AV, Harel D, Lubetzky E (2018) On the effects of signal processing on sample 
    entropy for postural control. PLoS ONE 13(3): e0193460. 
    https://doi.org/10.1371/journal.pone.0193460
    [12]
    Delgado-Bonal A, Marshak A. Approximate Entropy and Sample Entropy: A 
    Comprehensive Tutorial. Entropy. 2019; 21(6):541.

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose sample entropy is desired.

    Returns
    -------
    float
        Returns the sample entropy of the signal. 
   
    """
    N = len(signal)
    m = 2
    r = 0.2
    
    A_list = max_dists(signal,m+1)
    B_list = max_dists(signal,m)
    
    count_A = 0
    count_B = 0
    for i in A_list:
        if i < r:
            count_A += 1
    for i in B_list: 
        if i < r:
            count_B += 1
    if count_B == 0:
        return np.nan
    return math.log10(count_A/count_B)*-1    


def multiscale_entropy(signal):
    """
    Returns the mean multiscale entropy of the signal.   

    Multiscale entropy (MSE) uses the same algorithm as sample entropy (see sample_entropy) 
    [12], and has been used to measure human postural signals [13], but further breaks down the 
    data by introducing another parameter. MSE collects every nth data point and then collects the 
    SampEn from that smaller dataset. This procedure is repeated for several values of n, ranging 
    from 2 to 20. From this series of sample entropies, the mean, standard deviation, and integral 
    are collected. MSE allows for the assessment of variability and fluctuations over a range of time 
    scales, meaning that the temporal dynamics of several different network structures can be 
    measured [14]. 

    [12]
    Delgado-Bonal A, Marshak A. Approximate Entropy and Sample Entropy: A 
    Comprehensive Tutorial. Entropy. 2019; 21(6):541.
    [13]
    S. Lu, J. Shieh and C. Hansen, "Applied a Multi-scale Entropy Algorithm to Analyze 
    Dynamic COP Signal via Accelerometer Sensor," 2016 Intl IEEE Conferences on 
    Ubiquitous Intelligence & Computing, Advanced and Trusted Computing, Scalable 
    Computing and Communications, Cloud and Big Data Computing, Internet of People, 
    and Smart World Congress (UIC/ATC/ScalCom/CBDCom/IoP/SmartWorld), Toulouse, 
    2016, pp. 127-132.
    [14]
    Gao J, Hu J, Liu F and Cao Y (2015) Multiscale entropy analysis of biological signals: a 
    fundamental bi-scaling law. Front. Comput. Neurosci. 9:64. doi: 
    10.3389/fncom.2015.00064

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose mean multiscale entropy is desired.

    Returns
    -------
    ndarray
        Returns the mean multiscale entropy of the signal.  
   
    """

    mses = []
    for T in range(2,20,2):
        coarse_grain = []
        i = 0
        while i < len(signal):
            new_val = sum(signal[i:i+T])/T
            coarse_grain.append(new_val)
            i += T
        mses.append(sample_entropy(coarse_grain))
    return np.mean(mses)

# def fnn(signal, tau = None, max_m = 10):
#     """
#     Returns the fnn of the signal. (***Unsure what this does, so it is not included in shimmer_all_features***)

#     Parameters
#     ----------
#     signal : array-like
#         Array containing numbers whose fnns are desired.
#     tau : array-like
#         Tau, default None
#     max_m : int
#         Maximum number of rows. 

#     Returns
#     -------
#     array-like
#         Returns the fnns.  
   
#     """
#     fnns = [];
#     if tau is None:
#         tau = np.argmax(delay.acorr(signal) < 1 / np.e)

#     for m in tqdm(range(1,max_m + 1)):

#         N2 = len(signal) - tau*(m-1);
#         xe = [];
#         for mi in np.arange(m):
#             xe.append(signal[(np.arange(N2) + tau * (mi-1))]);
#         Rtol = 15;
#         falsecount = 0;
#         xe = np.asarray(xe).T
#         for i in np.arange(len(xe[:,0])-1): #check minus 1
#             Rdmin = 1000;
#             for j in np.arange(len(xe[:,0])-1):

#                 Rd = scipy.linalg.norm(xe[i,:]-xe[j,:]);

#                 if j == i:
#                     continue;
#                 elif Rd < Rdmin:
#                     idx = j;
#                     Rdmin = Rd;
#             j = idx;
#             Rdnext =  scipy.linalg.norm(xe[i+1,:]-xe[j+1,:]);
#             Rd = scipy.linalg.norm(xe[i,:]-xe[j,:]);
#             if Rd == 0:
#                 continue
#             R = Rdnext/Rd;
#             if R > Rtol:
#                 falsecount = falsecount + 1;    
#         fnnprop = falsecount/len(xe[:,0]);
#         fnns.append(fnnprop)

#         if fnnprop < 0.005:
#             break
    
#     return fnns

def spectral_flux(signal, nsplits):
    """
    Returns the means and standard deviations of the spectral flux for the signal.   

    Spectral flux is a “low-uncertainty” feature, meaning that it is not highly impacted by suboptimal 
    calibration or placement of the sensing device [1]. It measures how quickly the power spectral 
    changes between two adjacent windows, and was calculated using the squared difference 
    between the normalized magnitudes of these spectra within the windows [2].

    [1]
    Dargie, Waltenegus. (2009). Analysis of Time and Frequency Domain Features of 
    Accelerometer Measurements. Proceedings - International Conference on Computer 
    Communications and Networks, ICCCN. 1 - 6. 10.1109/ICCCN.2009.5235366.
    [2]
    B. Boashash, "Detection classification and estimation in the (t-f) domain" in 
    Time-Frequency Signal Analysis and Processing: A Comprehensive Reference, London, 
    U.K.:Elsevier, pp. 693-743, 2016. 

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose spectral flux means and standard deviations are desired.
    nsplits : int
        Number of splits of the signal. 

    Returns
    -------
    array-like
        Returns array of spectral flux features [means, stds]

        means - mean of the spectral flux values for each signal with window sizes ranging from 2 to 500
        stds - standard deviation of the spectral flux values for each signal with window sizes ranging from 2 to 500
   
    """
    window = len(signal)//nsplits
    sig_list = []
    i = 0
    while i < len(signal):
        fs, fourier_sig = sig.periodogram(signal[i:i+window], scaling = 'density')
        sig_list.append(fourier_sig)
        i = i+window
    flux = []
    for j in range(0,len(sig_list)-2):
        norm_window = sig_list[j]/sum(sig_list[j])
        prev_window = sig_list[j+1]/sum(sig_list[j+1])
        flux.append(sum((norm_window-prev_window)**2))
    return [np.mean(flux), np.std(flux)]

def flux_stats(signal):
    """
    Returns statistics extracted from the spectral flux (see spectral_flux) of the signal (maximum, 
    minimum, integral, max inflection point, min inflection point, slope, intercept, r value)   

    Parameters
    ----------
    signal : array-like
        Array containing numbers whose spectral flux statistics are desired. 

    Returns
    -------
    array-like
        Returns array of spectral flux statistics including the linear regression [maximum, minimum, 
        integral, inflection_pt_max, inflection_pt_min, slope, intercept, rval]


        maximum - maximum of the spectral_flux means
        minimum - minimum of the spectral_flux means
        integral - integral of the spectral_flux means using the trapezoidal integration method
        inflection_pt_max - maximum inflection point after a smoothing Savitzky-Golay filter was applied 
        inflection_pt_min - minimum inflection point after a smoothing Savitzky-Golay filter was applied
        slope - slope of spectral_flux means after a smoothing Savitzky-Golay filter was applied
        intercept - intercept of spectral_flux means after a smoothing Savitzky-Golay filter was applied
        rval - r value of spectral_flux means after a smoothing Savitzky-Golay filter was applied
   
    """


    fluxmeans = []
    fluxstds = []
    n = range(2,500,10)
    for i in range(2,500,10):
        nsplits = i
        mean, std = spectral_flux(signal, nsplits)
        fluxmeans.append(mean)
        fluxstds.append(std)

    maximum = max(fluxmeans)
    minimum = min(fluxmeans)
    integral = np.trapz(fluxmeans,n)
    yhat = sig.savgol_filter(fluxmeans, 9, 3)
    inflection_pt_max = sig.argrelextrema(yhat, np.greater)[0][0]
    inflection_pt_min = sig.argrelextrema(yhat, np.less)[0][0]
    start = max([inflection_pt_max,inflection_pt_min])
    later_smooth = yhat[start:]
    later_slope = stats.linregress(range(len(later_smooth)),later_smooth)
    slope = later_slope[0]
    intercept = later_slope[1]
    rval = later_slope[2]

    return [maximum, minimum, integral, inflection_pt_max, inflection_pt_min, slope, intercept, rval]


def shimmer_all_features(signal):
    """
    Returns all of the Shimmer features of the signal in the form of a labeled data frame 
    (basics.signal_statistics, spectrum_statistics, flux_stats, approx_entropy, sample_entropy, 
    multiscale_entropy).  

    Parameters
    ----------
    signal : array-like 
        Array containing numbers whose Shimmer features are desired.

    Returns
    -------
    DataFrame
        Returns a data frame of all the Shimmer features with the column headings [shim_mean, 
        shim_std, shim_skewness, shim_kurtosis, shim_maximum, shim_minimum, shim_iqr, 
        shim_variation, shim_entropy, shim_corrtime, shim_peakfreq, shim_peakpower, shim_powerint, 
        shim_specenergy, shim_shannon, shim_spectral_centroid, flux_maximum, flux_minimum, 
        flux_integral, flux_inflection_pt_max, flux_inflection_pt_min, flux_slope, flux_intercept, flux_rval, 
        approx_entropy, sample_entropy, multiscale_entropy]

        shim_mean, shim_std, ..., shim_corrtime - see signal_statistics
        shim_peakfreq, shim_peakpower, ..., shim_spectral_centroid - see spectrum_statistics
        flux_maximum, flux_minimum, ..., flux_rval - see flux_stats
        approx_entropy - see approx_entropy
        sample_entropy - see sample_entropy
        multiscale_entropy - see multiscale_entropy

    """    
    shim_stats = signal_statistics(signal).reshape(-1,1)
    stat_names = ['shim_mean', 'shim_std', 'shim_skewness', 'shim_kurtosis', 'shim_maximum', 'shim_minimum', 'shim_iqr', 'shim_entropy', 'shim_corrtime']
    fdf = pd.DataFrame(columns = stat_names, data = shim_stats.T)

    spec_stats = np.asarray(spectrum_statistics(signal)).reshape(-1,1)
    specstat_names = ['shim_peakfreq','shim_peakpower','shim_powerint','shim_specenergy', 'shim_shannon', 'shim_spectral_centroid']
    spec_fdf = pd.DataFrame(columns = specstat_names, data = spec_stats.T)

    flux_stat = np.asarray(flux_stats(signal)).reshape(-1,1)
    fluxstat_names = ['flux_maximum', 'flux_minimum', 'flux_integral', 'flux_inflection_pt_max', 'flux_inflection_pt_min', 'flux_slope', 'flux_intercept', 'flux_rval']
    flux_fdf = pd.DataFrame(columns = fluxstat_names, data = flux_stat.T)
    
    functions = [approx_entropy, sample_entropy, multiscale_entropy]
    measure_names = ['approx_entropy', 'sample_entropy', 'multiscale_entropy']
    features = np.asarray([func(signal) for func in functions]).reshape(-1,1)
    entropy_fdf = pd.DataFrame(columns = measure_names, data = features.T)

    return ((fdf.join(spec_fdf)).join(flux_fdf)).join(entropy_fdf)