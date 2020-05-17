"""
Outline of package

								Package hsltools

__init__.py				basics.py				ibi.py				eda.py				etc.			
							basic funcs				specialized			specialized
													run all				run all				
					

"""

"""questions:

	in signalstatistics, the dfa_exp is set to 0 but dfa(signal) is commented out, is this intentional?  
	in multivar, xcorr_lagtime(signal1, signal2, make_plot = False, sig1 = '', sig2 = ''), should the makeplot just be deleted? 



"""

__all__ = ["basics", "eda", "ibi", "multivar", "shimmer"]
