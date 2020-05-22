"""
Outline of package

								Package hsltools

__init__.py				basics.py				ibi.py				eda.py				etc.			
							basic funcs				specialized			specialized
													run all				run all				
					

"""

"""
questions/notes:

	in signalstatistics, the dfa_exp is set to 0 but dfa(signal) is commented out, is this intentional? 
        update: dfa was replaced with scaled correlation time according to apara's sandbox, is this preferred? 
	are get_shimmer and get_split_shimmer necessary? They seem specific to our data. 
    unsure what shimmer.fnn does
    multivar features are not detailed / Cxyy & xbicorr are incomplete. 


"""

__all__ = ["basics", "eda", "ibi", "multivar", "shimmer"]
