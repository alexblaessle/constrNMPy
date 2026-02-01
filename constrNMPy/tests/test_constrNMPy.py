import constrNMPy as cNM 
import numpy as np

def test_constrNM_rosenbrock_non_global():

	res=cNM.constrNM(cNM.test_funcs.rosenbrock,[2.5,2.5],[2,2],[None,3],full_output=True)
	assert abs(res['xopt']-np.array([2.,3.])).sum()<1E-3

def test_constNM_beales():

	res=cNM.constrNM(cNM.test_funcs.beales,[0.5,1],[-5,-5],[10,None],full_output=False)
	
	assert abs(res['xopt']-np.array([3.,.5])).sum()<1E-3
	
	