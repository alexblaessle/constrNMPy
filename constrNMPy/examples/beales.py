"""Example script for a simple optimization using constrNM.

We want to find the minimum of the beales function with constraints -5<x<2.5 and -5<y.

The beales function for two dimensions is already implemented in ``test_funcs.beales``.

"""

# Define initial guess
x0=[2,5]

# Define lower and upper bounds
LB=[-5,-5]
UB=[2.5,None]

# Call optimizer
import constrNMPy as cNM
res=cNM.constrNM(cNM.test_funcs.beales,x0,LB,UB,full_output=True)

# Print results
cNM.printDict(res)
