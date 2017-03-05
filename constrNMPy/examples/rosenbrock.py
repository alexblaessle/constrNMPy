"""Example script for a simple optimization using constrNM.

We want to find the minimum of the rosenbrock function with constraints 2<x and 2<y<3. 

The rosenbrock function for two dimensions is already implemented in ``test_funcs.rosenbrock``.

"""

# Define initial guess
x0=[2.5,2.5]

# Define lower and upper bounds
LB=[2,2]
UB=[None,3]

# Call optimizer
import constrNMPy as cNM 
res=cNM.constrNM(cNM.test_funcs.rosenbrock,x0,LB,UB,full_output=True)

# Print results
cNM.printDict(res)

