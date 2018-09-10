"""Example script for a simple optimization using constrNM.

We want to find the minimum of a shifted two dimensionsal quadrativ function with constraints -10<x and -10<y<10.

This example should illustrate how to pass option arguments to constrNM.

"""

# Import modules
import constrNMPy as cNM
import matplotlib.pyplot as plt
import numpy as np

def obj(x,a):

    """Objective function."""

    return (x[0]-a)**2+x[1]**2

# Parameters
a=5

# Create figure
fig = plt.figure()
fig.show()

# Plot function
X,Y=np.meshgrid(np.linspace(-10,10,100),np.linspace(-10,10,100))
x=np.array([X.flatten(),Y.flatten()])

# Create beales function for plot
f=obj(x,a)
F=f.reshape(100,100)

# Plot objective function
ax = fig.add_subplot(111)
p1=ax.contourf(X,Y,F,levels=np.linspace(f.min(),f.max(),100))
plt.draw()
#raw_input()


# Define initial guess
x0=[2.5,2.5]

# Define lower and upper bounds
LB=[-10,-10]
UB=[None,10]

# Call optimizer
res=cNM.constrNM(obj,x0,LB,UB,full_output=True,args=[a])

# Print results
cNM.printDict(res)
