# constrNMPy
A Python package for constrained Nelder-Mead optimization.

constrNMPy is a Python package that allows to use scipy's `fmin` function for constrained problems. It is a reimplementation
of a popular [matlab package](https://de.mathworks.com/matlabcentral/fileexchange/8277-fminsearchbnd--fminsearchcon/content/FMINSEARCHBND/demo/html/fminsearchbnd_demo.html).

## Installation

Simply clone this repository via 

```bash
git clone https://github.com/alexblaessle/constrNMPy.git
```

and install with

```bash
python setup.py install
```

## Requirements

constrNMPy solely requires

* numpy
* scipy

## Usage

Using constrNMPy is analogous to other scipy optimization functions, such as `fmin`. Let's say we want
to optimize a rosenbrock function with a given range for `x` and `y`

```python

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

```

More examples are in `constrNMPy/examples/`. 

## Tests

To test constrNMPy, simply run 

```bash
pytest constrNMPy/tests/
```
## API

The API of  can be found [here]() .