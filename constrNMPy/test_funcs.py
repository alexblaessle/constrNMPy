#=====================================================================================================================================
#Copyright
#=====================================================================================================================================

#Copyright (C) 2017 Alexander Blaessle.
#This software is distributed under the terms of the GNU General Public License.

#This file is part of constNMPY.

#constNMPy is a small python package allowing to run a Nelder-Mead optimization via scipy's fmin function.

#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

#===========================================================================================================================================================================
#Module Functions
#===========================================================================================================================================================================

def beales(x):

    r"""Beales function.

    Beales function given by

    .. math:: f(x,y)=(1.5-x+xy)^2+(2.25-x+xy^2)^2+(2.625-x+xy^3)^2

    See also https://en.wikipedia.org/wiki/Test_functions_for_optimization .

    Args:
        x (numpy.ndarray): 2-D input array.

    Returns:
        float: ``f(x,y)``.
    """

    return (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2

def rosenbrock(x):

    r"""rosenbrock function.

    Rosenbrock function given by

    .. math:: f(x,y)=(1-x)^2+100(y-x^2)^2

    See also https://en.wikipedia.org/wiki/Test_functions_for_optimization .

    Args:
        x (numpy.ndarray): 2-D input array.

    Returns:
    float: ``f(x,y)``.
    """

    return (1-x[0])**2+100*(x[1]-x[0]**2)**2
