"""A B-spline basis class and tools for working with it.

"""

import numpy as np

from scipy.interpolate import splev
from matplotlib import pyplot as plt


class BSplineBasis:
    """A function-like object for evaluating B-spline bases.

    Basic Usage
    -----------

    The main interface is through function call syntax; an instance of
    the class can be called with a sequence of real-values and will
    produce a design matrix where each row is the value of a basis
    function at the corresponding value in the input sequence. For
    example, if b is a basis then:

    >>> b = BSplineBasis.with_knots([0, 5, 10], degree=1)
    >>> b([2.5, 7.5])
    array([[ 0.5,  0.5,  0. ],
           [ 0. ,  0.5,  0.5]])

    A basis can be constructed in one of two ways:

    1. Call the uniform class method with a lower and upper bound on
    the domain, the number of desired bases (i.e. the number of
    columns in the design matrices), and a degree (higher degrees
    yield smoother bases).

    2. Call the with_knots class method with a knot sequence that
    contains the lower and upper bound as the first and last elements
    respectively and a degree.

    You can visualize a basis by calling the plot method. You may need
    to call matplotlib.pyplot.show if your backend is not interactive.

    """

    def __init__(self, full_knots, degree):
        self._knots = np.array(full_knots)
        self._degree = int(degree)
        self._dimension = len(self._knots) - self._degree - 1
        self._tck = (self._knots, np.eye(self._dimension), self._degree)

    def __len__(self):
        return self._dimension

    def __call__(self, x):
        """Evaluate the bases at the given inputs."""
        bases = np.array(splev(x, self._tck))
        return bases.transpose()

    def __repr__(self):
        knot_str = '[' + ', '.join(str(k) for k in self._knots) + ']'
        return 'BSplineBasis({}, {})'.format(knot_str, self._degree)

    def plot(self, grid_size=200):
        """Plot the individual bases in the basis."""
        xgrid = np.linspace(self._knots[0], self._knots[-1], grid_size)
        for ygrid in self(xgrid).T:
            plt.plot(xgrid, ygrid)

    @classmethod
    def uniform(cls, low, high, num_bases, degree):
        '''Construct a uniform basis between low and high.

        Parameters
        ----------
        low : Lower bound of the basis.
        high : Upper bound.
        num_bases : The number of bases (dimension) of the basis.
        degree : The degree of the polynomial pieces.

        Returns
        -------
        A new BSplineBasis.

        '''
        num_knots = num_bases + degree + 1
        num_interior_knots = num_knots - 2 * (degree + 1)
        knots = np.linspace(low, high, num_interior_knots + 2)
        return cls.with_knots(knots, degree)

    @classmethod
    def with_knots(cls, knots, degree):
        '''Construct a basis based on a knot sequence.

        The knot sequence should contain the lower and upper bound as
        the first and last elements of the sequence. The elements in
        between are the interior knots of the basis.

        Parameters
        ----------
        knots : The knots (including the boundaries and interior knots).
        degree : The degree of the polynomial pieces.

        Returns
        -------
        A new BSplineBasis.

        '''
        knots = list(knots)
        knots = ([knots[0]] * degree) + knots + ([knots[-1]] * degree)
        return cls(knots, degree)


def pspline_penalty(basis, order=1):
    '''Return a differences penalty matrix.

    The penalty matrix can be used in least squares regression to bias
    the result to be smooth. In other words, rapid changes in
    coefficient values are penalized. Higher order difference matrices
    effectively decrease the degrees of freedom of the regression.

    '''
    D = np.diff(np.eye(len(basis)), order)
    return D @ D.T
