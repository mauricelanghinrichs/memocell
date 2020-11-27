
"""
Some memocell utility functions are listed below. Particularly, methods to
compute the phase-type distribution for a hidden layer from (multiple, parallel)
Erlang channels. The methods were partially adapted from
`butools <https://github.com/ghorvath78/butools>`_, please see there
for more waiting time and phase-type distribution related helper methods.
"""

import numpy as np
from scipy.linalg import expm
import numpy.linalg as la

### general functions
### Phase-type densities from selected hidden network schemes
def phase_type_pdf(alpha, A, x):
    """Returns the probability density function of a continuous
    phase-type distribution.

    `Note`: Method adapted from `butools <https://github.com/ghorvath78/butools>`_,
    see there for further phase-type related methods.

    Parameters
    ----------
    alpha : 1d numpy.matrix
        The initial probability vector of the phase-type
        distribution (with shape `(1,m)`).
    A : 2d numpy.matrix
        The transient generator matrix of the phase-type
        distribution (with shape `(m,m)`).
    x : 1d numpy.ndarray or array-like
        The density function will be computed at these points.

    Returns
    -------
    ph_pdf : 1d numpy.ndarray
        The values of the phase-type density function at the
        corresponding `x` values.
    """
    ### adapted from butools "PdfFromME" and "PdfFromPH" methods
    _validate_phase_type(alpha, A)

    ph_pdf = np.array([np.sum(alpha*expm(A*xv)*(-A)) for xv in x])
    return ph_pdf

def phase_type_from_erlang(theta, n):
    """Returns initial probabilities :math:`\\alpha` and generator matrix :math:`A`
    for phase-type representation of an Erlang waiting time distribution with :math:`n` steps
    and rate :math:`\\theta` (mean waiting time :math:`1/\\theta`).

    `Note`: To obtain a phase-type density pass the results of this method into
    the method `utils.phase_type_pdf`.

    `Note`: The parametrisation implies the rate :math:`n\\cdot\\theta` on the
    individual exponentially-distributed substeps.

    `Note`: This method is more a complement for other phase-type representations.
    For computational tasks one can simply use the faster
    `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ version:

    >>> erlang_pdf = stats.gamma.pdf(x, a=n, scale=1/(theta*n))

    Parameters
    ----------
    theta : float
        Rate parameter of the complete Erlang channel (inverse of the mean Erlang
        waiting time).
    n : int or float
        Number of steps of the Erlang channel (shape parameter).

    Returns
    -------
    alpha : 1d numpy.matrix
        The initial probability vector of the phase-type
        distribution (with shape `(1,n)`).
    A : 2d numpy.matrix
        The transient generator matrix of the phase-type
        distribution (with shape `(n,n)`).
    """
    ### self-written, copied from env_PHdensity notebook
    ### butools can then be used to get density and network image with:
    ### 1) pdf = ph.PdfFromPH(a, A, x)
    ### 2) ph.ImageFromPH(a, A, 'display')

    # some checks
    if not isinstance(theta, float):
        raise ValueError('Float expected for theta.')
    if isinstance(n, int):
        pass
    elif isinstance(n, float) and n.is_integer():
        pass
    else:
        raise ValueError('Integer number expected for n.')
    if n<1:
        raise ValueError('Steps n expected to be 1 or more.')

    # preallocate initial probs and subgenerator matrix
    alpha = np.zeros((1, int(n)))
    A = np.zeros((int(n), int(n)))

    # first index sets source
    alpha[0, 0] = 1.0

    # substep rate
    r = n * theta

    # outflux from source
    A[0, 0] = -r

    # fill matrix
    for i in range(1, int(n)):
        A[i-1, i] = r
        A[i, i] = -r

    return np.matrix(alpha), np.matrix(A)

def phase_type_from_parallel_erlang2(theta1, theta2, n1, n2):
    """Returns initial probabilities :math:`\\alpha` and generator matrix :math:`A`
    for a phase-type representation of two parallel Erlang channels with parametrisation
    :math:`(\\theta_1, n_1)` and :math:`(\\theta_2, n_2)` (rate and steps of Erlang
    channels).

    `Note`: To obtain a phase-type density pass the results of this method into
    the method `utils.phase_type_pdf`.

    `Note`: The two Erlang channels split at the first substep into each channel.
    The parametrisation implies the rate :math:`n\\cdot\\theta` on the
    individual exponentially-distributed substeps for the respective channel.

    Parameters
    ----------
    theta1 : float
        Rate parameter of the first complete Erlang channel (inverse of the mean Erlang
        waiting time).
    theta2 : float
        Rate parameter of the second complete Erlang channel (inverse of the mean Erlang
        waiting time).
    n1 : int or float
        Number of steps of the first Erlang channel (shape parameter).
    n2 : int or float
        Number of steps of the second Erlang channel (shape parameter).

    Returns
    -------
    alpha : 1d numpy.matrix
        The initial probability vector of the phase-type
        distribution (with shape `(1,m)` where :math:`m=n_1+n_2-1`).
    A : 2d numpy.matrix
        The transient generator matrix of the phase-type
        distribution (with shape `(m,m)` where :math:`m=n_1+n_2-1`).
    """
    ### self-written, copied from env_PHdensity notebook
    ### butools can then be used to get density and network image with:
    ### 1) pdf = ph.PdfFromPH(a, A, x)
    ### 2) ph.ImageFromPH(a, A, 'display')

    # some checks
    for theta in (theta1, theta2):
        if not isinstance(theta, float):
            raise ValueError('Float expected for theta.')
    for n in (n1, n2):
        if isinstance(n, int):
            pass
        elif isinstance(n, float) and n.is_integer():
            pass
        else:
            raise ValueError('Integer number expected for n.')
        if n<1:
            raise ValueError('Steps n expected to be 1 or more.')

    # preallocate initial probs and subgenerator matrix
    alpha = np.zeros((1, int(n1 + n2)-1))
    A = np.zeros((int(n1 + n2)-1, int(n1 + n2)-1))

    # first index sets source
    alpha[0, 0] = 1.0

    # substep rates
    r1 = n1 * theta1
    r2 = n2 * theta2

    # outflux from source
    # (from competing channels)
    A[0, 0] = -(r1+r2)

    # fill matrix (first channel)
    l = [0] + list(range(1, int(n1)))
    for i, inext in zip(l[0:-1], l[1:]):
        A[i, inext] = r1
        A[inext, inext] = -r1

    # fill matrix (second channel)
    l = [0] + list(range(int(n1), int(n1+n2)-1))
    for i, inext in zip(l[0:-1], l[1:]):
        A[i, inext] = r2
        A[inext, inext] = -r2

    return np.matrix(alpha), np.matrix(A)

def phase_type_from_parallel_erlang3(theta1, theta2, theta3, n1, n2, n3):
    """Returns initial probabilities :math:`\\alpha` and generator matrix :math:`A`
    for a phase-type representation of three parallel Erlang channels with parametrisation
    :math:`(\\theta_1, n_1)`, :math:`(\\theta_2, n_2)` and :math:`(\\theta_3, n_3)`
    (rate and steps of Erlang channels).

    `Note`: To obtain a phase-type density pass the results of this method into
    the method `utils.phase_type_pdf`.

    `Note`: The three Erlang channels split at the first substep into each channel.
    The parametrisation implies the rate :math:`n\\cdot\\theta` on the
    individual exponentially-distributed substeps for the respective channel.

    Parameters
    ----------
    theta1 : float
        Rate parameter of the first complete Erlang channel (inverse of the mean Erlang
        waiting time).
    theta2 : float
        Rate parameter of the second complete Erlang channel (inverse of the mean Erlang
        waiting time).
    theta3 : float
        Rate parameter of the third complete Erlang channel (inverse of the mean Erlang
        waiting time).
    n1 : int or float
        Number of steps of the first Erlang channel (shape parameter).
    n2 : int or float
        Number of steps of the second Erlang channel (shape parameter).
    n3 : int or float
        Number of steps of the third Erlang channel (shape parameter).

    Returns
    -------
    alpha : 1d numpy.matrix
        The initial probability vector of the phase-type
        distribution (with shape `(1,m)` where :math:`m=n_1+n_2+n_3-2`).
    A : 2d numpy.matrix
        The transient generator matrix of the phase-type
        distribution (with shape `(m,m)` where :math:`m=n_1+n_2+n_3-2`).
    """
    ### self-written, copied from env_PHdensity notebook
    ### butools can then be used to get density and network image with:
    ### 1) pdf = ph.PdfFromPH(a, A, x)
    ### 2) ph.ImageFromPH(a, A, 'display')

    # some checks
    for theta in (theta1, theta2, theta3):
        if not isinstance(theta, float):
            raise ValueError('Float expected for theta.')
    for n in (n1, n2, n3):
        if isinstance(n, int):
            pass
        elif isinstance(n, float) and n.is_integer():
            pass
        else:
            raise ValueError('Integer number expected for n.')
        if n<1:
            raise ValueError('Steps n expected to be 1 or more.')

    # preallocate initial probs and subgenerator matrix
    alpha = np.zeros((1, int(n1 + n2 + n3)-2))
    A = np.zeros((int(n1 + n2 + n3)-2, int(n1 + n2 + n3)-2))

    # first index sets source
    alpha[0, 0] = 1.0

    # substep rates
    r1 = n1 * theta1
    r2 = n2 * theta2
    r3 = n3 * theta3

    # outflux from source
    # (from competing channels)
    A[0, 0] = -(r1+r2+r3)

    # fill matrix (first channel)
    l = [0] + list(range(1, int(n1)))
    for i, inext in zip(l[0:-1], l[1:]):
        A[i, inext] = r1
        A[inext, inext] = -r1

    # fill matrix (second channel)
    l = [0] + list(range(int(n1), int(n1+n2)-1))
    for i, inext in zip(l[0:-1], l[1:]):
        A[i, inext] = r2
        A[inext, inext] = -r2

    # fill matrix (third channel)
    l = [0] + list(range(int(n1+n2)-1, int(n1+n2+n3)-2))
    for i, inext in zip(l[0:-1], l[1:]):
        A[i, inext] = r3
        A[inext, inext] = -r3

    return np.matrix(alpha), np.matrix(A)

def _validate_phase_type(alpha, A, prec=1e-6):
    """Private validation method. Adapted from butools."""
    ### adapted from butools "CheckPHRepresentation" and "CheckMERepresentation" methods
    # if butools.checkInput and not CheckPHRepresentation (alpha, A):
    #     raise Exception("PdfFromPH: Input is not a valid PH representation!")
    #
    # if butools.checkInput and not CheckMERepresentation (alpha, A):
    #     raise Exception("PdfFromME: Input is not a valid ME representation!")

    ### check initial probability vector
    if not isinstance(alpha, np.matrix):
        raise TypeError('Numpy matrix expected for alpha.')

    if len(alpha.shape)<2:
        raise ValueError('Shape of (1,m) expected for alpha.')

    # alpha doesn't include the initial probability of the absorbing state
    # so it can be between 0 and 1
    if np.sum(alpha)<-prec*alpha.size or np.sum(alpha)>1.0+prec*alpha.size:
        raise ValueError('The sum of the alpha elements is less than zero or greater than one.')

    if np.min(alpha)<-prec:
        raise ValueError('Alpha has a negative element.')

    ### check generator matrix
    if not isinstance(A, np.matrix):
        raise TypeError('Numpy matrix expected for A.')

    if A.shape[0]!=A.shape[1]:
        raise ValueError('A is not a square matrix.')

    if np.any(np.diag(A)>=prec):
        raise ValueError('The diagonal of the generator A is not negative.')

    N = A.shape[0]
    odQ = A<-prec
    for i in range(N):
        odQ[i,i] = 0
    if np.sum(np.any(odQ))>0:
        raise ValueError('The generator A has negative off-diagonal element.')

    if np.max(np.sum(A,1))>prec:
        raise ValueError('The rowsum of the transient generator A is greater than 0.')

    ev = la.eigvals(A)
    if np.max(np.real(ev))>=prec:
        raise ValueError('The transient generator A has non-negative eigenvalue.')

    ix = np.argsort(np.abs(np.real(ev)))
    maxev = ev[ix[0]]
    if not np.isreal(maxev):
        raise ValueError('The dominant eigenvalue of the matrix is not real.')

    ### checks related to both
    if alpha.shape[1]!=A.shape[0]:
        raise ValueError('A and alpha have non-matching shapes.')
