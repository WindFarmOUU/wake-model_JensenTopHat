
import subprocess
import sys
import numpy as np
from dakotaInterface import RedirectOutput


def getSamplePoints(dakotaFile):
    """Call Dakota to get the sample points.

    Args:
        dakotaFile (string): The dakota input file

    Returns:
        x (np.array): A vector of sample points

    """
    dakotaInput = dakotaFile + '.tmp'

    print 'Calling Dakota...'
    # Pipe the output
    log = 'logDakota.out'
    err = log  # will append the error to the output
    with RedirectOutput(log, err):
        # dakotaInput = '--version'
        subprocess.check_call(['dakota', dakotaInput], stdout=sys.stdout,
                              stderr=sys.stderr)

    print 'finished calling Dakota.'

    # read the points from the dakota tabular file
    dakotaTabular = 'dakota_tabular.dat'
    f = open(dakotaTabular, 'r')
    line = f.readline()
    n = len(line.split())
    x = [[] for i in range(2, n-1)]  # create list to hold the variables

    for line in f:
        for i in range(len(x)):
            x[i].append(float(line.split()[2+i]))

    f.close()

    # Read the input file to determine what to do for the weights
    f = open(dakotaInput, 'r')
    for line in f:
        if 'quadrature_order' in line and not line.strip().startswith('#'):
            dakotaTabular = 'dakota_quadrature_tabular.dat'
        elif 'sparse_grid_level' in line and not line.strip().startswith('#'):
            dakotaTabular = 'dakota_sparse_tabular.dat'
        elif 'expansion_order' in line and not line.strip().startswith('#'):
            dakotaTabular = ''
        elif 'sampling' in line and not line.strip().startswith('#'):
            dakotaTabular = ''
        else:
            pass

    f.close()

    # read the weights from the dakota quadrature tabular file (Only prints when running verbose)
    if dakotaTabular:
        f = open(dakotaTabular, 'r')
        f.readline()
        w = []
        for line in f:
            w.append(float(line.split()[1]))

        f.close()
        w = np.array(w)
        np.testing.assert_almost_equal(np.sum(w), 1.0, decimal=8, err_msg='the weights should add to 1.')
    else:
        w = np.array(None)  # The array is necessary because of OpenMDAO

    return x, w


if __name__ == '__main__':
    dakotaFileName = 'dakotaAEPdirection.in'
    points, weights = getSamplePoints(dakotaFileName)
    print 'points = ', points
    print 'weights = ', weights
