
import re
import sys
import shutil
import itertools
import random


def parseDakotaParametersFile(paramsfilename):
    """Return parameters for application."""

    # setup regular expressions for parameter/label matching
    e = r'-?(?:\d+\.?\d*|\.\d+)[eEdD](?:\+|-)?\d+'  # exponential notation
    f = r'-?\d+\.\d*|-?\.\d+'                       # floating point
    i = r'-?\d+'                                    # integer
    value = e + '|' + f + '|' + i                   # numeric field
    tag = r'\w+(?::\w+)*'                           # text tag field

    # regular expression for standard parameters format
    standard_regex = re.compile('^\s*(' + value + ')\s+(' + tag + ')$')

    # open DAKOTA parameters file for reading
    paramsfile = open(paramsfilename, 'r')

    # extract the parameters from the file and store in a dictionary
    paramsdict = {}
    for line in paramsfile:
        m = standard_regex.match(line)
        if m:
            # print m.group()
            paramsdict[m.group(2)] = m.group(1)

    paramsfile.close()

    return paramsdict


def checknVar(nvar, paramsdict):
    """Check to make sure we have the right number of uncertain variables."""
    num_vars = 0
    if 'variables' in paramsdict:
        num_vars = int(paramsdict['variables'])
    if num_vars != nvar:
        message = 'Error: Simulation expected ' + str(nvar) + ' variables, found ' \
            + str(num_vars) + ' variables.'
        raise Exception(message)


def writeDakotaResultsFile(
        resultfilename, resultsdict, paramsdict, active_set_vector):
    """Write results of application for Dakota."""

    # Make sure number of functions is as expected.
    num_fns = 0
    if 'functions' in paramsdict:
        num_fns = int(paramsdict['functions'])
    if num_fns != len(resultsdict['fns']):
        raise Exception('Number of functions not as expected.')

    # write outputfile
    outfile = open(resultfilename, 'w')

    for func_ind in range(0, num_fns):
        # write functions
        if active_set_vector[func_ind] & 1:
            functions = resultsdict['fns']
            outfile.write(str(functions[func_ind]) +
                          ' f' + str(func_ind) + '\n')

    # write gradients
    for func_ind in range(0, num_fns):
        if active_set_vector[func_ind] & 2:
            grad = resultsdict['fnGrads'][func_ind]
            outfile.write('[ ')
            for deriv in grad:
                outfile.write(str(deriv) + ' ')
            outfile.write(']\n')

    outfile.close()


# -------------------------------------------------------------------
#  Output Redirection
# -------------------------------------------------------------------
# original source: http://stackoverflow.com/questions/6796492/python-temporarily-redirect-stdout-stderr
class RedirectOutput(object):
    """ with RedirectOutput(stdout,stderr)

        Temporarily redirects sys.stdout and sys.stderr when used in
        a 'with' contextmanager

        Example:
        with Redirect_output('stdout.txt','stderr.txt'):
            sys.stdout.write("standard out")
            sys.stderr.write("standard error")
            # code
        #: with output redirection

        Inputs:
            stdout - None, a filename, or a file stream
            stderr - None, a filename, or a file stream
        None will not redirect output

    """
    def __init__(self, stdout=None, stderr=None):

        _newout = False
        _newerr = False

        if isinstance(stdout, str):
            stdout = open(stdout, 'w')
            _newout = True
        if isinstance(stderr, str):
            stderr = open(stderr, 'a')
            _newerr = True

        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr
        self._newout = _newout
        self._newerr = _newerr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

        if self._newout:
            self._stdout.close()
        if self._newerr:
            self._stderr.close()

#: class output()


def formatAbscissasOrdinates(x, f):
    """Return 1 long string for both the ordinates and the abscissas.
              and the number n of uncertain variables."""

    # Make the elements lists this way it can handle multiple dimensions
    if type(x) is not list:
        x = [x]
    if type(f) is not list:
        f = [f]

    assert len(x) == len(f), 'Make sure the abscissas and the ordinates have the same outer dimension.'
    n = len(x)  # number of uncertain variables

    # Convert the abscissas to a list of strings for writing
    y = []
    for xi in x:
        yi = ['%.2f' % xj for xj in xi]
        y.append(yi)
    x = list(itertools.chain.from_iterable(y))  # Flattens out the list

    # Convert the ordinates to a list of strings for writing
    g = []
    for fi in f:
        gi = ['%.14f' % fj for fj in fi]
        gi.append('0.0')
        g.append(gi)
    f = list(itertools.chain.from_iterable(g))

    return x, f, n

def updateDakotaFile(method_dict, sample_number, x, f):
    """Update number of quadrature (expansion) points in Dakota file,
              method for PC,
              the histogram bin distributions."""

    x, f, n = formatAbscissasOrdinates(x, f)  # n is number of uncertain variables

    # Read in the dakota input file (assumes it is a working input file with histogram_bin_uncertain variables)
    # and write out an updated strip out file
    dakotaFilename = method_dict['dakota_filename']
    filein = dakotaFilename
    fileout = dakotaFilename + '.tmp'
    fr = open(filein, 'r')
    fw = open(fileout, 'w')
    filelines = fr.readlines()
    lines = []
    fr.close()

    # Get the key values for the options
    for line in filelines:
        if not line.strip().startswith('#') and line.rstrip():  # Remove comment lines and blank lines
            # parse input, assign values to variables
            line = line.replace('=', ' ')  # Replace the = with a space, so the split below takes care of it.
            splitline = line.split()
            key = splitline[0]
            value = splitline[1:]
            lines.append({key: value})

    # Update the number of points and the method block

    keys = [line.keys()[0] for line in lines]
    if 'polynomial_chaos' in keys:  # If we are running polynomial chaos, specified in the dakota input file.

        # Remove expansion order options
        expansion_order_options = ['collocation_points', 'expansion_samples', 'collocation_ratio']
        lines = [line for line in lines if line.keys()[0] not in expansion_order_options]

        # Write the desired method to select the coefficients
        coeff_method_map = {'quadrature': 'quadrature_order', 'sparse_grid': 'sparse_grid_level',
                            'regression': 'expansion_order'}
        coeff_method = coeff_method_map[method_dict['coeff_method']]
        coeff_methods = ['quadrature_order', 'sparse_grid_level', 'expansion_order']
        for i, line in enumerate(lines):
            if line.keys()[0] in coeff_methods:
                if coeff_method == 'expansion_order':
                    lines[i] = {coeff_method: [str(sample_number-1)]}
                    lines.insert(i+1, {'collocation_ratio': ['1']})
                    # lines.insert(i+2, {'least_absolute_shrinkage': []})
                    # lines.insert(i+2, {'noise_tolerance': ['1000']})
                    # lines.insert(i+2, {'l2_penalty': ['5']})
                    # Here I could insert the other options for this case.
                    # lines.insert(i+1, {'collocation_points': [str(sample_number)]})
                    # lines.insert(i+1, {'expansion_samples': [str(sample_number)]})
                    # lines.insert(i+2, {'tensor_grid': []})
                    # if sample_number > 9:  # The 9 works at least for the 1d case # I fixed Cross_validation in dakota src so no need for the if statement.
                    #     lines.insert(i+2, {'cross_validation': []})
                    lines.insert(i+2, {'cross_validation': []})
                    # lines.insert(i+3, {'noise_only': []})
                    # Use a random seed
                    # We want a consistent seed for when dakota gets called for the points and then with the actual powers
                    # lines.insert(i+2, {'seed': ['15347']})
                    if 'seed' not in keys:  # If we had already specified a seed don't overwrite it.
                        seed = random.randrange(1, 100000000)  # As long as the seed is less the max int (2147483647)should be fine
                        lines.insert(i+2, {'seed': [str(seed)]})
                else:
                    lines[i] = {coeff_method: [str(sample_number)]}
                break
        if method_dict['verbose']:
            # Add file with the polynomial approximation
            lines.insert(i+1, {'import_approx_points_file': ["'approximate_at.dat'"]})
            lines.insert(i+2, {'annotated': []})
            lines.insert(i+3, {'export_approx_points_file': ["'approximated.dat'"]})
            lines.insert(i+4, {'annotated': []})

    if 'sampling' in keys:  # If we are running Monte Carlo, specified in the dakota input file
        for i, line in enumerate(lines):
            if line.keys()[0] == 'samples':
                lines[i]['samples'] = [str(sample_number)]
                # We want a consistent seed for when dakota gets called for the points and then with the actual powers
                # lines.insert(i+1, {'seed': ['15347']})
                if 'seed' not in keys:  # If we had already specified a seed don't overwrite it.
                    seed = random.randrange(1, 100000000)  # As long as the seed is less the max int (2147483647)should be fine
                    # In case you want to control the seed through the offset, useful in the sampling case
                    # seed = method_dict['offset'] + 1  # a seed of 0 doesn't work, so make sure it is at least 1.
                    lines.insert(i+1, {'seed': [str(seed)]})

    # Update the variables

    # Because the descriptor keyword can exist multiple times (for the variables and for the responses).
    # Here assumes the variables block is before the responses block
    already_updated = False
    for i, line in enumerate(lines):
        if 'histogram_bin_uncertain' in line:
            lines[i] = {'histogram_bin_uncertain': str(n)}
        if 'abscissas' in line:
            lines[i] = {'abscissas': x}
        if 'ordinates' in line:
            lines[i] = {'ordinates': f}
        if 'descriptors' in line and not already_updated:
            descriptor = []
            for j in range(n):
                value = "'x%s'" % str(j+1)
                descriptor.append(value)
            lines[i] = {'descriptors': descriptor}
            already_updated = True

    # Write the new temp file
    for line in lines:
        towrite = line.keys()[0] + ' ' + ' '.join(line.values()[0]) + ' \n'
        fw.write(towrite)
    fw.close()

    # shutil.move(fileout, filein)
