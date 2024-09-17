
import numpy as np
# import matplotlib.pyplot as plt
import json
import shutil
import argparse
import time
import chaospy as cp
from openmdao.api import Problem
from AEPGroups import AEPGroupMulti
import distributions
import windfarm_setup
import approximate


def run(method_dict, n_high, n_low):
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
        'wake_model = 'floris', 'jensen', 'gauss', 'larsen' # larsen is not working  # a list containing 2 models
        'coeff_method' = 'quadrature', 'sparse_grid' or 'regression'
        'uncertain_var' = 'speed', 'direction' or 'direction_and_speed'
        'layout' = 'amalia', 'optimized', 'grid', 'random', 'test'
        'distribution' = a distribution object
        'dakota_filename' = 'dakotaInput.in', applicable for dakota method
        'offset' = [0, 1, 2, Noffset-1]
        'Noffset' = 'number of starting directions to consider'

    Returns:
        Writes a json file 'record.json' with the run information.
    """

    ### For visualization purposes. Set up the file that specifies the points for the polynomial approximation ###
    # Technically I should also check if I am running dakota with PC and not sampling. Just don't run sampling with verbose option
    if method_dict['method'] == 'dakota' and method_dict['verbose']:
        approximate.generate_approx_file(method_dict['uncertain_var'])

    # Create dakota files for the high and low fidelity
    High = 0
    Low = 1
    dakota_file = method_dict['dakota_filename']
    dakota_file_high = dakota_file + method_dict['file_extensions'][High]
    dakota_file_low = dakota_file + method_dict['file_extensions'][Low]
    shutil.copy(dakota_file, dakota_file_high)
    shutil.copy(dakota_file, dakota_file_low)

    ### Set up the wind speeds and wind directions for the problem ###

    # The high fidelity points
    method_dict['dakota_filename'] = dakota_file_high  # Update the input dakota file
    points = windfarm_setup.getPoints(method_dict, n_high)
    method_dict['dakota_filename'] = dakota_file  # Discard the Update of the input dakota file
    winddirections_high = points['winddirections']
    windspeeds_high = points['windspeeds']
    weights_high = points['weights']  # This might be None depending on the method.
    N_high = winddirections_high.size  # actual number of samples

    print 'Locations at which the HIGH-fidelity model is evaluated for power'
    print '\twindspeed \t winddirection'
    for i in range(N_high):
        print i+1, '\t', '%.2f' % windspeeds_high[i], '\t', '%.2f' % winddirections_high[i]

    # The low fidelity points
    method_dict['dakota_filename'] = dakota_file_low  # Update the input dakota file
    points = windfarm_setup.getPoints(method_dict, n_low)
    method_dict['dakota_filename'] = dakota_file  # Discard the Update of the input dakota file
    winddirections_low = points['winddirections']
    windspeeds_low = points['windspeeds']
    weights_low = points['weights']  # This might be None depending on the method.
    N_low = winddirections_low.size  # actual number of samples

    print 'Locations at which the LOW-fidelity model is evaluated for power'
    print '\twindspeed \t winddirection'
    for i in range(N_low):
        print i+1, '\t', '%.2f' % windspeeds_low[i], '\t', '%.2f' % winddirections_low[i]

    # Turbines layout
    turbineX, turbineY = windfarm_setup.getLayout(method_dict['layout'])

    # initialize problem
    prob = Problem(AEPGroupMulti(nTurbines=turbineX.size, nDirectionsHigh=N_high, nDirectionsLow=N_low, method_dict=method_dict))

    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeedsHigh'] = windspeeds_high
    prob['windDirectionsHigh'] = winddirections_high
    prob['windWeightsHigh'] = weights_high

    prob['windSpeedsLow'] = windspeeds_low
    prob['windDirectionsLow'] = winddirections_low
    prob['windWeightsLow'] = weights_low

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    # Run the problem
    prob.pre_run_check()
    prob.run()

    # Run the gradient
    # J = prob.calc_gradient(['turbineX', 'turbineY'], ['mean'], return_format='dict')
    # JacobianX = J['mean']['turbineX']
    # JacobianY = J['mean']['turbineY']
    # JacobianX = JacobianX.flatten()
    # JacobianY = JacobianY.flatten()
    # gradient = np.concatenate((JacobianX, JacobianY))
    # print 'dmean/dturbineX'
    # print JacobianX
    # print 'dmean/dturbineY'
    # print JacobianY
    # gradientfile = 'gradient.txt'
    # np.savetxt(gradientfile, gradient)
    #
    # fig, ax = plt.subplots()
    # ax.plot(gradient, label='gradient')
    # ax.legend()
    # plt.show()

    # For visualization purposes. Get the PC approximation
    # Technically I should also check if I am running dakota with PC and not sampling. Just don't run sampling with verbose option
    if method_dict['method'] == 'dakota' and method_dict['verbose']:
        winddirections_approx, windspeeds_approx, power_approx = approximate.get_approximation(method_dict)
    else:
        winddirections_approx = np.array([None])
        windspeeds_approx = np.array([None])
        power_approx = np.array([None])

    # print the results
    mean_data = prob['mean']
    std_data = prob['std']
    factor = 1e6
    print 'mean = ', mean_data/factor, ' GWhrs'
    print 'std = ', std_data/factor, ' GWhrs'
    power = prob['PowersCorr']
    print 'powers = ', power

    # What I'm returning needs to be updated. The approximation file has to be updated as well for multifidelity.
    N = N_high
    winddirections = winddirections_high
    windspeeds = windspeeds_high
    return mean_data/factor, std_data/factor, N, winddirections, windspeeds, power,\
           winddirections_approx, windspeeds_approx, power_approx


def plot():
    jsonfile = open('record.json','r')
    a = json.load(jsonfile)
    jsonfile.close()
    #print a
    print a.keys()
    # print json.dumps(a, indent=2)

    fig, ax = plt.subplots()
    ax.plot(a['winddirections'], a['power'])
    # ax.plot(a['windspeeds'], a['power'])
    ax.set_xlabel('wind directions (deg)')
    ax.set_ylabel('power')

    fig, ax = plt.subplots()
    ax.plot(a['samples'], a['mean'])
    ax.set_xlabel('Number of Wind Directions')
    ax.set_ylabel('mean annual energy production')
    ax.set_title('Mean annual energy as a function of the Number of Wind Directions')

    plt.show()


def get_args():
    parser = argparse.ArgumentParser(description='Run statistics convergence')
    parser.add_argument('--windspeed_ref', default=8, type=float, help='the wind speed for the wind direction case')
    parser.add_argument('--winddirection_ref', default=225, type=float, help='the wind direction for the wind speed case')
    parser.add_argument('-l', '--layout', default='optimized', help="specify layout: 'amalia', 'optimized', 'grid', 'random', 'test', 'local'")
    parser.add_argument('--offset', default=0, type=int, help='offset for starting direction. offset=[0, 1, 2, Noffset-1]')
    parser.add_argument('--Noffset', default=10, type=int, help='number of starting directions to consider')
    parser.add_argument('-n', '--nSamples', default=5, type=int, help='n is roughly a surrogate for the number of samples')
    parser.add_argument('--method', default='rect', help="specify method: 'rect', 'dakota'")
    parser.add_argument('--wake_model', default='floris', help="specify model: 'floris', 'jensen', 'gauss'")
    parser.add_argument('--uncertain_var', default='direction', help="specify uncertain variable: 'direction', 'speed', 'direction_and_speed'")
    parser.add_argument('--coeff_method', default='quadrature', help="specify coefficient method for dakota: 'quadrature', 'regression'")
    parser.add_argument('--dirdistribution', default='amaliaRaw', help="specify the desired distribution for the wind direction: 'amaliaModified', 'amaliaRaw', 'Uniform'")
    parser.add_argument('--gradient', action='store_true', help='Compute the power vs design variable gradient. Otherwise return None')
    parser.add_argument('--analytic_gradient', action='store_true', help='Compute gradient analytically (Only Floris), otherwise compute gradient by fd')
    parser.add_argument('--verbose', action='store_true', help='Includes results for every run in the output json file')
    parser.add_argument('--version', action='version', version='Statistics convergence 0.0')
    args = parser.parse_args()
    # print args
    # print args.offset
    return args

if __name__ == "__main__":

    # Get arguments
    args = get_args()
    verbose = args.verbose

    # Specify the rest of arguments
    # method_dict = {}
    method_dict = vars(args)  # Start a dictionary with the arguments specified in the command line
    method_dict['method']           = 'dakota'
    # select model: floris, jensen, gauss, larsen (larsen not working yet) TODO get larsen model working
    method_dict['wake_model']       = ['floris', 'jensen']  # High, low
    method_dict['uncertain_var']    = 'direction'
    # method_dict['layout']         = 'optimized'  # Now this is specified in the command line
    # method_dict['dakota_filename']  = 'dakotageneral.in'
    method_dict['dakota_filename']  = 'dakotageneralPy.in'  # Interface with python support
    # To Do specify the number of points (directions or speeds) as an option as well.
    method_dict['coeff_method']     = 'quadrature'
    method_dict['file_extensions']  = ['.high', '.low']  # These are used for correctly setting name for dakota file in multifidelity

    # Specify the distribution according to the uncertain variable
    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose(method_dict['dirdistribution'])
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction_and_speed':
        dist1 = distributions.getWindRose(method_dict['dirdistribution'])
        dist2 = distributions.getWeibull()
        dist = cp.J(dist1, dist2)
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed", "direction" or "direction_and_speed".' %method_dict['uncertain_var'])

    # Run the problem multiple times for statistics convergence
    mean = []
    std = []
    samples = []
    if verbose:
        winddir = []
        windspeed = []
        power = []
        power_approx = []

    tic = time.time()
    # Depending on the case n can represent number of quadrature points, sparse grid level, expansion order
    # n is roughly a surrogate for the number of samples
    n_low = 6
    for n_high in range(5, 6, 1):

        # Run the problem
        mean_data, std_data, N, winddirections, windspeeds, powers, \
        winddirections_approx, windspeeds_approx, powers_approx \
            = run(method_dict, n_high, n_low)
        mean.append(mean_data)
        std.append(std_data)
        samples.append(N)
        if verbose:
            winddir.append(winddirections.tolist())
            windspeed.append(windspeeds.tolist())
            power.append(powers.tolist())
            power_approx.append(powers_approx.tolist())

        # Save a record of the run
        if verbose:
            obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddir,
                'windspeeds': windspeed, 'power': power,
                'winddirections_approx': winddirections_approx.tolist(),
                'windspeeds_approx': windspeeds_approx.tolist(),
                'power_approx': power_approx,
                'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
                'layout': method_dict['layout'], 'wake_model': method_dict['wake_model'],
                'Noffset': method_dict['Noffset'], 'offset': method_dict['offset']}
        else:
            obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddirections.tolist(),
                'windspeeds': windspeeds.tolist(), 'power': powers.tolist(),
                'winddirections_approx': winddirections_approx.tolist(),
                'windspeeds_approx': windspeeds_approx.tolist(),
                'power_approx': powers_approx.tolist(),
                'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
                'layout': method_dict['layout'], 'wake_model': method_dict['wake_model'],
                'Noffset': method_dict['Noffset'], 'offset': method_dict['offset']}
        jsonfile = open('record.json', 'w')
        json.dump(obj, jsonfile, indent=2)
        jsonfile.close()

    toc = time.time()
    print 'Statistics Convergence Multifidelity took %.03f sec.' % (toc-tic)

    # plot()
