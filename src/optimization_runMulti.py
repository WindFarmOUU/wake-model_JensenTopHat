from openmdao.api import Problem, pyOptSparseDriver, SqliteRecorder
from OptimizationGroup import OptAEPMulti
from wakeexchange.GeneralWindFarmComponents import calculate_boundary

import time
import numpy as np
import matplotlib.pyplot as plt
import chaospy as cp
import json
import shutil
import argparse
import windfarm_setup
import distributions


def get_args():
    parser = argparse.ArgumentParser(description='Run optimization')
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
    parser.add_argument('--version', action='version', version='Layout optimization 0.0')
    args = parser.parse_args()
    # print args
    # print args.offset
    return args


if __name__ == "__main__":

    # Get arguments
    args = get_args()

    # Specify the rest of arguments and overwrite some if desired
    method_dict = vars(args)  # Start a dictionary with the arguments specified in the command line
    method_dict['method']           = 'dakota'
    method_dict['uncertain_var']    = 'direction'
    # select model: floris, jensen, gauss, larsen (larsen not working yet) TODO get larsen model working
    method_dict['wake_model']       = ['floris', 'jensen']  # High, low
    # method_dict['dakota_filename']  = 'dakotageneral.in'
    method_dict['dakota_filename']  = 'dakotageneralPy.in'  # Interface with python support
    method_dict['coeff_method']     = 'quadrature'
    method_dict['file_extensions']  = ['.high', '.low']  # These are used for correctly setting name for dakota file in multifidelity
    method_dict['gradient'] = True  # We are running optimization, so have this set to true.

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

    # Create dakota files for the high and low fidelity
    High = 0
    Low = 1
    dakota_file = method_dict['dakota_filename']
    dakota_file_high = dakota_file + method_dict['file_extensions'][High]
    dakota_file_low = dakota_file + method_dict['file_extensions'][Low]
    shutil.copy(dakota_file, dakota_file_high)
    shutil.copy(dakota_file, dakota_file_low)

    ### Set up the wind speeds and wind directions for the problem ###
    n = method_dict['nSamples']  # n is roughly a surrogate for the number of samples
    n_high = 5
    n_low = 10

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
    nTurbs = turbineX.size

    # generate boundary constraint
    locations = np.column_stack((turbineX, turbineY))  # Uses the starting layout boundary
    # locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')  # Pick the desired layout for the boundary
    boundaryVertices, boundaryNormals = calculate_boundary(locations)
    nVertices = boundaryVertices.shape[0]
    print 'boundary vertices', boundaryVertices

    minSpacing = 2.                         # number of rotor diameters

    # initialize problem
    prob = Problem(root=OptAEPMulti(nTurbines=nTurbs, nDirectionsHigh=N_high, nDirectionsLow=N_low, minSpacing=minSpacing, nVertices=nVertices, method_dict=method_dict))

    # set up optimizer
    # Scale everything (variables, objective, constraints) to make order 1.
    diameter = 126.4  # meters, used in the scaling
    prob.driver = pyOptSparseDriver()
    prob.driver.options['optimizer'] = 'SNOPT'
    prob.driver.add_objective('obj', scaler=1E-8)  # For 2d 1E-9

    # set optimizer options
    prob.driver.opt_settings['Verify level'] = -1  # 3
    prob.driver.opt_settings['Print file'] = 'SNOPT_print_exampleOptAEP.out'
    prob.driver.opt_settings['Summary file'] = 'SNOPT_summary_exampleOptAEP.out'
    prob.driver.opt_settings['Major iterations limit'] = 300
    prob.driver.opt_settings['Major optimality tolerance'] = 1E-4
    prob.driver.opt_settings['Major feasibility tolerance'] = 1E-4
    prob.driver.opt_settings['Minor feasibility tolerance'] = 1E-4
    prob.driver.opt_settings['Function precision'] = 1E-5

    # select design variables
    prob.driver.add_desvar('turbineX', adder=-turbineX, scaler=1.0/diameter)
    prob.driver.add_desvar('turbineY', adder=-turbineY, scaler=1.0/diameter)
    # for direction_id in range(0, N):
    #     prob.driver.add_desvar('yaw%i' % direction_id, lower=-30.0, upper=30.0, scaler=1.0)

    # add constraints
    prob.driver.add_constraint('sc', lower=np.zeros(((nTurbs-1.)*nTurbs/2.)), scaler=1.0/((20*diameter)**2))
    prob.driver.add_constraint('boundaryDistances', lower=np.zeros(nVertices*nTurbs), scaler=1.0/(10*diameter))

    # Reduces time of computation
    prob.root.ln_solver.options['single_voi_relevance_reduction'] = True

    # Set up a recorder
    recorder = SqliteRecorder('optimization.sqlite')
    # recorder.options['record_params'] = True
    # recorder.options['record_metadata'] = True
    prob.driver.add_recorder(recorder)

    tic = time.time()
    prob.setup(check=False)
    toc = time.time()

    # print the results
    print 'Optimization setup took %.03f sec.' % (toc-tic)

    # assign initial values to variables
    prob['windSpeedsHigh'] = windspeeds_high
    prob['windDirectionsHigh'] = winddirections_high
    prob['windWeightsHigh'] = weights_high

    prob['windSpeedsLow'] = windspeeds_low
    prob['windDirectionsLow'] = winddirections_low
    prob['windWeightsLow'] = weights_low

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY

    # provide values for the hull constraint
    prob['boundaryVertices'] = boundaryVertices
    prob['boundaryNormals'] = boundaryNormals

    # run the problem
    print 'start Optimization run'
    tic = time.time()
    prob.run()
    toc = time.time()

    prob.cleanup()  # this closes all recorders

    # print the results
    print 'Optimization calculation took %.03f sec.' % (toc-tic)

    print 'turbine X positions in wind frame (m): %s' % prob['turbineX']
    print 'turbine Y positions in wind frame (m): %s' % prob['turbineY']
    print 'wind farm power in each direction (kW): %s' % prob['PowersCorr']
    print 'AEP (GWh): ', prob['mean']/1e6

    np.savetxt('layout_local.txt', np.c_[prob['turbineX'], prob['turbineY']], header="turbineX, turbineY")

    # Think if I want to do something besides setting the high.
    N = N_high
    winddirections = winddirections_high
    windspeeds = windspeeds_high

    # Save details of the simulation
    obj = {'mean': prob['mean']/1e6, 'std': prob['std']/1e6, 'samples': N, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': prob['PowersCorr'].tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout'], 'turbineX': turbineX.tolist(), 'turbineY': turbineY.tolist(),
           'turbineXopt': prob['turbineX'].tolist(), 'turbineYopt': prob['turbineY'].tolist()}
    jsonfile = open('record_opt.json', 'w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()

    plt.figure()
    plt.plot(turbineX, turbineY, 'ok', label='Original')
    plt.plot(prob['turbineX'], prob['turbineY'], 'og', label='Optimized')
    for i in range(0, nTurbs):
        plt.plot([turbineX[i], prob['turbineX'][i]], [turbineY[i], prob['turbineY'][i]], '--k')
    plt.legend()
    plt.xlabel('Turbine X Position (m)')
    plt.ylabel('Turbine Y Position (m)')
    plt.show()
