
import numpy as np
# import matplotlib.pyplot as plt
import json
import argparse
import chaospy as cp
from openmdao.api import Problem
from AEPGroups import AEPGroup
import distributions
import windfarm_setup


def run(method_dict, n):
    """
    method_dict = {}
    keys of method_dict:
        'method' = 'dakota', 'rect' or 'chaospy'  # 'chaospy needs updating
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


    ### Set up the wind speeds and wind directions for the problem ###

    points = windfarm_setup.getPoints(method_dict, n)
    winddirections = points['winddirections']
    windspeeds = points['windspeeds']
    weights = points['weights']  # This might be None depending on the method.
    N = winddirections.size  # actual number of samples

    print 'Locations at which power is evaluated'
    print '\twindspeed \t winddirection'
    for i in range(N):
        print i+1, '\t', '%.2f' % windspeeds[i], '\t', '%.2f' % winddirections[i]

    # Turbines layout
    turbineX, turbineY = windfarm_setup.getLayout(method_dict['layout'])

    # turbine size and operating conditions

    rotor_diameter = 126.4  # (m)
    air_density = 1.1716    # kg/m^3

    # initialize arrays for each turbine properties
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generator_efficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(nTurbs):
        rotorDiameter[turbI] = rotor_diameter
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generator_efficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # initialize problem
    prob = Problem(AEPGroup(nTurbines=nTurbs, nDirections=N,
                            method_dict=method_dict))
    prob.setup(check=False)

    # assign initial values to variables
    prob['windSpeeds'] = windspeeds
    prob['windDirections'] = winddirections
    prob['weights'] = weights
    prob['rotorDiameter'] = rotorDiameter
    prob['axialInduction'] = axialInduction
    prob['generatorEfficiency'] = generator_efficiency
    prob['air_density'] = air_density
    prob['Ct_in'] = Ct
    prob['Cp_in'] = Cp

    prob['turbineX'] = turbineX
    prob['turbineY'] = turbineY
    for direction_id in range(0, N):
        prob['yaw%i' % direction_id] = yaw

    # Run the problem
    prob.run()

    # print the results
    mean_data = prob['mean']
    std_data = prob['std']
    factor = 1e6
    print 'mean = ', mean_data/factor, ' GWhrs'
    print 'std = ', std_data/factor, ' GWhrs'
    power = prob['power']

    return mean_data/factor, std_data/factor, N, winddirections, windspeeds, power


def plot():
    jsonfile = open('record.json','r')
    a = json.load(jsonfile)
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
    parser.add_argument('-l', '--layout', default='optimized', help="specify layout ['amalia', 'optimized', 'grid', 'random', 'test']")
    parser.add_argument('--offset', default=0, type=int, help='offset for starting direction. offset=[0, 1, 2, Noffset-1]')
    parser.add_argument('--Noffset', default=10, type=int, help='number of starting directions to consider')
    parser.add_argument('--version', action='version', version='Statistics convergence 0.0')
    args = parser.parse_args()
    # print args
    # print args.offset
    # print args.Noffset
    # print args.layout
    return args

if __name__ == "__main__":

    # Get arguments
    args = get_args()

    # Specify the rest of arguments
    # method_dict = {}
    method_dict = vars(args)  # Start a dictionary with the arguments specified in the command line
    method_dict['method']           = 'dakota'
    method_dict['uncertain_var']    = 'direction'
    # method_dict['layout']           = 'optimized'  # Now this is specified in the command line
    method_dict['dakota_filename'] = 'dakotageneral.in'
    # To Do specify the number of points (directions or speeds) as an option as well.
    method_dict['coeff_method'] = 'quadrature'

    # Specify the distribution according to the uncertain variable
    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction_and_speed':
        dist1 = distributions.getWindRose()
        dist2 = distributions.getWeibull()
        dist = cp.J(dist1, dist2)
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    # Run the problem multiple times for statistics convergence
    mean = []
    std = []
    samples = []

    # Depending on the case n can represent number of quadrature points, sparse grid level, expansion order
    # n is roughly a surrogate for the number of samples
    for n in range(5, 6, 1):

        # Run the problem
        mean_data, std_data, N, winddirections, windspeeds, power = run(method_dict, n)
        mean.append(mean_data)
        std.append(std_data)
        samples.append(N)

    # Save a record of the run

    obj = {'mean': mean, 'std': std, 'samples': samples, 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    jsonfile = open('record.json','w')
    json.dump(obj, jsonfile, indent=2)
    jsonfile.close()

    # plot()
