
import numpy as np
from functools import wraps
from openmdao.api import Problem
from WakeModelGroup import WakeModelGroup
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


def make_hashable(*args):
    """Create a key of hashable types"""
    key = []
    for arg in args:
        if type(arg) is np.ndarray:
            key.append(arg.tostring())
        else:
            key.append(arg)
    key = tuple(key)

    return key


def memoize(function):
    cache = {}

    @wraps(function)  # Preserves the metadata of the wrapped function
    def wrapper(*args):
        key = make_hashable(*args)
        if key in cache:
            print "Returning cached powers"
            return cache[key]
        else:
            rv = function(*args)
            cache[key] = rv
            return rv
    return wrapper


@memoize
def getPower(turbineX, turbineY, windDirections, windSpeeds, windWeights, gradient, analytic_gradient, wake_model, IndepVarFunc):
    """Calls the wake model, which is an openMDAO problem

    returns: The power for each wind direction and wind speed pair.
             The gradient of the power for each wind direction and wind speed pair
                with respect to the turbineX and turbineY locations.
    """

    # define turbine size
    rotor_diameter = 126.4  # (m)
    # Define flow properties
    air_density = 1.1716    # kg/m^3

    # initialize input variable arrays
    nTurbs = turbineX.size
    rotorDiameter = np.zeros(nTurbs)
    axialInduction = np.zeros(nTurbs)
    Ct = np.zeros(nTurbs)
    Cp = np.zeros(nTurbs)
    generatorEfficiency = np.zeros(nTurbs)
    yaw = np.zeros(nTurbs)

    # define initial values
    for turbI in range(0, nTurbs):
        rotorDiameter[turbI] = rotor_diameter      # m
        axialInduction[turbI] = 1.0/3.0
        Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
        Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
        generatorEfficiency[turbI] = 0.944
        yaw[turbI] = 0.     # deg.

    # Two options.
    # Call wake model for one condition at a time or for a vector of conditions (wind directions, wind speeds, etc)
    # If vector is too large, maybe slowdowns or memory issues when running in serial.
    allAtOnce = True

    if allAtOnce:

        # initialize problem
        n = windDirections.size
        prob = Problem(WakeModelGroup(nTurbines=nTurbs, nDirections=n, wake_model=wake_model, params_IdepVar_func=IndepVarFunc, analytic_gradient=analytic_gradient))
        prob.setup(check=False)

        # assign initial values to design variables
        prob['turbineX'] = turbineX
        prob['turbineY'] = turbineY
        for direction_id in range(0, n):
            prob['yaw%i' % direction_id] = yaw

        # assign values to constant inputs (not design variables)
        prob['rotorDiameter'] = rotorDiameter
        prob['axialInduction'] = axialInduction
        prob['generatorEfficiency'] = generatorEfficiency
        prob['windSpeeds'] = windSpeeds
        prob['air_density'] = air_density
        prob['windDirections'] = windDirections
        prob['windWeights'] = windWeights
        prob['Ct_in'] = Ct
        prob['Cp_in'] = Cp

        prob.run()

        powers = prob['powerMUX.Array']

        # Compute the gradient of the power wrt to the turbine locations
        if gradient:
            J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict')
            # To set finite difference options--step size, form, do so directly in the WakeModelGroup
            # J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict', mode='fd')  # force fd mode with default step 1e-6 absolute
            JacobianX = J['powerMUX.Array']['turbineX']
            JacobianY = J['powerMUX.Array']['turbineY']

        else:
            JacobianX = np.array([None])
            JacobianY = np.array([None])

    else:  # One power evaluation at a time
        powers = np.zeros(windDirections.size)
        JacobianX = np.zeros((windDirections.size, nTurbs))
        JacobianY = np.zeros((windDirections.size, nTurbs))

        # Call only one at a time
        for j, (windDirection, windSpeed, windWeight) in enumerate(zip(windDirections, windSpeeds, windWeights)):

            # initialize problem
            n = windDirection.size
            prob = Problem(WakeModelGroup(nTurbines=nTurbs, nDirections=n, wake_model=wake_model, params_IdepVar_func=IndepVarFunc, analytic_gradient=analytic_gradient))
            prob.setup(check=False)

            # assign initial values to design variables
            prob['turbineX'] = turbineX
            prob['turbineY'] = turbineY
            for direction_id in range(0, n):
                prob['yaw%i' % direction_id] = yaw

            # assign values to constant inputs (not design variables)
            prob['rotorDiameter'] = rotorDiameter
            prob['axialInduction'] = axialInduction
            prob['generatorEfficiency'] = generatorEfficiency
            prob['windSpeeds'] = windSpeed
            prob['air_density'] = air_density
            prob['windDirections'] = windDirection
            prob['windWeights'] = windWeight
            prob['Ct_in'] = Ct
            prob['Cp_in'] = Cp

            prob.run()

            powers[j] = prob['powerMUX.Array']

            # For now nothing is done for the gradient with the loop
            # Compute the gradient of the power wrt to the turbine locations
            if gradient:
                J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict')
                # To set finite difference options--step size, form, do so directly in the WakeModelGroup
                # J = prob.calc_gradient(['turbineX', 'turbineY'], ['powerMUX.Array'], return_format='dict', mode='fd')  # force fd mode with default step 1e-6 absolute
                JacobianX[j] = J['powerMUX.Array']['turbineX']
                JacobianY[j] = J['powerMUX.Array']['turbineY']

            else:
                JacobianX = np.array([None])
                JacobianY = np.array([None])

    return powers, JacobianX, JacobianY


if __name__ == "__main__":

    # define turbine locations in global reference frame
    turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
    turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

    # Define flow properties
    wind_speed = 8.0        # m/s
    windDirections = np.linspace(0, 270, 4)
    n = windDirections.size
    windSpeeds = np.ones(n)*wind_speed
    windWeights = np.ones(n)/n

    wake_model = floris_wrapper
    IndepVarFunc = add_floris_params_IndepVarComps

    powers, JacobianX, JacobianY = getPower(turbineX, turbineY, windDirections, windSpeeds, windWeights,
                                            wake_model, IndepVarFunc)

    print 'powers = ', powers
    print 'dpower/dturbineX'
    print JacobianX
    print 'dpower/dturbineY'
    print JacobianY
    print JacobianY.shape

