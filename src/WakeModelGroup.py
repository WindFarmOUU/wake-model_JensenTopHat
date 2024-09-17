import numpy as np

from openmdao.api import Group, IndepVarComp, ParallelGroup

from wakeexchange.GeneralWindFarmComponents import MUX, DeMUX, add_gen_params_IdepVarComps
from wakeexchange.GeneralWindFarmGroups import DirectionGroup
from wakeexchange.floris import floris_wrapper, add_floris_params_IndepVarComps


class WakeModelGroup(Group):
    """
    Group containing all necessary components for evaluation of the wake model
    """

    def __init__(self, nTurbines, nDirections=1, use_rotor_components=False, datasize=0,
                 differentiable=True, optimizingLayout=False, nSamples=0,
                 wake_model=floris_wrapper, wake_model_options=None,
                 params_IdepVar_func=add_floris_params_IndepVarComps, params_IndepVar_args=None, analytic_gradient=False):

        super(WakeModelGroup, self).__init__()

        # Finite difference for all wake models except Floris. Floris can be analytic or fd
        if wake_model is not floris_wrapper or not analytic_gradient:
            # Check derivatives
            self.deriv_options['type'] = 'fd'
            self.deriv_options['form'] = 'forward'
            self.deriv_options['step_size'] = 1.0e-5

        if wake_model_options is None:
            wake_model_options = {'differentiable': differentiable, 'use_rotor_components': use_rotor_components,
                             'nSamples': nSamples, 'verbose': False}

        # providing default unit types for general MUX/DeMUX components
        power_units = 'kW'
        direction_units = 'deg'
        wind_speed_units = 'm/s'
        length_units = 'm'

        # print 'SAMPLES: ', nSamples

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windWeights', np.zeros(nDirections)), promotes=['*'])

        self.add('dv3', IndepVarComp('turbineX', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv4', IndepVarComp('turbineY', np.zeros(nTurbines), units=length_units), promotes=['*'])

        # add vars to be seen by MPI and gradient calculations
        self.add('dv5', IndepVarComp('rotorDiameter', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv6', IndepVarComp('axialInduction', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv7', IndepVarComp('generatorEfficiency', np.zeros(nTurbines)), promotes=['*'])
        self.add('dv8', IndepVarComp('air_density', val=1.1716, units='kg/(m*m*m)'), promotes=['*'])

        # Can probably simplify the below rotor components logic
        if not use_rotor_components:
            self.add('dv9', IndepVarComp('Ct_in', np.zeros(nTurbines)), promotes=['*'])
            self.add('dv10', IndepVarComp('Cp_in', np.zeros(nTurbines)), promotes=['*'])

        # add variable tree IndepVarComps
        add_gen_params_IdepVarComps(self, datasize=datasize)

        # indep variable components for wake model
        if params_IdepVar_func is not None:
            if (params_IndepVar_args is None) and (wake_model is floris_wrapper):
                params_IndepVar_args = {'use_rotor_components': False}
            elif params_IndepVar_args is None:
                params_IndepVar_args = {}
            params_IdepVar_func(self, **params_IndepVar_args)

        # add components and groups
        self.add('windDirectionsDeMUX', DeMUX(nDirections, units=direction_units))
        self.add('windSpeedsDeMUX', DeMUX(nDirections, units=wind_speed_units))

        pg = self.add('all_directions', ParallelGroup(), promotes=['*'])

        #The if nSamples == 0 is left in for visualization
        if use_rotor_components:
            for direction_id in np.arange(0, nDirections):
                # print 'assigning direction group %i' % direction_id
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples),
                       promotes=(['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id]
                                 if (nSamples == 0) else
                                 ['gen_params:*', 'model_params:*', 'air_density',
                                  'axialInduction', 'generatorEfficiency', 'turbineX', 'turbineY', 'hubHeight',
                                  'yaw%i' % direction_id, 'rotorDiameter', 'wsPositionX', 'wsPositionY',
                                  'wsPositionZ', 'wtVelocity%i' % direction_id,
                                  'wtPower%i' % direction_id, 'dir_power%i' % direction_id, 'wsArray%i' % direction_id]))
        else:
            for direction_id in np.arange(0, nDirections):
                # print 'assigning direction group %i' % direction_id
                pg.add('direction_group%i' % direction_id,
                       DirectionGroup(nTurbines=nTurbines, direction_id=direction_id,
                                      use_rotor_components=use_rotor_components, datasize=datasize,
                                      differentiable=differentiable, add_IdepVarComps=False, nSamples=nSamples,
                                      wake_model=wake_model, wake_model_options=wake_model_options),
                       promotes=(['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight', 'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id]
                                 if (nSamples == 0) else
                                 ['Ct_in', 'Cp_in', 'gen_params:*', 'model_params:*', 'air_density', 'axialInduction',
                                  'generatorEfficiency', 'turbineX', 'turbineY', 'yaw%i' % direction_id, 'rotorDiameter',
                                  'hubHeight',  'wsPositionX', 'wsPositionY', 'wsPositionZ',
                                  'wtVelocity%i' % direction_id, 'wtPower%i' % direction_id,
                                  'dir_power%i' % direction_id, 'wsArray%i' % direction_id]))

        self.add('powerMUX', MUX(nDirections, units=power_units))

        # connect components
        self.connect('windDirections', 'windDirectionsDeMUX.Array')
        self.connect('windSpeeds', 'windSpeedsDeMUX.Array')
        for direction_id in range(0, nDirections):
            self.add('y%i' % direction_id, IndepVarComp('yaw%i' % direction_id, np.zeros(nTurbines), units=direction_units), promotes=['*'])
            self.connect('windDirectionsDeMUX.output%i' % direction_id, 'direction_group%i.wind_direction' % direction_id)
            self.connect('windSpeedsDeMUX.output%i' % direction_id, 'direction_group%i.wind_speed' % direction_id)
            self.connect('dir_power%i' % direction_id, 'powerMUX.input%i' % direction_id)
