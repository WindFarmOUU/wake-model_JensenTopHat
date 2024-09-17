import numpy as np
import sys

from openmdao.api import Group, IndepVarComp
from statisticsComponents import DakotaStatistics, RectStatistics, ChaospyStatistics, DakotaStatisticsMulti


class AEPGroup(Group):
    """
    Group containing all necessary components for wind plant AEP calculations
    """

    def __init__(self, nTurbines, nDirections=1, method_dict=None):

        super(AEPGroup, self).__init__()

        # providing default unit types
        direction_units = 'deg'
        wind_speed_units = 'm/s'
        length_units = 'm'

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirections', np.zeros(nDirections), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeeds', np.zeros(nDirections), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windWeights', np.zeros(nDirections)), promotes=['*'])

        self.add('dv3', IndepVarComp('turbineX', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv4', IndepVarComp('turbineY', np.zeros(nTurbines), units=length_units), promotes=['*'])

        method = method_dict['method']
        if method == 'dakota':
            self.add('AEPcomp', DakotaStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        elif method == 'chaospy':
            self.add('AEPcomp', ChaospyStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        elif method == 'rect':
            self.add('AEPcomp', RectStatistics(nTurbines, nDirections, method_dict), promotes=['*'])
        else:
            print "Specify one of these UQ methods = ['dakota', 'chaospy', 'rect']"
            sys.exit()


class AEPGroupMulti(Group):
    """
    Group containing all necessary components for multi-fidelity wind plant AEP calculations
    """

    def __init__(self, nTurbines, nDirectionsHigh=1, nDirectionsLow=1, method_dict=None):

        super(AEPGroupMulti, self).__init__()

        # providing default unit types
        direction_units = 'deg'
        wind_speed_units = 'm/s'
        length_units = 'm'

        # add necessary inputs for group
        self.add('dv0', IndepVarComp('windDirectionsHigh', np.zeros(nDirectionsHigh), units=direction_units), promotes=['*'])
        self.add('dv1', IndepVarComp('windSpeedsHigh', np.zeros(nDirectionsHigh), units=wind_speed_units), promotes=['*'])
        self.add('dv2', IndepVarComp('windWeightsHigh', np.zeros(nDirectionsHigh)), promotes=['*'])

        self.add('dv3', IndepVarComp('windDirectionsLow', np.zeros(nDirectionsLow), units=direction_units), promotes=['*'])
        self.add('dv4', IndepVarComp('windSpeedsLow', np.zeros(nDirectionsLow), units=wind_speed_units), promotes=['*'])
        self.add('dv5', IndepVarComp('windWeightsLow', np.zeros(nDirectionsLow)), promotes=['*'])

        self.add('dv6', IndepVarComp('turbineX', np.zeros(nTurbines), units=length_units), promotes=['*'])
        self.add('dv7', IndepVarComp('turbineY', np.zeros(nTurbines), units=length_units), promotes=['*'])

        method = method_dict['method']
        if method == 'dakota':
            self.add('AEPcomp', DakotaStatisticsMulti(nTurbines, nDirectionsHigh, nDirectionsLow, method_dict), promotes=['*'])
        else:
            raise ValueError('Only "dakota" keyword for method implemented for multi-fidelity.\nReceived method keyword "%s"' %method)
