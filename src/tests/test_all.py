# content of test_rectangle.py
import json
from statistics_convergence import run
import distributions

def assertions(test, baseline):
    assert test['samples'] == baseline['samples']
    assert test['method'] == baseline['method']
    assert test['uncertain_variable'] == baseline['uncertain_variable']
    assert test['mean'] == baseline['mean']
    assert test['std'] == baseline['std']
    assert test['power'] == baseline['power']
    assert test['winddirections'] == baseline['winddirections']
    assert test['windspeeds'] == baseline['windspeeds']
    assert test == baseline

def get_method_dict():
    method_dict = {'method': 'dakota',
                   'wake_model': 'floris',
                   'uncertain_var': 'direction',
                   'layout': 'optimized',
                   'offset': 0,
                   'Noffset': 10,
                   'dakota_filename': 'tests/dakotageneral.in',  # 'tests/dakotageneralPy.in'
                   'coeff_method': 'quadrature',
                   'windspeed_ref': 8,
                   'winddirection_ref': 225,
                   'dirdistribution': 'amaliaModified',
                   'gradient': False,
                   'analytic_gradient': False,
                   'verbose': False}
    return method_dict


def add_distribution(method_dict):
    # Specify the distribution according to the uncertain variable
    if method_dict['uncertain_var'] == 'speed':
        dist = distributions.getWeibull()
        method_dict['distribution'] = dist
    elif method_dict['uncertain_var'] == 'direction':
        dist = distributions.getWindRose(method_dict['dirdistribution'])
        method_dict['distribution'] = dist
    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])
    return method_dict


##### TESTS #####
def test_dakota_direction_expansion():
    n = 5
    method_dict = get_method_dict()
    method_dict['dakota_filename'] = 'tests/dakotagenerale.in'
    method_dict['coeff_method'] = 'regression'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_dakota_direction_expansion.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_dakota_direction_quadrature_offset1():
    n = 5
    method_dict = get_method_dict()
    method_dict['offset'] = 1
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_dakota_direction_quadrature_offset1.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_dakota_direction_quadrature():
    n = 5
    method_dict = get_method_dict()
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_dakota_direction_quadrature.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_dakota_direction_sparse():
    n = 1
    method_dict = get_method_dict()
    method_dict['coeff_method'] = 'sparse_grid'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_dakota_direction_sparse.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_dakota_speed_quadrature():
    n = 5
    method_dict = get_method_dict()
    method_dict['uncertain_var'] = 'speed'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_dakota_speed_quadrature.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_chaospy_speed_quadrature():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'chaospy'
    method_dict['uncertain_var'] = 'speed'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_chaospy_speed_quadrature.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction_30points():
    n = 30
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction_30points.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction_amalia():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict['layout'] = 'amalia'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction_amalia.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction_grid():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict['layout'] = 'grid'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction_grid.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction_offset1():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict['offset'] = 1
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction_offset1.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction_random():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict['layout'] = 'random'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction_random.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_direction():  # This is for the optimized layout
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_direction.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


def test_rect_speed():
    n = 5
    method_dict = get_method_dict()
    method_dict['method'] = 'rect'
    method_dict['uncertain_var'] = 'speed'
    method_dict = add_distribution(method_dict)

    jsonfile = open('tests/record_test_rect_speed.json','r')
    baseline = json.load(jsonfile)
    jsonfile.close()
    mean, std, N, winddirections, windspeeds, power, \
        winddirections_approx, windspeeds_approx, power_approx \
            = run(method_dict, n)
    obj = {'mean': [mean], 'std': [std], 'samples': [N], 'winddirections': winddirections.tolist(),
           'windspeeds': windspeeds.tolist(), 'power': power.tolist(),
           'method': method_dict['method'], 'uncertain_variable': method_dict['uncertain_var'],
           'layout': method_dict['layout']}
    test = obj
    assertions(test, baseline)


if __name__ == "__main__":
    test_dakota_direction_expansion()  # works if run from the src directory
