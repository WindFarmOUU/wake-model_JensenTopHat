import numpy as np
# import matplotlib.pyplot as plt
from functools import wraps
import chaospy as cp
from getSamplePoints import getSamplePoints
from dakotaInterface import updateDakotaFile


def make_hashable(method_dict, n):
    """Create a key of hashable types"""
    m = method_dict
    # key = (n, m['layout'], m['Noffset'], m['offset'], m['analytic_gradient'], m['uncertain_var'],
    #        m['gradient'], m['distribution'], m['method'], m['wake_model'])  # The wake model returns a list for multifidelity
    key = (n, m['layout'], m['Noffset'], m['offset'], m['analytic_gradient'], m['uncertain_var'],
           m['gradient'], m['distribution'], m['method'])  # Remove the wake model
    return key


def memoize(function):
    cache = {}

    @wraps(function)  # Preserves the metadata of the wrapped function
    def wrapper(method_dict, n):
        key = make_hashable(method_dict, n)
        if key in cache:
            print "Returning cached points"
            return cache[key]
        else:
            rv = function(method_dict, n)
            cache[key] = rv
            return rv
    return wrapper


@memoize
def getPoints(method_dict, n):

    if method_dict['uncertain_var'] == 'direction':
        dist = method_dict['distribution']
        winddirections, weights = getPointsDirection(dist, method_dict, n)
        windspeeds = np.ones(winddirections.size)*method_dict['windspeed_ref']  # 8m/s
        points = {'winddirections': winddirections, 'windspeeds': windspeeds, 'weights': weights}

    elif method_dict['uncertain_var'] == 'speed':
        dist = method_dict['distribution']
        windspeeds, weights = getPointsSpeed(dist, method_dict, n)
        winddirections = np.ones(windspeeds.size)*method_dict['winddirection_ref']  # 225 deg
        points = {'winddirections': winddirections, 'windspeeds': windspeeds, 'weights': weights}

    elif method_dict['uncertain_var'] == 'direction_and_speed':
        dist = method_dict['distribution']
        winddirections, windspeeds, weights = getPointsDirectionSpeed(dist, method_dict, n)
        points = {'winddirections': winddirections, 'windspeeds': windspeeds, 'weights': weights}

    else:
        raise ValueError('unknown uncertain_var option "%s", valid options "speed" or "direction".' %method_dict['uncertain_var'])

    return points


def getPointsDirectionSpeed(dist, method_dict, n):

    method = method_dict['method']

    if method == 'rect':
        dist_dir = dist[0]
        dist_speed = dist[1]
        winddirections, weights_dir = getPointsDirection(dist_dir, method_dict, n)
        windspeeds, weights_speed = getPointsSpeed(dist_speed, method_dict, n)

        # Create a 1-dimensional vector of the tensor product
        wind_dir = []
        wind_spd = []
        weights = []
        for i in range(windspeeds.size):
            for j in range(winddirections.size):
                wind_dir.append(winddirections[j])
                wind_spd.append(windspeeds[i])
                weights.append(weights_dir[j]*weights_speed[i])

        winddirections = np.array(wind_dir)
        windspeeds = np.array(wind_spd)
        weights = np.array(weights)

    if method == 'dakota':

        bnd = dist.range()
        a = bnd[0]  # left boundary
        b = bnd[1]  # right boundary
        a_d = a[0] # the left boundary for the direction
        b_d = b[0] # the right boundary for the direction
        a_s = a[1] # the left boundary for the direction
        b_s = b[1] # the right boundary for the direction

        ###### Do direction work
        dist_dir = dist[0]

        if dist_dir._str() == 'Amalia windrose':

            # Make sure the A, B, C values are the same than those in distribution
            A, B = dist_dir.get_zero_probability_region()
            # A = 110  # Left boundary of zero probability region
            # B = 140  # Right boundary of zero probability region

            C = 225  # Location of max probability or desired starting location
            r = b_d-a_d  # original range
            R = r - (B-A) # modified range

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            # Modify the starting point C with offset
            offset = i*r/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % r
            x_d, f_d = generate_direction_abscissas_ordinates(a_d, A, B, C, r, R, dist_dir)

        if dist_dir._str() == 'Amalia windrose raw' or dist_dir._str() == 'Uniform(0.0, 360.0)':

            C = 0  # 225  # Location of max probability or desired starting location.
            R = b_d-a_d  # range 360

            # Modify with offset, manually choose the offset you want
            N = method_dict['Noffset']  # N = 10
            i = method_dict['offset']  # i = [0, 1, 2, N-1]

            offset = i*R/N  # the offset modifies the starting point for N locations within the whole interval
            C = (C + offset) % R

            # Use the y to set the abscissas, and the pdf to set the ordinates
            y = np.linspace(a_d, R, 51)  # play with the number here
            dy = y[1]-y[0]
            mid = y[:-1]+dy/2

            # Modify the mid to start from the max probability location
            ynew = (mid+C) % R

            f_d = dist_dir.pdf(ynew)

            # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
            x_d = 2*(y-a_d) / R - 1

        ####### Do the speed work
        dist_speed = dist[1]
        x_s, f_s = generate_speed_abscissas_ordinates(a_s, b_s, dist_speed)

        # Update the dakota file
        updateDakotaFile(method_dict, n, [x_d, x_s], [f_d, f_s])

        # run Dakota file to get the points locations
        # This one also needs to work for the 1 and 2d cases.
        x, w = getSamplePoints(method_dict['dakota_filename'])
        assert len(x) == 2, 'Should be returning the directions and speeds'
        x_d = np.array(x[0])
        x_s = np.array(x[1])

        # Do stuff for the direction case
        if dist_dir._str() == 'Amalia windrose':
            # Rescale x
            x_d = R*x_d/2. + R/2. + a_d  # R = 330
            # Call modify x with the new x.
            x_d = modifyx(x_d, A, B, C, r)
        if dist_dir._str() == 'Amalia windrose raw' or dist_dir._str() == 'Uniform(0.0, 360.0)':
            # Rescale x
            x_d = R*x_d/2. + R/2. + a_d  # R = 360
            # Call modify x with the new x.
            x_d = (x_d+C) % R

        # Do stuff for the speed case
        # Rescale x
        x_s = (b_s-a_s)/2. + (b_s-a_s)/2.*x_s + a_s

        winddirections = x_d
        windspeeds = x_s
        weights = w

    return winddirections, windspeeds, weights


def getPointsModifiedAmaliaDistribution(dist, method_dict, n):

    # Modify the input range to start at max probability location
    # and account for zero probability regions.

    # f(x)
    #   |                   *
    #   |   ***            * *      **
    #   | **   *          *   **  **  *     ***
    #   |*      *        *      **     *  **   *
    #   |        *      *               **
    #   |         *    *
    # --+----------****-----+------------------+--
    #  lo          A  B     C                  hi    (x)

    method = method_dict['method']
    bnd = dist.range()
    a = bnd[0]  # left boundary
    b = bnd[1]  # right boundary
    a = a[0]  # get rid of the list
    b = b[0]  # get rid of the list
    # Make sure the A, B, C values are the same than those in distribution
    A, B = dist.get_zero_probability_region()
    # A = 110  # Left boundary of zero probability region
    # B = 140  # Right boundary of zero probability region

    C = 225  # Location of max probability or desired starting location.  Don't put this between A and B.
    r = b-a  # original range
    R = r - (B-A) # modified range

    # Modify with offset, manually choose the offset you want
    N = method_dict['Noffset']  # N = 10
    i = method_dict['offset']  # i = [0, 1, 2, N-1]

    if method == 'rect':
        # the offset fits N points in the given dx interval
        dx = R/n
        offset = i*dx/N  # make sure this is float
        bounds = [a+offset, R+offset]
        x = np.linspace(bounds[0], bounds[1], n+1)
        x = x[:-1]+dx/2  # Take the midpoints of the bins
        # Modify x, to start from the max probability location
        x = modifyx(x, A, B, C, r)
        # Get the weights associated with the points locations
        # w = getWeights(x, dx, dist)
        w = getWeightsModifiedAmalia(x, dx, dist)

    if method == 'dakota':

        # Modify the starting point C with offset
        offset = i*r/N  # the offset modifies the starting point for N locations within the whole interval
        C = (C + offset) % r
        x, f = generate_direction_abscissas_ordinates(a, A, B, C, r, R, dist)
        updateDakotaFile(method_dict, n, x, f)
        # run Dakota file to get the points locations
        x, w = getSamplePoints(method_dict['dakota_filename'])
        assert len(x) == 1, 'Should only be returning the directions'
        x = np.array(x[0])
        # Rescale x
        x = R*x/2. + R/2. + a
        # x = (330/2. + 330/2.*x  # Should be in terms of the variables
        # Call modify x with the new x.
        x = modifyx(x, A, B, C, r)

    if method == 'chaospy':
        # I need to adjust the starting position and all of that.
        x, w = cp.generate_quadrature(n-1, dist, rule='G')
        x = x[0]

    return x, w


def getPointsUnmodifiedDistribution(dist, method_dict, n):

    method = method_dict['method']
    bnd = dist.range()
    a = bnd[0]  # left boundary
    b = bnd[1]  # right boundary
    a = a[0]  # get rid of the list
    b = b[0]  # get rid of the list

    C = 0  # 225  # Location of max probability or desired starting location.
    R = b-a  # range 360

    # Modify with offset, manually choose the offset you want
    N = method_dict['Noffset']  # N = 10
    i = method_dict['offset']  # i = [0, 1, 2, N-1]

    if method == 'rect':
        # the offset fits N points in the given dx interval
        dx = R/n
        offset = i*dx/N  # make sure this is float
        bounds = [a+offset, R+offset]
        x = np.linspace(bounds[0], bounds[1], n+1)
        x = x[:-1]+dx/2  # Take the midpoints of the bins
        # Modify x, to start from the max probability location
        x = (x+C) % R
        # Get the weights associated with the points locations
        w = getWeights(x, dx, dist)

    if method == 'dakota':

        # Modify the starting point C with offset
        offset = i*R/N  # the offset modifies the starting point for N locations within the whole interval
        C = (C + offset) % R
        # Use the y to set the abscissas, and the pdf to set the ordinates
        y = np.linspace(a, R, 51)  # play with the number here
        dy = y[1]-y[0]
        mid = y[:-1]+dy/2

        # Modify the mid to start from the max probability location
        ynew = (mid+C) % R
        f = dist.pdf(ynew)

        # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
        x = 2*(y-a) / R - 1

        updateDakotaFile(method_dict, n, x, f)
        # run Dakota file to get the points locations
        x, w = getSamplePoints(method_dict['dakota_filename'])
        assert len(x) == 1, 'Should only be returning the directions'
        x = np.array(x[0])
        # Rescale x
        x = R*x/2. + R/2. + a

        # Call modify x with the new x.
        x = (x+C) % R

    if method == 'chaospy':
        # I need to adjust the starting position and all of that.
        x, w = cp.generate_quadrature(n-1, dist, rule='G')
        x = x[0]

    return x, w


def getPointsDirection(dist, method_dict, n):

    if dist._str() == 'Amalia windrose':
        x, w = getPointsModifiedAmaliaDistribution(dist, method_dict, n)
    if dist._str() == 'Amalia windrose raw' or dist._str() == 'Uniform(0.0, 360.0)':
        x, w = getPointsUnmodifiedDistribution(dist, method_dict, n)

    return x, w


def getPointsSpeed(dist, method_dict, n):

    method = method_dict['method']
    bnd = dist.range()
    a = bnd[0]  # lower boundary
    b = bnd[1]  # upper boundary
    a = a[0]  # get rid of the list
    b = b[0]  # get rid of the list

    if method == 'rect':

        X = np.linspace(a, b, n+1)
        dx = X[1]-X[0]
        x = X[:-1]+dx/2  # Take the midpoints of the bins
        # Get the weights associated with the points locations
        w = []
        for i in range(n):
            w.append(dist._cdf(X[i+1]) - dist._cdf(X[i]))

        w = np.array(w).flatten()

    if method == 'dakota':

        x, f = generate_speed_abscissas_ordinates(a, b, dist)
        updateDakotaFile(method_dict, n, x, f)
        # run Dakota file to get the points locations
        x, w = getSamplePoints(method_dict['dakota_filename'])
        assert len(x) == 1, 'Should only be returning the speeds'
        x = np.array(x[0])

        # Rescale x
        x = (b-a)/2. + (b-a)/2.*x + a

    if method == 'chaospy':
        x, w = cp.generate_quadrature(n-1, dist, rule='G')
        x = x[0]

    return x, w


def generate_direction_abscissas_ordinates(a, A, B, C, r, R, dist):

    # Use the y to set the abscissas, and the pdf to set the ordinates
    y = np.linspace(a, R, 51)  # play with the number here
    dy = y[1]-y[0]
    mid = y[:-1]+dy/2

    ynew = modifyx(mid, A, B, C, r)
    f = dist.pdf(ynew)

    # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
    y = 2*(y-a) / R - 1
    return y, f


def generate_speed_abscissas_ordinates(a, b, dist):

    # Use the y to set the abscissas, and the pdf to set the ordinates
    y = np.linspace(a, b, 51)  # play with the number of points here
    dy = y[1]-y[0]
    ymid = y[:-1]+dy/2
    f = dist.pdf(ymid)
    # Modify y to -1 to 1 range, I think makes dakota generation of polynomials easier
    y = (2.0 / (b-a)) * (y-a) - 1.0
    return y, f


def modifyx(x, A=110, B=140, C=225, r=360):

    # Make sure the offset is not between A and B
    if A < C and C < B:
        C = min([A, B], key=lambda x: abs(x-C))  # It doesn't really matter if C gets set to A or B

    # Modify x, to start from the max probability location
    x = (x+C) % r
    y = []
    for xi in x:
        if A<C:
            if xi > A and xi < C:
                xi = (xi + B-A) % r  # I don't think the mod r is necessary for all of these.
            y.append(xi)
        else:
            if xi > A:
                xi = (xi + B-A) % r
            else:
                if xi < C:
                    xi = (xi + B-A) % r
            y.append(xi)

    return np.array(y)


def getWeights(x, dx, dist):
    # Logic to get the weights from integrating the pdf between the bins
    w = []
    for xi in x:
        xleft = xi-dx/2.
        xright = xi+dx/2.

        if xright >= 360.0:  # The equal because the pdf for the raw distribution doesn't integrate exactly to 1 in the cdf.
            w.append(1 - dist._cdf(xleft) + dist._cdf(xright-360))
        elif xleft < 0.0:
            w.append(dist._cdf(xright) + (1 - dist._cdf(360+xleft)))
        else:
            w.append(dist._cdf(xright) - dist._cdf(xleft))

    w = np.array(w).flatten()
    # print w  # all weights should be positive
    # print 'the sum', np.sum(w)
    np.testing.assert_almost_equal(np.sum(w), 1.0, decimal=12, err_msg='the weights should add to 1.')
    return w


def getWeightsModifiedAmalia(x, dx, dist):
    # Logic to get the weights from integrating the pdf between the bins

    w = []

    if len(x) == 1:  # Avoids having to do the logic when there is only one point.
        w.append(1)
    else:
        counter = 0
        for xi in x:
            xleft = xi-dx/2.
            xright = xi+dx/2.

            if counter == 0:
                xleft_first = xleft
            if counter+1 == len(x) and not np.isclose(xleft_first%360, xright%360, rtol=0.0, atol=1e-12):  # I think mostly the case when len(x) = 2
                if xright < xleft_first:
                    xright = xleft_first
                if xright > xleft_first:
                    xright = 360+xleft_first

            # This logic is to make sure that the weights add up to 1, because of the skipping over the zero probability region.
            # if counter > 0 and (xleft%360) != (xright_old%360):  # Doesn't work properly because of rounding errors.
            if counter > 0 and not np.isclose(xleft%360, xright_old%360, rtol=0.0, atol=1e-12):
                xleft = xright_old%360

            if xright >= 360.0:  # The equal because the pdf for the modified amalia distribution doesn't integrate exactly to 1 ind the cdf.
                w.append(1 - dist._cdf(xleft) + dist._cdf(xright-360))
            elif xleft < 0.0:
                w.append(dist._cdf(xright) + (1 - dist._cdf(360+xleft)))
            else:
                w.append(dist._cdf(xright) - dist._cdf(xleft))

            xright_old = xright
            counter += 1

    w = np.array(w).flatten()
    # print w  # all weights should be positive
    # print 'the sum', np.sum(w)
    np.testing.assert_almost_equal(np.sum(w), 1.0, decimal=12, err_msg='the weights should add to 1.')
    return w


def getLayout(layout='grid'):
    ### Set up the farm ###

    # Set up position of each turbine

    if layout == 'grid':

        # Grid wind farm
        locations = np.genfromtxt('../WindFarms/layout_grid.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'test':
        # Small Farm
        nRows = 2  # 10   # number of rows and columns in grid
        spacing = 5  # turbine grid spacing in diameters
        rotor_diameter = 126.4  # (m)

        # Grid farm
        points = np.linspace(start=spacing*rotor_diameter, stop=nRows*spacing*rotor_diameter, num=nRows)
        xpoints, ypoints = np.meshgrid(points, points)
        turbineX = np.ndarray.flatten(xpoints)
        turbineY = np.ndarray.flatten(ypoints)

    elif layout == 'random':

        # Random farm
        locations = np.genfromtxt('../WindFarms/layout_random.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia':

        # Amalia wind farm
        locations = np.genfromtxt('../WindFarms/layout_amalia.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'optimized':

        # Amalia optimized
        # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ') # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/layout_optimized.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'layout1':

        # Amalia optimized
        # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ') # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/layout_1.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'layout2':

        # Amalia optimized
        # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ') # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/layout_2.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'layout3':

        # Amalia optimized
        # locations = np.genfromtxt('../WindFarms/AmaliaOptimizedXY.txt', delimiter=' ') # Amalia optimized Jared
        locations = np.genfromtxt('../WindFarms/layout_3.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia_sub1':
        # A sub part of the amalia layout
        locations = np.genfromtxt('../WindFarms/layout_amalia_sub1.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia_sub2':
        # A sub part of the amalia layout
        locations = np.genfromtxt('../WindFarms/layout_amalia_sub2.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia_sub3':
        # A sub part of the amalia layout
        locations = np.genfromtxt('../WindFarms/layout_amalia_sub3.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia_sub4':
        # A sub part of the amalia layout
        locations = np.genfromtxt('../WindFarms/layout_amalia_sub4.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'amalia_sub5':
        # A sub part of the amalia layout
        locations = np.genfromtxt('../WindFarms/layout_amalia_sub5.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    elif layout == 'local':
        # A layout in your local directory
        locations = np.genfromtxt('layout_local.txt', delimiter=' ')
        turbineX = locations[:, 0]
        turbineY = locations[:, 1]

    else:
        raise ValueError('unknown layout option "%s", \nvalid options ["amalia", "optimized", "random", "test", "grid", "layout1", "layout2", "layout3", "amalia_sub1", "amalia_sub2", "amalia_sub3", "amalia_sub4", "amalia_sub5", "local"]' %layout)

    # plt.figure()
    # plt.scatter(turbineX, turbineY)
    # plt.show()

    return turbineX, turbineY
