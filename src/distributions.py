import chaospy as cp
import numpy as np
# import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import special


class amaliaWindRose(object):
    """The smoothed amalia distribution."""

    def __init__(self):

        # f(x)
        #   |                   *
        #   |   ***            * *      **
        #   | **   *          *   **  **  *     ***
        #   |*      *        *      **     *  **   *
        #   |        *      *               **
        #   |         *    *
        # --+----------****-----+------------------+--
        #  lo          A  B     C                  hi    (x)

        self.lo = 0.0
        self.hi = 360.0
        self.A = 110  # Left boundary of zero probability region
        self.B = 140  # Right boundary of zero probability region
        # self.C = 225  # Location of max probability

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. But not for cdf
        f = self._wind_rose_poly_func()  # This will give me different results for the rectangle method. It's because the values were different
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose"

    def bnd(self):
        return self.lo, self.hi

    def _wind_rose_poly_func(self):
        def f(x):
            return np.array([self._f_helper(z) for z in x])
        return f

    def _windrose_polyfit(self, x):
        y = 493597.250387841  *np.power(x, 12) + \
            -207774.160030495 *np.power(x, 11) + \
            -413203.013010848 *np.power(x, 10) + \
            158080.893880027  *np.power(x, 9) + \
            127607.500730722  *np.power(x, 8) + \
            -44242.1722820275 *np.power(x, 7) + \
            -17735.2623897828 *np.power(x, 6) + \
            5422.11156037294  *np.power(x, 5) + \
            1057.31910521884  *np.power(x, 4) + \
            -253.807324825523 *np.power(x, 3) + \
            -19.8973363502958 *np.power(x, 2) + \
            1.43458543839655  *np.power(x, 1) + \
            1.05778787373732  *np.power(x, 0)
        return y

    def _f_helper(self, x):
        # Linear transformation from interval [a,b] to [-0.5,0.5]
        A = self.A
        B = self.B
        a = B  # 140
        b = self.hi - self.lo + A  # 470
        if x >= B:
            x1 = (x - (b+a)/2.) / (b-a)
            return self._windrose_polyfit(x1)/(b-a)  # 330
        elif x <= A:
            x1 = (x + 360 - (b+a)/2.) / (b-a)
            return self._windrose_polyfit(x1)/(b-a)  # 330
        else:
            # If I'm only calling the pdf I should not go in here, but when calling the cdf I get in here.
            return 0.0

    def get_zero_probability_region(self):
        return self.A, self.B


class amaliaWindRoseRaw(object):
    """The raw amalia distribution."""

    def __init__(self):
        self.lo = 0.0
        self.hi = 360.0
        self.inputfile = '../WindRoses/windrose_amalia_8ms.txt'
        # self.inputfile = '/Users/Santi/gitSoftware/windfarm-ouu/WindRoses/windrose_amalia_8ms.txt'

    def _wind_rose_func(self):
        wind_data = np.loadtxt(self.inputfile)
        direction = wind_data[:, 0]
        speed = wind_data[:, 1]  # Speed is a constant for this file.
        likelihood = wind_data[:, 2]
        # Get rid of the zeros. Average this out, so distribution still integrates to 1.
        likelihood[23:26] = np.average(likelihood[23:26])
        dx = direction[1] - direction[0]  # the step 5 deg
        # Adjust the likelihood so it is a pdf for the [0, 360] range
        w = likelihood/dx
        # Make sure it adds up to 1.
        # print 'integral of the pdf = ', np.sum(w*dx)
        # Assume the weights correspond to the midpoint values of the directions
        direction = direction + dx/2.
        # Add the end point directions and endpoint weights
        weight_end = (w[0]+w[-1])/2.
        direction = np.concatenate([[0], direction, [360]])
        w = np.concatenate([[weight_end], w, [weight_end]])
        f = interp1d(direction, w)
        return f

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. For this amalia class this flattening is unnecesary
        f = self._wind_rose_func()
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose raw"

    def bnd(self):
        return self.lo, self.hi


class Uniform(object):
    """A Uniform distribution."""

    def __init__(self):
        self.lo = 0.0
        self.hi = 360.0

    def pdf(self, x):

        lo = self.lo
        hi = self.hi

        pdf = []
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. But not for cdf
        x = np.atleast_1d(x)  # makes it work if x is a scalar
        for x_i in x:
            if x_i < lo:
                pdf.append(0.0)
            elif x_i > hi:
                pdf.append(0.0)
            else:
                pdf.append(1.0/(hi-lo))
        return np.array(pdf)

    def cdf(self, x):

        lo = self.lo
        hi = self.hi

        cdf = []
        x = np.atleast_1d(x)  # makes it work if x is a scalar
        for x_i in x:
            if x_i < lo:
                cdf.append(0.0)
            elif x_i > hi:
                cdf.append(1.0)
            else:
                cdf.append((x_i - lo)/(hi-lo))
        return np.array(cdf)

    def str(self):
        return "Uniform(%s, %s)" % (self.lo, self.hi)

    def bnd(self):
        return self.lo, self.hi


class amaliaWindRoseRaw01(object):
    """The raw amalia distribution."""

    def __init__(self):
        self.lo = 0.0
        self.hi = 1.0
        self.inputfile = '../WindRoses/windrose_amalia_8ms.txt'

    def _wind_rose_func(self):
        wind_data = np.loadtxt(self.inputfile)
        # direction = wind_data[:, 0]
        # speed = wind_data[:, 1]
        probability = wind_data[:, 2]
        N = len(probability)
        probability[probability == 0] = 2e-5  # think about this, # Update were the wind data is zero to the next lowest value
        probability = np.append(probability, probability[0])  # Include the value at 360, which is the same as 0.
        probability = probability*N  # normalize for the [0, 1] range.
        x = np.linspace(0, 1, N+1)
        f = interp1d(x, probability)
        return f

    def pdf(self, x):
        x = x.flatten()  # In the constructor of the distribution it gets made a 2d array for some reason. For this amalia class this flattening is unnecesary
        f = self._wind_rose_func()
        return f(x)

    def cdf(self, x):
        # Integrate by rectangle rule
        # dx = 0.001  # interval spacing
        cdf = []
        for x_i in np.array(x, copy=False, ndmin=1):  # makes it work if x is a scalar
            dx = x_i/100.  # interval spacing
            if x_i == 0:
                cdf.append(0.0)
            else:
                X = np.arange(dx/2, x_i, dx)
                cdf.append(np.sum(self.pdf(X)*dx))  # integration by rectangle rule
        return np.array(cdf)

    def str(self):
        return "Amalia windrose raw [0,1]"

    def bnd(self):
        return self.lo, self.hi


class myWeibull(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 0.0
        self.hi = 80.0  # cdf(80) = 1, to machine precision

    def cdf(self, x):
        a = self.a
        b = self.b
        F = 1-np.exp(-(x/b)**a)
        return F

    def pdf(self, x):
        a = self.a
        b = self.b
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        return f

    def mom(self, k):
        # I don't think providing the moments here changes anything down the road when
        # doing PC expansions. Although if I don't pass mom the moments calculated are
        # bad. It appears that they just take the moments assuming a uniform with
        # a range, as that obtained from the bounds.
        # Actually it does affect the PC expansions, this comes into play when getting
        # the orthogonal polynomials.
        a = self.a
        b = self.b
        return b * special.gamma(1.+k*1./a)

    def str(self):
        return "weibull(%s, %s)" % (self.a, self.b)

    def bnd(self):
        return self.lo, self.hi


class TruncatedWeibull(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 3.0  # 0.0
        self.hi = 20.0  # 30.0
        self.k = self.set_truncation_value()

    def set_truncation_value(self):
        """Sets k, which represents how much of the distribution is truncated"""
        weibull = myWeibull()
        k = weibull.cdf(self.lo) + (1.0 - weibull.cdf(self.hi))
        return k

    def get_truncation_value(self):
        return self.k

    def cdf(self, x):
        a = self.a
        b = self.b
        F = np.exp(-(self.lo/b)**a) - np.exp(-(x/b)**a)  # Account for the truncation
        F = 1 / (1.0-self.k) * F  # Account for the truncation
        return F

    def pdf(self, x):
        a = self.a
        b = self.b
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        f = 1 / (1.0-self.k) * f  # Account for the truncation
        return f

    def str(self):
        return "Truncated [%s, %s] weibull(%s, %s)" % (self.lo, self.hi, self.a, self.b)

    def bnd(self):
        return self.lo, self.hi


class TruncatedWeibull01(object):
    def __init__(self):
        self.a = 1.8
        self.b = 12.552983
        self.lo = 0.0
        self.hi = 1.0
        self.lo1 = 0.0  # The original low and high bounds
        self.hi1 = 30.0  # The original low and high bounds
        self.k = self.set_truncation_value()

    def set_truncation_value(self):
        """Sets k, which represents how much of the distribution is truncated"""
        weibull = myWeibull()
        k = weibull.cdf(self.lo1) + (1.0 - weibull.cdf(self.hi1))
        return k

    def get_truncation_value(self):
        return self.k

    def cdf(self, x):
        a = self.a
        b = self.b
        # Adjust x
        x = self.lo1 + (self.hi1-self.lo1) * x
        F = np.exp(-(self.lo/b)**a) - np.exp(-(x/b)**a)  # Account for the truncation
        F = 1 / (1.0-self.k) * F  # Account for the truncation
        return F

    def pdf(self, x):
        a = self.a
        b = self.b
        # Adjust x
        x = self.lo1 + (self.hi1-self.lo1) * x
        f = a/b * (x/b)**(a-1) * np.exp(-(x/b)**a)
        f = 1 / (1.0-self.k) * f  # Account for the truncation
        f = (self.hi1-self.lo1) * f  # Account for the rescaling
        return f

    def str(self):
        return "Truncated [%s, %s] weibull01(%s, %s)" % (self.lo, self.hi, self.a, self.b)

    def bnd(self):
        return self.lo, self.hi

def getWeibull():

    # my_weibull = myWeibull()
    my_weibull = TruncatedWeibull()
    # my_weibull = TruncatedWeibull01()
    # Set the necessary functions to construct a chaospy distribution
    Weibull = cp.construct(
        cdf=lambda self, x: my_weibull.cdf(x),
        bnd=lambda self: my_weibull.bnd(),
        pdf=lambda self, x: my_weibull.pdf(x),
        # mom=lambda self, k: my_weibull.mom(k),
        str=lambda self: my_weibull.str()
    )

    weibull_dist = Weibull()
    # Dynamically add method
    weibull_dist.get_truncation_value = my_weibull.get_truncation_value

    return weibull_dist


def getWindRose(distribution):
    """Gets a chaospy distribution,
        which is initialized with a distribution class I created
        and extended by it.
    """

    # Return the desired distribution (windRose)
    if distribution == 'amaliaModified':
        wind_rose = amaliaWindRose()
    elif distribution == 'amaliaRaw':
        wind_rose = amaliaWindRoseRaw()
    elif distribution == 'Uniform':
        wind_rose = Uniform()
    else:
        raise ValueError('unknown dirdistribution option "%s", valid options "amaliaModified", "amaliaRaw", "Uniform".' % distribution)

    # wind_rose = amaliaWindRoseRaw01()  # This options needs updating

    # Set the necessary functions to construct a chaospy distribution
    windRose = cp.construct(
        cdf=lambda self, x: wind_rose.cdf(x),
        bnd=lambda self: wind_rose.bnd(),
        pdf=lambda self, x: wind_rose.pdf(x),
        str=lambda self: wind_rose.str()
    )

    windrose_dist = windRose()
    # print windrose_dist
    # print windrose_dist.pdf(180)
    # print windrose_dist.pdf(365)
    # print windrose_dist.range()

    # Dynamically add method
    if wind_rose.str() == 'Amalia windrose':
        windrose_dist.get_zero_probability_region = wind_rose.get_zero_probability_region


    return windrose_dist


# # Make nice plots of the distributions
# import prettify
# import matplotlib as mpl
#
# # Wind Direction plot
# dist = getWindRose()
# x = np.linspace(0, 360, 361)
# y = dist.pdf(x)
# fig, ax = plt.subplots(figsize=(9, 5.4))
# prettify.set_color_cycle(ax)
# prettify.remove_junk(ax)
# ax.plot(x, y, linewidth=3)
# major_formatter = mpl.ticker.FormatStrFormatter('%g')
# ax.yaxis.set_major_formatter(major_formatter)
# ax.set_xticks(range(0, 361, 90))
# # ax.set_xticks([0, 110, 140, 225, 360])
# # ax.set_yticks([0, 0.0019, 0.0051])
# ax.set_yticks([])
# ax.spines['left'].set_visible(False)
# ax.set_xlim([-10, 370])
# ax.set_ylim([-0.00025, 0.00535])
# ax.tick_params(axis='both', labelsize=24)
# ax.set_xlabel('wind direction (deg)', fontsize=24)
#
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
# plt.savefig('PdfWindDirection.pdf', bbox_inches='tight')
#
#
# # Wind Speed plot
# dist = getWeibull()
# x = np.linspace(0, 30, 301)
# y = dist.pdf(x)
# fig, ax = plt.subplots(figsize=(9, 5.4))
# prettify.set_color_cycle(ax)
# prettify.remove_junk(ax)
# ax.plot(x, y, linewidth=3)
# major_formatter = mpl.ticker.FormatStrFormatter('%g')
# ax.yaxis.set_major_formatter(major_formatter)
# ax.set_xticks([0, 15, 30])
# ax.set_yticks([])
# ax.spines['left'].set_visible(False)
# ax.set_xlim([-1.5, 31.5])
# ax.set_ylim([-0.0015, 0.0665])
# ax.tick_params(axis='both', labelsize=24)
# ax.set_xlabel('wind speed (m/s)', fontsize=24)
#
# plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
# plt.savefig('PdfWindSpeed.pdf', bbox_inches='tight')
#
# plt.show()
