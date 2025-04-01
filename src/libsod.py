"Standard routines for SOD-related calculations."

import numpy as np
import consts


def auto_fit(x, y, r_min=0.999, min_fraction=0.5):
    '''Authomatically find the linear range of a data set.

    Parameters:
        x (sequence): x values for dataset
        y (sequence): y values for dataset
        r_min (float): minimum acceptable r² value for fitting
        min_fraction (float): minimum fraction of dataset to use.
    Returns:
        The parameters found to represent the best linear interval in the data
        which are, in order:
        - float: the lower cutoff
        - float: the upper cutoff
        - float: the r² parameter
    Raises:
        ValueError: if *x* and *y* are not the same size
        ValueError: is *r_min* or *min_fraction* are not float numbers between
            0 and 1.
    '''
    if len(x) != len(y):
        raise ValueError("'x' and 'y' must be the same size")
    if not 0 < r_min < 1:
        raise ValueError("'r_min' must be a float between 0 and 1")
    if not 0 < min_fraction < 1:
        raise ValueError("'min_fraction' must be a float between 0 and 1")

    lower = 0
    upper = len(x)

    def r_squared(slice_):
        return (np.corrcoef(x[slice_], y[slice_])[0, 1])**2

    r_curr = r_squared(slice(lower, upper))

    while True:
        if r_curr < r_min:
            top_slice = slice((lower+1), upper)
            r_top = r_squared(top_slice)
            bottom_slice = slice(lower, (upper-1))
            r_bottom = r_squared(bottom_slice)
            if r_top > r_bottom:
                r_curr = r_top
                lower += 1
            else:
                r_curr = r_bottom
                upper -= 1
        else:
            break

        if upper - lower < min_fraction*len(x):
            break

    return lower, upper, r_curr


def autoprocess_spectra(stream):
    '''Automate task of processing data.

    A stream of data is plugged in. Each element is a tuple containing the
    concentration and the spectra data. Each spectra data is a tuple of several
    spectra.

    Parameters:
        stream (iterable): the stream of data
    Returns:
        - list of concentration values
        - list of found value slopes
        - list of errors for the slopes
        - list of lists with the raw spectra
    '''
    import statistics       # python >=3.4

    concentration = []
    slopes = []
    slope_error = []
    raw_points = []

    for conc, spectra in stream:
        concentration.append(conc)
        current_group = []
        current_slopes = []
        for spectrum in spectra:
            # x, y = spectrum.T
            # lower, upper, _ = auto_fit(x, y)
            # slope, _, _, _, _ = spst.linregress(x, y)
            bounds, slope = process_spectrum(spectrum)
            # current_group.append((True, (lower, upper), spectrum.T))
            current_group.append((True, bounds, spectrum.T))
            current_slopes.append(slope)
        # mean_slope = sum(current_slopes)/len(current_slopes)
        mean_slope = statistics.mean(current_slopes)
        slopes.append(mean_slope)

        std_slope = statistics.stdev(current_slopes)
        # if len(current_slopes) == 1:
        #     std_slope = 0.0
        # else:
        #     std_slope = sum((i - mean_slope)**2
        #                     for i in current_slopes)**0.5 \
        #                     / (len(current_slopes) - 1)

        slope_error.append(std_slope)
        raw_points.append(current_group)
    return concentration, slopes, slope_error, raw_points


def process_spectrum(spectrum):
    import scipy.stats as spst
    x, y = spectrum.T
    lower, upper, _ = auto_fit(x, y)
    slope, _, _, _, _ = spst.linregress(x, y)
    return (lower, upper), slope


def calc_kcat(IC50, error_IC50, k_ind, errk_ind, c_ind, errc_ind=0.0):
    '''Calculate catalytic constant from IC50.

    Catalytic rate can be estimated as
    :math:`k_{cat} = k_{ind}C_{ind}/({IC}_{50})`

    Parameters:
        IC50 (float): the IC50
        error_IC50 (float): the error for IC50
        k_ind (float): the rate constant for the indicator
        errk_ind (float): the error of k_ind
        c_ind (float): the concentration of indicator
        errc_ind (float): the error of c_ind
    Return:
        tuple: (1) the k_cat and (2) the error
    '''
    k_cat = k_ind*c_ind/IC50
    error_kcat = k_cat*(
        (errk_ind/k_ind)**2 +
        (errc_ind/c_ind)**2 +
        (error_IC50/IC50)**2)**0.5
    return k_cat, error_kcat


def calc_co2(ind, c_ind, slope0):
    '''Calculate [O2-] and d[O2-]/dt in the steady state.

    S_blank = epsilon_NBT * l * k_NBT * [NBT] * [O2-]_t
    the 1e12 factor is for [NBT] being in mM (x 10^-3)
    and for cO2m being in nM (x10^9)

    Parameters:
        ind (int): the indicator used
        c_ind (float): concentration of the indicator, in µM
        slope0 (iterable): values of dA/dt for the blank in 1e4 scale.
    Yields:
        tuple: the values of the concentration of superoxide in the steady
            state (in nanomole per liter) and the superoxide rate generation
            (in nanomole per liter and second )for each value in
            slope0.
    '''
    optical_path = 1.0     # cm
    k_sd = 5e5  # M^-1s^-1 -> spontaneous dismutation of O2-
    k_ind = consts.k_indicator[ind]
    eps_ind = consts.epsilon_indicator[ind]
    for s0 in slope0:
        cO2m = 1e10 * s0 / (eps_ind * optical_path * k_ind * c_ind)
        dcO2m = s0 / (eps_ind*optical_path) * (1 + k_sd/(k_ind*c_ind*1e-3))
        yield cO2m, dcO2m


def calc_IC(conc, slope, error_slope):
    '''Calculate IC and error.

    The IC is defined as :math:`IC=S/S_0` where *S* are the values of the
    slopes measured and *S₀* is the slope of the blank.

    Parameters:
        conc (iterable): values of the concentrations
        slope (iterable): values of the slope
        error_slope (iterable): errors of the slope
    Yields:
        tuple: with values of IC and error of IC
    '''
    assert conc[0] == 0.0
    for c, s, es in zip(conc, slope, error_slope):
        if c == 0.0:
            es0 = es
            s0 = s
        yield 100*(s0-s)/s0, 100*(((s0*es)**2 + (s*es0)**2)**0.5)*s0**-2


def f_IC(p, x):
    assert isinstance(x, np.ndarray)
    assert len(p) == 2, str(p)
    return (1-p[0])*p[1]*x/(1+p[1]*x)


def fit(x, y, w, error_y=None):
    """Fit curve with experimental data.

    This is the main routine where calculations are done.

    Parameters:
        x (:class:`numpy.ndarray`): the concentration array
        y (:class:`numpy.ndarray`): the IC array
        w (:class:`numpy.ndarray`): the weight array
    Returns:
        float IC50 (float):
        error_IC50 (float):
        par_f (float):
        error_f (float):
        par_K (float):
        error_K (float):
    """
    # import pudb
    # pudb.set_trace()
    for array in (x, y, w):
        if not isinstance(array, np.ndarray):
            raise TypeError("argument must be a numpy array")
        if array.ndim != 1:
            raise ValueError("Array must be 1D.")
    if not len(x) == len(y) == len(w):
        raise ValueError("Arrays must have the same length")

    # ! from scipy.optimize import leastsq

    init_f = 0.01
    init_K = 5.0

    # Redefined to include weights
    # ! def obj_func(p, w, x, y):
    # ! def obj_func(x, y, p):
    # !     return w*(f_IC(p, x) - y)

    def obj_func(p):
        ic = (1-p[0])*p[1]*x/(1+p[1]*x)
        return w*(y-ic)

    # from scipy.optimize import curve_fit
    # popt, pcov = curve_fit(obj_func, x, y, p0=(init_f, init_K),sigma=error_y,
    #                        method='trf', ftol=1e-15, bounds=(0.0, np.inf))

    from scipy.optimize import leastsq
    popt, pcov, infodict, msg, ier = leastsq(
        obj_func, (init_f, init_K), full_output=1)

    # ! par_f = p_fit[0]
    # ! par_K = p_fit[1]
    par_f, par_K = popt
    IC50 = 1/((1-2*par_f)*par_K)

    # estimate covariance. Taken out of octave's optimization module
    # ! m = y.shape[0]
    # ! n = 2
    # ! wt = np.ones((m, 1))        # by default, weights are one
    # ! Q = np.matrix(np.diagflat(1./wt**2))
    # ! Qinv = np.matrix(np.diagflat(wt**2))
    # ! residT = np.matrix(obj_func((par_f, par_K), w, x, y))
    # ! resid = np.transpose(residT)
    # ! covr = float(residT*Qinv*resid)*Q/(m-n)
    # ! Vy = 1/(1-n/m)*covr
    # ! dydf = np.transpose(np.matrix(-y/(1-par_f)))
    # ! dydK = np.transpose(np.matrix(y/(par_K*(1+par_K*x))))
    # ! Jac = np.bmat([dydf, dydK])
    # ! JacT = np.transpose(Jac)
    # ! jtgjinv = np.linalg.inv(JacT*Qinv*Jac)
    # ! covp = jtgjinv*JacT*Qinv*Vy*Qinv*Jac*jtgjinv

    # ! error_f = (covp[0, 0])**0.5
    # ! error_K = (covp[1, 1])**0.5
    resvar = np.sum(infodict['fvec']**2)/(len(x)-2)
    error_f, error_K = resvar*np.sqrt(np.diag(pcov))

    # error_IC50 = ((2/IC50)**2 * (error_f/par_f)**2 +
    #               (IC50/par_K)**4 * (error_K/par_K)**2)**0.5
    error_IC50 = IC50*((2*error_f/(1-2*par_f))**2 + (error_K/par_K)**2)**0.5

    return IC50, error_IC50, par_f, error_f, par_K, error_K


def quadratic_weighting_scheme(IC, w0=0.5):
    '''Compute the quadratic weighting scheme.

    In this scheme a quadratic function with the following constraints is
    calculated:
        - if x = 0.5 then y = 1.0
        - if x = 0   then y = w0
        - if x = 1   then y = w0
    The purpose of this scheme is to assign bigger weights for the
    data close to 0.5 and lesser weights to the fartest points.

    Parameters:
        IC (iterable): stream of floats
        w0 (float): parameter
    Yields:
        float: the weight value
    '''
    # TODO issue warnings
    # Warning: negative values
    # Warning: values greater than 1.0
    return (4*(w0-1)*y*(y-1)+w0 for y in map(lambda x: 0.01*x, IC))


def html_report(**kwargs):
    """Pop up a window with the report.

    Parameters:
        concentration (sequence):
        inhibition (sequence):
    """
    report_vars = {}
    # ic = kwargs['inhibition']
    # use = kwargs['use']
    slopes = kwargs['slope']
    blank_slopes = [s[0] for c, s in splitdata1(kwargs['concentration'],
                                                slopes)]
    superox_data = calc_co2(kwargs['indicator'],
                            kwargs['cindicator'],
                            blank_slopes)
    superox_text = ("""<h3>Dataset #{}</h3>
        <p>Superoxide concentration in the steady state: {} nM</p>
        <p>Superoxide generation rate in the steady state: {} µmol/L.s</p>
        """.format(n, a, b) for n, (a, b) in enumerate(superox_data))

    warning_stream = warnings(**kwargs, blank_slopes=np.array(blank_slopes))
    warning_text = ('<p>Warning #{}: {}</p>'.format(n, s)
                    for n, s in enumerate(warning_stream))

    def _round(number, error, nsignif=1):
        from math import floor, log10
        precision = nsignif - floor(log10(error)) - 1
        return round(number, precision), round(error, precision)

    number, error = _round(kwargs['IC50'], kwargs['error_IC50'])
    report_vars['ic50f'], report_vars['error_ic50f'] = number, error
    number, error = _round(kwargs['kcat'], kwargs['error_kcat'])
    report_vars['kcat_f'], report_vars['error_kcat_f'] = number, error
    report_vars.update(kwargs)

    text = """<html>
    <h1>Results of the fitting</h1>
    <p>IC<sub>50</sub> = {ic50f} ± {error_ic50f} µmol/L <br />
       ({IC50} ± {error_IC50})</p>
    <p>k<sub>cat</sub> = {kcat_f:e} ± {error_kcat_f:e} mol/L.s <br />
       ({kcat} ± {error_kcat})</p>
    <p>f = {f} ± {error_f}</p>
    <p>K = {K} ± {error_K}</p>

    <h1>Other parameters</h1>
    <h2>Superoxide generation</h2>
    {superoxide_generation}

    <h1>Warnings</h1>
    {warnings}
    </html>""".format(
        **report_vars,
        superoxide_generation=''.join(superox_text),
        warnings=''.join(warning_text))

    return text


def warnings(**kwargs):
    """Analyse the data and return some warnings as text.

    Arguments:
        blank_slopes (:class:`numpy.ndarray`): The slopes of the blank
        kcat (float): The catalytic constant
        error_kcat (float): The error for k_cat.
        inhibition (:class:`numpy.ndarray`): The calculated
            values for the reduction inhibition.
        use (:class:`numpy.ndarray`): The use flags (boolean).

    Yields:
        str: a text describing the warnings found.
    """
    # warning #1: blank slopes out of range
    # warning #8: only one dataset
    if 'blank_slopes' in kwargs:
        blank_slopes_upper_limit = 12.0
        blank_slopes_lower_limit = 8.0
        s0 = kwargs['blank_slopes']
        if any(i < blank_slopes_lower_limit for i in s0):
            yield "some blank value probably too low"

        if any(i > blank_slopes_upper_limit for i in s0):
            yield "some blank value probably too high"
        if len(s0) == 1:
            yield ("Only one dataset. "
                   "At least two reproducible datasets are recommended")

    # warning #2: kcat similar to k spontaneous dismutation
    if 'kcat' in kwargs:
        k_self_dismutation = 6.0e5
        if kwargs['kcat'] <= k_self_dismutation:
            yield """<i>k</i><sub>cat</sub> is similar or lower than <i>k</i> of
                spontaneous dismutation. Results are probably meaningless."""

    # warning #3: when residuals are not random
    # TODO

    # warning #4: any IC < 0
    if 'inhibition' in kwargs:
        if any([x < 0 for x in kwargs['inhibition']]):
            yield "one or more points have negative IC values."

    if 'use' in kwargs:
        # warning #5: not enough points
        fair_enough_points = 8
        if len(kwargs['use']) < fair_enough_points:
            yield "few points for a fit"

        # warning #6: not enough points
        if len([i for i in kwargs['use'] if i]) < fair_enough_points:
            yield "few usable points for a fit"

    # warning #7: few points around 50%
    imbalance_threshold = 0.2
    under = sum(1 for i in kwargs['inhibition'] if i < 50)
    length = len(kwargs['inhibition'])
    if under == 0:
        yield "No points under 50%"
    elif under/length < imbalance_threshold:
        yield "Few points under 50%"
    if length - under == 0:
        yield "No points over 50%"
    elif 1 - under/length < imbalance_threshold:
        yield "Few points over 50%"


def splitdata1(concentration, *args):
    """Split stream of data into datasets.

    Each dataset must start with concentration = 0. All the arguments
    provided, included *concentration* are split in this way.

    Parameters:
        concentration (sequence): the concentration values.
        args (sequence): other data that will be split as *concentration*.
    Yields:
        All the data split accordingly
    """
    idx = [n for n, i in enumerate(concentration) if i == 0.0]
    yield from zip(np.split(concentration, idx[1:]),
                   *(np.split(a, idx[1:]) for a in args))


def splitdata2(use, data):
    """Split continuous stream of data usable/non usable data.

    Parameters:
        use (sequence): A bool sequence stating whether the data is going
            to be used or not.
        data (sequence): A stream of data with the same length than *use*.
    Returns:
        tuple: elements of data to be used and elements of data not used.
    """
    return np.extract(use[1:], data[1:]), \
        np.extract(np.logical_not(use[1:]), data[1:])
