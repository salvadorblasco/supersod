""" ++++ LIBSTAT.PY +++++

Contains functions related to statistical calculations.
"""


def dixon_critical(number_samples, confidence):
    '''Critical Dixon's Q value.

    Parameters:
        number_samples (int): The number of observations (min 3, max 30)
        confidence (int): The confidence percent. Valid values are 90, 95 or 99
    Returns:
        float: The critical Dixon's Q value
    '''
    Q90 = (0.941, 0.765, 0.642, 0.56, 0.507, 0.468, 0.437, 0.412, 0.392, 0.376,
           0.361, 0.349, 0.338, 0.329, 0.32, 0.313, 0.306, 0.3, 0.295, 0.29,
           0.285, 0.281, 0.277, 0.273, 0.269, 0.266, 0.263, 0.26)
    Q95 = (0.97, 0.829, 0.71, 0.625, 0.568, 0.526, 0.493, 0.466, 0.444, 0.426,
           0.41, 0.396, 0.384, 0.374, 0.365, 0.356, 0.349, 0.342, 0.337, 0.331,
           0.326, 0.321, 0.317, 0.312, 0.308, 0.305, 0.301, 0.29)
    Q99 = (0.994, 0.926, 0.821, 0.74, 0.68, 0.634, 0.598, 0.568, 0.542, 0.522,
           0.503, 0.488, 0.475, 0.463, 0.452, 0.442, 0.433, 0.425, 0.418,
           0.411, 0.404, 0.399, 0.393, 0.388, 0.384, 0.38, 0.376, 0.372)

    def dictify(q):
        return {n: v for n, v in enumerate(q, start=3)}

    Q = {90: dictify(Q90), 95: dictify(Q95), 99: dictify(Q99)}
    return Q[confidence][number_samples]


def dixons_Q(sample):
    r'''The Dixon's Q value.

    The Q value is defined as :math:`\frac{x-x_{max}}{x_{max}-x_{min}}`

    Parameters:
        sample (sequence): the values of the observations. It must be
            sorteable.
    Returns:
        tuple: Dixon's Q value the smallest and biggest items in sample.
    '''
    ss = sorted(sample)
    q1 = (ss[1]-ss[0])/(ss[-1]-ss[0])
    q2 = (ss[-1]-ss[-2])/(ss[-1]-ss[0])
    return q1, q2


