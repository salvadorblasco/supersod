""" ++++++ LIBIO.PY ++++++
Author: Salvador Blasco <salvador.blasco@gmail.com>

Set of routines for input/output.

load_raw_spectrum
save_sodfile
open_sodfile
import_data
wildcard_files
numwerr
"""

import sys

import numpy as np

import consts


def load_raw_spectrum(filename:str, csvformat:int=0):
    '''Load raw spectrum from file.

    Parameters:
        filename (str): The filename to be read
        format (int): 0 for the Agilent format; 1 for the Cary format

    Returns:
        :class:`numpy.ndarray`: The data read
    '''

    if csvformat not in (0,1):
        raise ValueError("Valid values are 0 (Agilent) or 1 (Cary)")

    import os.path
    if not os.path.exists(filename):
        raise FileNotFoundError

    valid_formats = ({'skiprows':1},
                     {'skiprows':2, 'usecols':(0,1)})
    formatting = valid_formats[csvformat]

    try:
        data = np.loadtxt(filename, delimiter=',', **formatting)
    except:
        import codecs
        try:
            with codecs.open(filename, encoding='utf-16') as fhandler:
                data = np.loadtxt(fhandler, delimiter=',', **formatting)
        except:
            # print(f"Error happened while loading file {filename}")
            # sys.exit(1)
            raise RuntimeError(f"Formatting error while loading file {filename}")

    if csvformat == 1:          # convert minutes to seconds
        data[:,0] *= 60.0

    return data


def save_sodfile(*data):
    """Convert data to XML text and save to a file.

    This function must contain 10 arguments which in turn must be:
    (1) type of indicator, (2) concentration of the indicator,
    (3) concentration, (4) means of the slopes, (5) standard deviation of
    slopes, (6) inhibitory concentration, (7) error in the inhibitory
    concentration, (8) the weights, (9) the raw data.
    """
    import scipy.stats as spst

    sod_template = '''<supersod version="1.0">
         <indicator>
          <name>{}</name>
          <concentration units="uM">{}</concentration>
         </indicator>
         <conditions T="298.15" pH="7.40" />
         <!-- dataset(s) -->
         {}
        </supersod>'''

    dataset_template = '''<dataset name="{}" use="yes">
          <concentrations>{}</concentrations>
          <means>{}</means>
          <stds>{}</stds>
          <IC>{}</IC>
          <error_IC>{}</error_IC>
          <use>{}</use>
          <weight>{}</weight>
          <!-- rpoint(s) -->
          {}
         </dataset>'''

    rpoint_template = '''<rpoint>
         <slopes>{}</slopes>
         <r-squared>{}</r-squared>
         <use>{}</use>
         <slope_multiplier>4</slope_multiplier>
         <!-- rawdata -->
         {}
        </rpoint>'''

    rawdata_template1 = '''<rawdata max_cutoff="{}" min_cutoff="{}">
         <time units="s">{}</time>
         <absorbance>{}</absorbance>
        </rawdata>'''

    rawdata_template2 = '<rawdata>{}</rawdata>'

    indicator = data[0]
    c_indicator = data[1]
    raw_data = data[9]

    def crush(array):
        return " ".join(str(i) for i in iter(array))

    def splitdata(c, arg):
        idx = [n for n, i in enumerate(c) if i == 0.0] + [-1]
        assert idx[0] == 0
        for i, j in zip(idx[:-1], idx[1:]):
            yield arg[i:j]

    points = []
    for point in raw_data:
        use__ = []
        slopes = []
        rsquared = []
        current_point = []
        for useflag, cutoff, spectrum in point:
            use__.append(useflag)

            if cutoff is None:
                txt = rawdata_template2.format(spectrum)
                slopes.append(spectrum)
                rsquared.append('-')
            else:
                x = crush(spectrum[0])
                y = crush(spectrum[1])
                s, _, r, _, _ = spst.linregress(spectrum[0], spectrum[1])
                slopes.append(s)
                rsquared.append(r)
                assert len(cutoff) == 2
                txt = rawdata_template1.format(max(cutoff), min(cutoff), x, y)
                # txt = rawdata_template1.format(cutoff[0], cutoff[1], x, y)
            current_point.append(txt)

        points.append(rpoint_template.format(crush(slopes), crush(rsquared),
                                             crush(use__),
                                             "".join(current_point)))
    datasets = []
    c = tuple(float(n) for n in data[2])
    concentration = splitdata(c, data[2])
    means = splitdata(c, data[3])
    stds = splitdata(c, data[4])
    IC = splitdata(c, data[5])
    error_IC = splitdata(c, data[6])
    use = splitdata(c, data[7])
    weight = splitdata(c, data[8])
    assert len(points) == len(c)
    for point in splitdata(c, points):
        dataset = dataset_template.format(
            'noname',
            crush(next(concentration)),
            crush(next(means)),
            crush(next(stds)),
            crush(next(IC)),
            crush(next(error_IC)),
            crush(next(use)),
            crush(next(weight)),
            "\n".join(point)
        )
        datasets.append(dataset)
    return sod_template.format(indicator, c_indicator, "\n".join(datasets))


def open_sodfile(filename):
    '''Open XML file containing data

    Parameters:
        filename (str): The name of the file to be read
    Yields:
        - str: name of the indicator
        - str: concentration of the indicator
        -
    '''
    import itertools
    import xml.etree.ElementTree as ET

    def bconv(et, tag):
        yes_pattern = ('yes', 'True', '1', 'true')
        return (i in yes_pattern for i in et.findtext(tag).split())

    def fconv(et, tag):
        if tag == 'weight':
            add = '0.5 '
        else:
            add = ''
        return (float(s) for s in (add + et.findtext(tag)).split())

    def dumpds(et):
        for dataset in et.iterfind('dataset'):
            tags = ('concentrations', 'means', 'stds', 'weight')
            for tag in tags:
                ret = fconv(dataset, tag)
                yield ret
            yield bconv(dataset, 'use')

            thisds = []
            for rpoint in dataset.iterfind('rpoint'):
                rawbundle = []
                use = bconv(rpoint, 'use')
                for rdata in rpoint.iterfind('rawdata'):
                    min_cut = int(rdata.attrib['min_cutoff'])
                    max_cut = int(rdata.attrib['max_cutoff'])
                    t = np.fromstring(rdata.find('time').text, sep=' ')
                    A = np.fromstring(rdata.find('absorbance').text, sep=' ')
                    rawbundle.append((next(use), (min_cut, max_cut),
                                      np.vstack((t, A))))
                thisds.append(rawbundle)
            yield thisds

    tree = ET.parse(filename)
    root = tree.getroot()
    assert root.tag == 'supersod'
    # version = root.attrib['version']

    yield root.find('indicator/name').text
    yield float(root.find('indicator/concentration').text)

    # conc, means, stds, weight, use, rpoints
    placer = ([], [], [], [], [], [])
    iplacer = itertools.cycle(placer)
    for p, item in zip(iplacer, dumpds(root)):
        assert isinstance(p, list)
        p.extend(item)

    for p in placer:
        yield p


def import_data(filename:str, csvformat:int=0):
    '''Open script file and import data therein.

    The import script is a text file with the following information:
    LINE1: indicator(NBT or CYTC) indicator_concentration(float)
    LINE2: blank
    LINE3: path to CSV files
    LINE4+: concentration(float) pattern(regex)
    ...... repeat for each point in titration
    LINE:  end titration with blank line or EOF
    ...... repeat from line 3 for additional titration

    Parameters:
        filename (str): the file name to be read
    Yields:
        str: indicator
        float: concentration of the indicator
        tuple: concentrations, spectra evry point
        tuple: the spectra read
        Null: the a titration ends
    '''
    import os.path

    line_n = 0
    with open(filename, 'r') as fh:
        indic, cindic = fh.readline().split()
        line_n += 1
        if indic not in consts.valid_indicators:
            raise ValueError('Line {}: Indicator unknown.'.format(line_n))
        yield indic
        yield float(cindic)

        line_n += 1
        if fh.readline().strip() != '':
            raise ValueError('Line {}: Expected blank line.'.format(line_n))

        while True:
            line = fh.readline().strip()
            line_n += 1
            if line == '':      # two blank lines ends reading
                break
            if os.path.isabs(line):
                apath = line
            else:
                base = os.path.dirname(filename)
                apath = os.path.join(base, line)

            if not os.path.exists(apath):
                msg = 'Line {}: Folder "{}" not found.'.format(line_n, apath)
                raise FileNotFoundError(msg)

            while True:
                line = fh.readline()
                line_n += 1
                if line.strip() == '':  # blank line marks end of titration
                    break
                c, fpatt = line.split()
                if c.lower() == 'blank':
                    c = 0.0
                _files = wildcard_files(apath, fpatt)
                if not _files:  # len(_files) == 0:
                    msg = 'No files matching for "{}"'.format(fpatt)
                    raise RuntimeError(msg)
                yield float(c), \
                    tuple(load_raw_spectrum(_file, csvformat) for _file in _files)


def wildcard_files(folder, wildcard):
    '''Convert regexp file to list of files matching

    Parameters:
        folder (str): folder to look into
        wildcard (str): regex for target files
    Return:
        tuple: the files found in 'folder' matching 'wildcard'
    '''
    import os
    import re
    return tuple(os.path.join(folder, f)
                 for f in os.listdir(folder)
                 if re.match(wildcard, f))


def numwerr(value, error, template="%%.%df(%%d)"):
    """Given two floats representing a value and an error to that value this
    routine returns an human-readable string.

    Parameters:
        value (float): a number
        error (float): the error associated with **n**

    Returns:
        str: A string representation of value(error)

    Raises:
        ValueError: is arguments are not float numbers or error is negative.

    Example:
        >>> numwerr(10.1234, 0.03)
        "10.12(3)"
    """
    import math
    for a in (value, error):
        if not isinstance(a, float):
            raise ValueError('"%s" must be float' % a)
    if error < 0:
        raise ValueError('Negative errors are meaningless')

    if error == 0.0:
        return str(value)
    else:
        dp = -math.floor(math.log10(error))
        if dp <= 0:
            dp = 0
        s = template % dp
        return s % (value, int(error*10**dp))
