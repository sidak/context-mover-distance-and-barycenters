import os
import timeit
import time
import datetime
import pickle
import numpy as np


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result


    return timed


def has_keyword_in_filenames(root, keyword):
    for file in os.listdir(root):
        if (keyword in file):
            return True
    return False


def pickle_obj(obj, path, mode="wb", protocol=pickle.HIGHEST_PROTOCOL):
    '''
    Pickle object 'obj' and dump at 'path' using specified 
    'mode' and 'protocol'
    Returns time taken to pickle
    '''

    import time
    st_time = time.perf_counter()
    pkl_file = open(path, mode)
    pickle.dump(obj, pkl_file, protocol=protocol)
    end_time = time.perf_counter()
    return (end_time - st_time)


def joblib_obj(obj, path, mode=None, protocol=None):
    '''
    Use joblib to dump object 'obj' at 'path' using specified
    Returns time taken to dump
    '''

    import time
    from sklearn.externals import joblib
    st_time = time.perf_counter()
    joblib.dump(obj, path)
    end_time = time.perf_counter()
    return (end_time - st_time)


def sparse_obj(obj, path, mode=None, protocol=None):
    '''
    Dump the co-occurrence matrix 'obj' at 'path' using scipy sparse
    Returns time taken to dump
    '''
    import time
    import scipy.sparse as sp
    st_time = time.perf_counter()
    sp.save_npz(path, obj)
    end_time = time.perf_counter()
    return (end_time - st_time)


def load_joblib(path, nick=""):
    from sklearn.externals import joblib
    st_time = time.perf_counter()
    obj = joblib.load(path)
    end_time = time.perf_counter()
    print("Loaded " + nick + " in " + str(end_time - st_time) + " seconds")
    return obj


def load_pickle(path, nick=""):
    st_time = time.perf_counter()
    obj = pickle.load(open(path, 'rb'))
    end_time = time.perf_counter()
    print("Loaded " + nick + " in " + str(end_time - st_time) + " seconds")
    return obj


def load_sparse(path, nick=""):
    import scipy.sparse as sp
    st_time = time.perf_counter()
    obj = sp.load_npz(path)
    end_time = time.perf_counter()
    print("Loaded " + nick + " in " + str(end_time - st_time) + " seconds")
    return obj


def load_cooc(path, joblib=False, sparse=False, nick=""):
    if joblib:
        return load_joblib(path, nick=nick)
    elif sparse:
        return load_sparse(path, nick=nick)
    else:
        return load_pickle(path, nick=nick)


def test_utils():
    print("yo baby")


def get_timestamp():
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H:%M:%S')
    return timestamp


def get_timestamp_other():
    ts = time.time()
    timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d_%H-%M-%S')
    return timestamp


'''
Below code for unit_vector and angle_between is taken from stackoverflow
'''


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
