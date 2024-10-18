import os, sys
import re
import shlex
import numpy as np
from pathlib import Path
from subprocess import Popen, PIPE
import panedr
from MDAnalysis.auxiliary.XVG import XVGReader
from glob import glob
import pickle
import mdtraj as md


PRINT_FLUSH = True


# noinspection DuplicatedCode
def parse_xvg(fname):
    """Parses XVG file legends and data"""

    _ignored = {'legend', 'view'}
    _re_series = re.compile('s[0-9]+$')
    _re_xyaxis = re.compile('[xy]axis$')

    metadata = {}
    num_data = []

    metadata['labels'] = {}
    metadata['labels']['series'] = []

    ff_path = os.path.abspath(fname)
    if not os.path.isfile(ff_path):
        raise IOError('File not readable: {0}'.format(ff_path))

    with open(ff_path, 'r') as fhandle:
        for line in fhandle:
            line = line.strip()
            if line.startswith('@'):
                tokens = shlex.split(line[1:])
                if tokens[0] in _ignored:
                    continue
                elif tokens[0] == 'TYPE':
                    if tokens[1] != 'xy':
                        raise ValueError('Chart type unsupported: \'{0}\'. Must be \'xy\''.format(tokens[1]))
                elif _re_series.match(tokens[0]):
                    metadata['labels']['series'].append(tokens[-1])
                elif _re_xyaxis.match(tokens[0]):
                    metadata['labels'][tokens[0]] = tokens[-1]
                elif len(tokens) == 2:
                    metadata[tokens[0]] = tokens[1]
                else:
                    print('Unsupported entry: {0} - ignoring'.format(tokens[0]), file=sys.stderr)
            elif line[0].isdigit():
                num_data.append(list(map(float, line.split())))

    if not metadata['labels']['series']:
        for series in range(len(num_data) - 1):
            metadata['labels']['series'].append('')

    return metadata, np.array(num_data)


# def xvg_to_pd(fname):
#     """Parses XVG file legends and data"""
#
#     _ignored = {'view'}
#
#     metadata = {}
#     num_data = []
#
#     metadata['labels'] = {}
#     metadata['labels']['series'] = []
#
#     ff_path = os.path.abspath(fname)
#     if not os.path.isfile(ff_path):
#         raise IOError('File not readable: {0}'.format(ff_path))
#
#     with open(ff_path, 'r') as fhandle:
#         for line in fhandle:
#             line = line.strip()
#             if not line.startswith('#'):
#                 if line.startswith('@'):
#                     tokens = line.split()
#                     if tokens[0] == 'TYPE':
#                         if tokens[1] != 'xy':
#                             raise ValueError('Chart type unsupported: \'{0}\'. Must be \'xy\''.format(tokens[1]))
#
#                 elif line[0].isdigit():
#                     num_data.append(list(map(float, line.split())))
#
#     if not metadata['labels']['series']:
#         for series in range(len(num_data) - 1):
#             metadata['labels']['series'].append('')
#
#     return metadata, np.array(num_data)


def read_density(file_list):
    file_list.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True)
    Times = np.array([])
    Volumes = np.array([])
    Errors = []

    for file in file_list:
        file = os.path.join(Path(file))
        try:
            data = panedr.edr_to_df(file)
            Times_i = data['Time'].values
            Volumes_i = data['Volume'].values
        except ValueError as err1:
            if str(err1) == "Something went wrong":
                Errors.append("WARNING: File " + str(file) + " is corrupted")
                gmx_box_cmd = "gmx traj -f {trr} -s {tpr} -ob {box}"
                trr_file = ".".join((file.split(".")[0], ".trr"))
                tpr_file = ".".join((file.split(".")[0], ".tpr"))
                box_file = ".".join((file.split(".")[0], "_BOX.xvg"))
                try:
                    cmd = gmx_box_cmd.format(trr=trr_file, tpr=tpr_file, box=box_file)
                    print("Running command:", cmd)
                    p = Popen(cmd, shell=True, cwd=os.path.dirname(file), stdout=PIPE, stdin=PIPE, stderr=PIPE)
                    p.communicate(input=b"0")
                    p.wait()
                    if p.returncode != 0:
                        subproc_error_mssg = p.communicate()[1].decode()
                        raise Exception(subproc_error_mssg)

                    data = XVGReader(box_file)
                    Times_i = data._auxdata_values[:, 0]
                    Volumes_i = data._auxdata_values[:, 1] * data._auxdata_values[:, 2] * data._auxdata_values[:, 3]
                except Exception as err2:
                    # noinspection PyUnboundLocalVariable
                    if str(err2) == subproc_error_mssg:
                        Errors.append("ERROR: File " + str(box_file) + " is corrupted")
                        data = np.array([])
                        Times_i = np.array([])
                        Volumes_i = np.array([])
                    else:
                        raise err2
            else:
                raise err1
        if len(data) != 0:
            if len(Times) != 0:
                Times = np.concatenate((Times, Times_i + Times[-1]+1))
            else:
                Times = np.concatenate((Times, Times_i))
            Volumes = np.concatenate((Volumes, Volumes_i))
            print(f"File {file} loaded", flush=PRINT_FLUSH)

    if len(Errors) != 0:
        with open(os.path.join(os.path.dirname(file_list[0]), "ERRORS.log"), "w") as log:
            log.write("\n".join(Errors))

    Times = Times / 10**3
    return Times, Volumes


def a_starts_with_b(a, b):
    return b == a[:len(b)]


def read_pressure(data_dir, keys_to_return=('Pressure',), pattern='DECANE_-*.edr'):
    return_data = dict([(i, []) for i in ['Time'] + list(keys_to_return)])

    path = os.path.join(data_dir, pattern)
    files = glob(path)
    files.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)

    last_time = 0
    for file in files:
        try:
            data = panedr.edr_to_df(file)
            return_data['Time'].append(data['Time'].values + last_time)
            last_time = return_data['Time'][-1][-1]

            for key in keys_to_return:
                return_data[key].append(data[key].values)
        except EOFError:
            print(f'WARNING: Error opening file: {file}')

    return dict([(k, np.concatenate(return_data[k])[1:]) for k in return_data])


def load_radial_density(data_dir, pattern='DECANE_-*', recalculate=False):
    cache_file = os.path.join(data_dir, 'RADIAL_DENSITY_CACHE.pkl')

    if not recalculate and os.path.exists(cache_file):
        print(f'Loading from cache: {data_dir}')
        with open(cache_file, 'rb') as f:
            cached_data = pickle.load(f)
        return cached_data['radii'], cached_data['densities'], cached_data['pressures']

    pattern_xtc = '.'.join((pattern.split('.')[0], 'xtc'))
    path_xtc = os.path.join(data_dir, pattern_xtc)

    files_xtc = glob(path_xtc)
    files_xtc.sort(key=lambda x: int(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
    files_xtc = [os.path.join(data_dir, 'NPT.xtc')] + files_xtc[:-1]
    # files_xtc = [os.path.join(data_dir, 'NPT.xtc')] + files_xtc
    files_gro = ['.'.join((i.split('.')[0], 'gro')) for i in files_xtc]

    radii = []
    densities = []
    pressures = []

    for file_xtc, file_gro in zip(files_xtc, files_gro):
        print('Loading file:', file_xtc)
        if os.path.basename(file_xtc).split('.')[0] == 'NPT':
            pressure = 1
        else:
            pressure = int(os.path.basename(file_xtc).split('.')[0].split('_')[-1])
        pressures.append(pressure)

        traj = md.load(file_xtc, top=file_gro)
        # select oxygen atom indices:
        decane = traj.top.select('resname DEC')
        system = traj.top.select('all')

        # calculate masses of all atoms:
        # masses = np.array([i.element.mass for i in traj.topology.atoms])

        # iterate through all frames and save the distances of each atom to the centre of mass:
        d = list()
        for frame in traj.xyz:
            decane_coords = frame[decane]
            all_coords = frame[system]
            # COM of whole system:
            # centre_of_mass = np.average(frame, axis=0, weights=masses)
            # COM of just the oxygens:
            centre_of_mass = decane_coords.mean(0)

            vectors = (all_coords - centre_of_mass)
            distances = np.linalg.norm(vectors, axis=1)
            d += list(distances)

        # Take a histogram:
        counts, lengths = np.histogram(d, bins=100)
        # normalize 'counts' by averaging across the number of trajectory frames:
        counts = counts / len(traj)
        # calculate the volume of each spherical shell:
        shell_volumes = 4/3*np.pi * (lengths[1:]**3 - lengths[:-1]**3)
        # normalize 'counts' by the volume of each shell:
        counts = counts / shell_volumes

        radii.append((lengths[:-1]+lengths[1:])/2)
        densities.append(counts)
    pressures = np.array(pressures)

    cached_data = {
        'radii': radii,
        'densities': densities,
        'pressures': pressures,
    }
    with open(cache_file, 'wb') as f:
        pickle.dump(cached_data, f)
        print(f'Saved to cache: {data_dir}')

    print(f'Finished loading: {data_dir}')
    return radii, densities, pressures


def read_p_cav_from_cache(data_dir):
    with open(os.path.join(data_dir, 'RESULTS.pkl'), 'rb') as f:
        data = pickle.load(f)
    return data['p_star']


def numeric_derivative(y, x):
    dx = np.diff(x)
    dy = np.diff(y)
    return dy/dx


def error_est(x):
    return np.std(x)/(np.sqrt(len(x)))


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
        return False


def lin(x, a, b):
    return a*x + b
