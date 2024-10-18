import os
os.environ['GMX_MAXBACKUP'] = '-1'

import numpy as np
import pandas as pd

import panedr
from glob import glob
from scipy.interpolate import splev, splrep, splint
from subprocess import run
from pathlib import Path
from utils import parse_xvg
import matplotlib.pyplot as plt


def lin(x, a, b):
    return a*x + b


def moving_avg(x, n):
    # cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[n:] - cumsum[:-n]) / float(n)
    return pd.Series(x).rolling(n, min_periods=1).mean().values


def block_avg(x, n_blocks):
    tmp = np.array_split(x, n_blocks)
    # avg = np.vectorize(lambda X: np.mean(X))(tmp)
    avg = np.array([np.mean(X) for X in tmp])
    return avg


def error_est(x, axis=None):
    x = np.array(x.copy())
    if axis is not None:
        N = x.shape[axis]
    else:
        N = len(x.ravel())
    return np.std(x, axis=axis)/(np.sqrt(N))


def error_est_block(x, n_blocks=5):
    blocks = block_avg(x, n_blocks=n_blocks)
    return error_est(blocks)


def drift(x, y):
    k, b = np.polyfit(x, y, 1)
    line = k*x + b
    return line[-1]-line[0], line


def fit_line(x, y):
    k, b = np.polyfit(x, y, 1)
    line = k*x + b
    return line, round(k, 4), round(b, 4)


def integ(x, tck, constant=-1):
    x = np.atleast_1d(x)
    out = np.zeros(x.shape, dtype=x.dtype)
    for n in range(len(out)):
        out[n] = splint(0, x[n], tck)
    out += constant
    return out

# def integ_error(x, delta_y):
#     errors = np.zeros(len(x))
#     for i in range(1, len(x)):
#         errors[i] = integrate.simps(delta_y[0:i], x[0:i])
#     return errors


def integ_error(x, delta_y):
    x = np.diff(x.copy())
    delta_y = delta_y.copy()[1:]
    errors = np.zeros(len(x))
    for i in range(1, len(x)):
        errors[i] = np.sqrt(np.sum(delta_y[0:i]**2 * x[0:i]**2))
    return errors


def spline_integrate(x, y, y_errors, y_std=None, s=None, k=3, N=200):
    if y_std is not None:
        w = 1/y_std
    else:
        w = None
    spl = splrep(x, y, k=k, s=s, w=w)
    spl_x = np.linspace(x[0], x[-1], N)
    spl_y = splev(spl_x, spl)

    spl_int = integ(spl_x, spl)

    spl_int_error = integ_error(x, y_errors)
    spl_err = splrep(x[:-1], spl_int_error)
    spl_y_err = splev(spl_x, spl_err)
    return (spl_x, spl_y), (spl_int, spl_y_err), splint(x[0], x[-1], spl)


def plot_var_avg(x, y, varname, xlabel, ylabel, title, t0=0, n_blocks=5, N=50):
    x = x[t0*5:]
    y = y[t0*5:]

    d, line = drift(x, y)
    RMSD = np.sqrt(np.sum((line - y)**2)/len(y))

    fig, axs = plt.subplots(ncols=2, nrows=1, dpi=100, sharex=True)
    fig_size = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(fig_size[0]*2, fig_size[1])
    fig.suptitle(title)

    axs[0].plot(x, y, alpha=0.5)
    axs[0].plot(x, line, "y")
    axs[0].plot(block_avg(x, N), block_avg(y, N), color="r", marker=".", linestyle="None", alpha=0.6)
    axs[0].axvline(x=500, color="orange", linestyle="dotted")
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].legend([varname, 'fitted line', 'averages'])

    axs[1].plot(x, y, alpha=0.2)
    axs[1].plot(x, line)
    axs[1].set_ylim((line[int(len(line)/2)]-100, line[int(len(line)/2)]+100))
    axs[1].axvline(x=500, color="orange", linestyle="dotted")
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel(ylabel)
    axs[1].legend([varname, 'fitted line'])

    plt.tight_layout()
    plt.show()
    print("Average =", np.mean(y), "| Error =", error_est(block_avg(y, n_blocks)), "| Drift =", d, "| RMSD =", RMSD)


def read_pressure_Lz_sam(folder, gro_files_pattern, columns, start_time=500):
    gro_files = glob(Path(os.path.join(folder, gro_files_pattern)).absolute().__str__())
    gro_files.sort(key=lambda x: float(os.path.basename(x).split("step")[-1].split(".")[0]))

    pdb_file = os.path.join(folder, 'CONF_step0.pdb')
    pdb = pd.read_csv(pdb_file, header=None, skipinitialspace=True, skiprows=4, skipfooter=2, sep=' ')
    idx_r = (pdb[3] == 'DLPC') & (pdb[2] == 'C2')
    pdb[idx_r][1].to_csv(os.path.join(folder, 'DLPC_R.ndx'), header=['[ DLPC_R ]'], index=False)
    idx_l = (pdb[3] == 'DOL') & (pdb[2] == 'C7')
    pdb[idx_l][1].to_csv(os.path.join(folder, 'SAM_L.ndx'), header=['[ SAM_L ]'], index=False)

    posre_files = glob(Path(os.path.join(folder, 'CONF_step*_posre.pdb')).absolute().__str__())
    posre_files.sort(key=lambda x: float(os.path.basename(x).split("step")[-1].split("_")[0]))

    edr_files = [".".join((f[:-4], "edr")) for f in gro_files]

    trr_files = glob(Path(os.path.join(folder, 'MEMBRANE_step*.trr')).absolute().__str__())
    trr_files.sort(key=lambda x: float(os.path.basename(x).split("step")[-1].split(".")[0]))
    coord_l_files = ["_".join((f[:-4], "COORD_L.xvg")) for f in trr_files]
    coord_r_files = ["_".join((f[:-4], "COORD_R.xvg")) for f in trr_files]

    coords_l = np.zeros(np.shape(gro_files))

    coords_r = np.zeros(np.shape(gro_files))
    coords_r_err = np.zeros_like(coords_r)
    coords_posre_r = np.zeros(np.shape(gro_files))

    columns2 = [column+'_Err' for column in columns]
    columns3 = [column+'_std' for column in columns]
    data = {}
    for column in columns+columns2+columns3:
        data[column] = np.zeros(np.shape(gro_files))

    with open(gro_files[-1], 'r') as f:
        tmp = f.read().strip()
    pbc_x, pbc_y, _ = tmp.split('\n')[-1].split()
    data['pbc_x'] = float(pbc_x); data['pbc_y'] = float(pbc_y)

    for i, (edr, gro, posre, trr_file, coord_l_file, coord_r_file) in enumerate(zip(edr_files, gro_files, posre_files, trr_files, coord_l_files, coord_r_files)):
        gmx_traj = 'gmx traj -f {}.xtc -s {}.tpr -n SAM_L.ndx -nox -noy -fp -ox {}.xvg -b {}'
        file_root = os.path.basename(trr_file).split('.')[0]
        cmd = gmx_traj.format(file_root, file_root, os.path.basename(coord_l_file).split('.')[0], start_time)
        run(cmd, shell=True, check=True, cwd=folder, capture_output=True)

        gmx_traj = 'gmx traj -f {}.xtc -s {}.tpr -n DLPC_R.ndx -nox -noy -fp -ox {}.xvg -b {}'
        file_root = os.path.basename(trr_file).split('.')[0]
        cmd = gmx_traj.format(file_root, file_root, os.path.basename(coord_r_file).split('.')[0], start_time)
        run(cmd, shell=True, check=True, cwd=folder, capture_output=True)

        # print(edr)
        edr_data = panedr.edr_to_df(edr)
        for column in columns:
            data[column][i] = np.mean(edr_data[column][edr_data['Time'] >= start_time].values)
            data[column+'_Err'][i] = error_est_block(edr_data[column][edr_data['Time'] >= start_time].values, n_blocks=10)
            data[column+'_std'][i] = np.std(edr_data[column][edr_data['Time'] >= start_time].values)

        pdb = pd.read_csv(posre, header=None, skipinitialspace=True, skiprows=4, skipfooter=2, sep=' ')
        idx_r = (pdb[3] == 'DLPC') & (pdb[2] == 'C2')
        z_avg_r = pdb[idx_r][7].mean()
        coords_posre_r[i] = z_avg_r/10

        _, xvg_data = parse_xvg(coord_l_file)
        tmp = np.mean(xvg_data[:, 1:], axis=1)
        coords_l[i] = np.mean(tmp)

        _, xvg_data = parse_xvg(coord_r_file)
        tmp = np.mean(xvg_data[:, 1:], axis=1)
        coords_r[i] = np.mean(tmp)
        coords_r_err[i] = error_est_block(tmp, n_blocks=5)

    data['coords_posre_r'] = coords_posre_r
    data['coords_r'] = coords_r
    data['coords_r_err'] = coords_r_err
    data['coords_l'] = coords_l
    return data
