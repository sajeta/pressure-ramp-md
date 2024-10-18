import numpy as np
from glob import glob
import os
from subprocess import run, Popen, PIPE
from utils import parse_xvg
from sklearn import linear_model
from numba import jit, njit


def calculate_A_V(DIR, file_pattern='DECANE_-*.gro'):
    @jit(nopython=True, parallel=False)
    def do_stuff(coord_data):
        n_0 = []
        for row_l, row_r in coord_data:
            n = 0
            for x1 in row_r.reshape(-1, 3):
                for x2 in row_l.reshape(-1, 3):
                    if np.linalg.norm(x1-x2) < 0.7:
                        n += 1
            n_0.append(n)
        return np.array(n_0)

    DIR = os.path.abspath(DIR)
    tpr_files = glob(os.path.join(DIR, file_pattern))
    if len(tpr_files) > 1:
        tpr_files.sort(key=lambda x: float(os.path.basename(x).split('_')[-1].split('.')[0]), reverse=True)
        tpr_files = tpr_files[:-10]
    xtc_files = [tpr_file.split('.')[0] + '.xtc' for tpr_file in tpr_files]
    box_files = [tpr_file.split('.')[0] + '_BOX.xvg' for tpr_file in tpr_files]
    coord_l_files = [tpr_file.split('.')[0] + '_COORD_L.xvg' for tpr_file in tpr_files]
    coord_r_files = [tpr_file.split('.')[0] + '_COORD_R.xvg' for tpr_file in tpr_files]

    gmx_traj_box = 'gmx traj -f {xtc} -s {tpr} -fp -ob {xvg}'
    gmx_traj_coord = 'gmx traj -f {xtc} -s {tpr} -n {ndx} -fp -ox {xvg}'
    N = []
    t = []
    V = []

    if not os.path.exists(box_files[0]):
        cmd = gmx_traj_box.format(xtc=os.path.basename(xtc_files[0]), tpr=os.path.basename(tpr_files[0]), xvg=os.path.basename(box_files[0]))
        print("Running command:", cmd)
        # run(cmd, shell=True, check=True, cwd=DIR, capture_output=True)
        p = Popen(cmd, shell=True, cwd=DIR, stdout=PIPE, stdin=PIPE, stderr=PIPE)
        p.communicate(input=b"0")
        p.wait()
        if p.returncode != 0:
            subproc_error_mssg = p.communicate()[1].decode()
            raise Exception(subproc_error_mssg)

    _, box_data = parse_xvg(box_files[0])
    pbc_x = box_data[0, 1]
    pbc_y = box_data[0, 2]
    Axy = pbc_x * pbc_y

    for box_file, coord_l_file, coord_r_file, xtc_file, tpr_file in zip(box_files, coord_l_files, coord_r_files, xtc_files, tpr_files):
        if not os.path.exists(box_file):
            cmd = gmx_traj_box.format(xtc=os.path.basename(xtc_file), tpr=os.path.basename(tpr_file), xvg=os.path.basename(box_file))
            print("Running command:", cmd)
            # run(cmd, shell=True, check=True, cwd=DIR, capture_output=True)
            p = Popen(cmd, shell=True, cwd=DIR, stdout=PIPE, stdin=PIPE, stderr=PIPE)
            p.communicate(input=b"0")
            p.wait()
            if p.returncode != 0:
                subproc_error_mssg = p.communicate()[1].decode()
                raise Exception(subproc_error_mssg)

        if not os.path.exists(coord_l_file):
            cmd = gmx_traj_coord.format(xtc=os.path.basename(xtc_file), tpr=os.path.basename(tpr_file), ndx='CH3_L.ndx', xvg=os.path.basename(coord_l_file))
            run(cmd, shell=True, check=True, cwd=DIR, capture_output=True)

        if not os.path.exists(coord_r_file):
            cmd = gmx_traj_coord.format(xtc=os.path.basename(xtc_file), tpr=os.path.basename(tpr_file), ndx='CH3_R.ndx', xvg=os.path.basename(coord_r_file))
            run(cmd, shell=True, check=True, cwd=DIR, capture_output=True)

        meta_l, data_l = parse_xvg(coord_l_file)
        meta_r, data_r = parse_xvg(coord_r_file)

        if len(t) == 0:
            t_0 = data_l[:, 0]
        else:
            t_0 = data_l[:, 0] + t[-1]
        t.append(t_0)

        coord_l_data = data_l[:, 1:]
        coord_r_data = data_r[:, 1:]
        # n_0 = []
        # for row_l, row_r in zip(coord_l_data, coord_r_data):
        #     n = 0
        #     for x1 in row_r.reshape(-1, 3):
        #         for x2 in row_l.reshape(-1, 3):
        #             if np.linalg.norm(x1-x2) < 0.7:
        #                 n += 1
        #     n_0.append(n)
        coord_data = list(zip(coord_l_data, coord_r_data))
        n_0 = do_stuff(coord_data)
        N.append(n_0)

        _, box_data = parse_xvg(box_file)
        V.append(box_data[:, 1]*box_data[:, 2]*box_data[:, 3])

    N = np.array(N).ravel()
    t = np.array(t).ravel()
    V = np.array(V).ravel()

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), N.reshape(-1, 1), sample_weight=None)
    N_linfit = ransac.predict(t.reshape(-1, 1)).ravel()
    delta_N = N_linfit - N

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), V.reshape(-1, 1), sample_weight=None)
    V_linfit = ransac.predict(t.reshape(-1, 1)).ravel()
    delta_V = V - V_linfit

    A = Axy * delta_N/N_linfit

    return A, delta_V, t, N, (pbc_x, pbc_y)
