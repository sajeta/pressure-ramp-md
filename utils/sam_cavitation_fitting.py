import numpy as np
import os
import panedr
from utils.smoothing import block_avg
from sklearn import linear_model
import mdtraj as md
from more_itertools import locate


def logistic(x, M, k, x0):
    return M/(1 + np.exp(-k * (x - x0)))


def inverted_logistic(x, M, k, x0):
    return M - logistic(x, M, k, x0)


def calculate_A_V(
        DIR: str,
        filename_xtc: str,
        filename_gro: str,
        filename_edr: str,
        data_fraction: float = 0.5,
):
    file_xtc = os.path.join(DIR, filename_xtc)
    file_gro = os.path.join(DIR, filename_gro)
    file_edr = os.path.join(DIR, filename_edr)

    return_data = {}

    data = panedr.edr_to_df(file_edr)
    Times = data['Time'].values
    Volumes = data['Volume'].values

    with open(file_gro, 'r') as f:
        data = f.read().strip().split('\n')
        box_data = [float(i) for i in data[-1].split()]
        pbc_x = box_data[0]
        pbc_y = box_data[1]

    traj = md.load(file_xtc, top=file_gro)
    frames = traj.xyz

    oh_idx = list(locate(traj.topology.atoms, lambda x: any([
        # x.name == 'H22',
        x.name == 'O1',
        ])))

    c10_idx = list(locate(traj.topology.atoms, lambda x: any([
        x.name == 'C10',
        ])))

    ch3_idx = list(locate(traj.topology.atoms, lambda x: any([
        x.name == 'C312',
        # x.name == 'H12X',
        # x.name == 'H12Y',
        # x.name == 'H12Z',

        x.name == 'C212',
        # x.name == 'H12R',
        # x.name == 'H12T',
        # x.name == 'H12S',
    ])))

    N = []
    for frame in frames:
        oh_atoms_xyz = frame[oh_idx]
        c10_atoms_xyz = frame[c10_idx]
        ch3_atoms_xyz = frame[ch3_idx]
        n = 0
        for ch3_atom_xyz in ch3_atoms_xyz:
            n += len(np.where(
                (np.linalg.norm(ch3_atom_xyz - oh_atoms_xyz, axis=1) < 0.7) |
                (np.linalg.norm(ch3_atom_xyz - c10_atoms_xyz, axis=1) < 0.7)
            )[0])
        N.append(n)
    N = np.array(N)

    if len(Times) > len(N):
        Times = block_avg(Times, len(N))
        Volumes = block_avg(Volumes, len(N))
    elif len(N) > len(Times):
        N = block_avg(N, len(Times))

    # f = lambda X: np.linalg.norm(inverted_logistic(x=Times, M=X[1], k=0.03, x0=X[0]) - N)
    # # sol = least_squares(f, x0=np.array([np.mean(Times), np.max(N)], dtype=np.float64), loss='linear')
    # sol = minimize(f, x0=np.array([np.mean(Times), np.max(N)], dtype=np.float64))
    # logistic_fit = inverted_logistic(Times, M=sol.x[1], k=0.03, x0=sol.x[0])
    # i = np.argmin(np.abs(logistic_fit - sol.x[1]*0.99))
    #
    # t = np.array(Times[:i])
    # n = np.array(N[:i])

    i = int(len(Times)*data_fraction)
    t = np.array(Times[:i])
    n = np.array(N[:i])

    tms = np.linspace(0, Times[-1], len(N))

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), n.reshape(-1, 1), sample_weight=None)
    N_linfit = ransac.predict(tms.reshape(-1, 1)).ravel()
    delta_N = N_linfit - N

    t = np.array(Times[:i])
    V = np.array(Volumes[:i])

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), V.reshape(-1, 1), sample_weight=None)
    V_linfit = ransac.predict(Times.reshape(-1, 1)).ravel()
    delta_V = Volumes - V_linfit

    Axy = pbc_x * pbc_y
    A = Axy * delta_N/N_linfit

    return_data['A'] = A
    return_data['delta_V'] = delta_V
    return_data['Times'] = Times
    return_data['Volumes'] = Volumes
    return_data['N'] = N
    return_data['N_linfit'] = N_linfit
    return_data['delta_N'] = delta_N
    return_data['V_linfit'] = V_linfit
    return_data['pbc_x'] = pbc_x
    return_data['pbc_y'] = pbc_y
    return return_data


def calculate_A_V_multi_files(
        DIR: str,
        filenames_xtc: list[str],
        filenames_gro: list[str],
        filenames_edr: list[str],
):
    data_complete = {}

    Times_ = []
    Volumes_ = []
    N_ = []

    for i, (filename_xtc, filename_gro, filename_edr) in enumerate(zip(filenames_xtc, filenames_gro, filenames_edr)):
        # print(filename_xtc)
        file_xtc = os.path.join(DIR, filename_xtc)
        file_gro = os.path.join(DIR, filename_gro)
        file_edr = os.path.join(DIR, filename_edr)

        data = panedr.edr_to_df(file_edr)
        Times = data['Time'].values
        Volumes = data['Volume'].values

        with open(file_gro, 'r') as f:
            data = f.read().strip().split('\n')
            box_data = [float(i) for i in data[-1].split()]
            pbc_x = box_data[0]
            pbc_y = box_data[1]

        traj = md.load(file_xtc, top=file_gro)
        frames = traj.xyz

        oh_idx = list(locate(traj.topology.atoms, lambda x: any([
            # x.name == 'H22',
            x.name == 'O1',
            ])))

        c10_idx = list(locate(traj.topology.atoms, lambda x: any([
            x.name == 'C10',
            ])))

        ch3_idx = list(locate(traj.topology.atoms, lambda x: any([
            x.name == 'C312',
            # x.name == 'H12X',
            # x.name == 'H12Y',
            # x.name == 'H12Z',

            x.name == 'C212',
            # x.name == 'H12R',
            # x.name == 'H12T',
            # x.name == 'H12S',
        ])))

        N = []
        for frame in frames:
            oh_atoms_xyz = frame[oh_idx]
            c10_atoms_xyz = frame[c10_idx]
            ch3_atoms_xyz = frame[ch3_idx]
            n = 0
            for ch3_atom_xyz in ch3_atoms_xyz:
                n += len(np.where(
                    (np.linalg.norm(ch3_atom_xyz - oh_atoms_xyz, axis=1) < 0.7) |
                    (np.linalg.norm(ch3_atom_xyz - c10_atoms_xyz, axis=1) < 0.7)
                )[0])
            N.append(n)
        N = np.array(N)

        if len(Times) > len(N):
            Times = block_avg(Times, len(N))
            Volumes = block_avg(Volumes, len(N))
        elif len(N) > len(Times):
            N = block_avg(N, len(Times))

        Times_.append(Times + i*Times[-1])
        Volumes_.append(Volumes)
        N_.append(N)

    Times_ = np.concatenate(Times_)
    Volumes_ = np.concatenate(Volumes_)
    N_ = np.concatenate(N_)

    # f = lambda X: np.linalg.norm(inverted_logistic(x=Times_, M=X[1], k=0.03, x0=X[0]) - N_)
    # # sol = least_squares(f, x0=np.array([np.mean(Times_), np.max(N_)], dtype=np.float64), loss='linear')
    # sol = minimize(f, x0=np.array([np.mean(Times_), np.max(N_)], dtype=np.float64))
    # logistic_fit = inverted_logistic(Times_, M=sol.x[1], k=0.03, x0=sol.x[0])
    # i = np.argmin(np.abs(logistic_fit - sol.x[1]*0.99))

    # t = np.array(Times_[:i])
    # n = np.array(N_[:i])
    t = Times_
    n = N_

    tms = np.linspace(0, Times_[-1], len(N_))

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), n.reshape(-1, 1), sample_weight=None)
    N_linfit = ransac.predict(tms.reshape(-1, 1)).ravel()
    delta_N = N_linfit - N_

    t = np.array(Times_[:500])
    V = np.array(Volumes_[:500])

    ransac = linear_model.RANSACRegressor(loss='absolute_error')
    ransac.fit(t.reshape(-1, 1), V.reshape(-1, 1), sample_weight=None)
    V_linfit = ransac.predict(Times_.reshape(-1, 1)).ravel()
    delta_V = Volumes_ - V_linfit

    Axy = pbc_x * pbc_y
    A = Axy * delta_N/N_linfit

    data_complete['A'] = A
    data_complete['delta_V'] = delta_V
    data_complete['Times'] = Times_
    data_complete['Volumes'] = Volumes_
    data_complete['N'] = N_
    data_complete['N_linfit'] = N_linfit
    data_complete['delta_N'] = delta_N
    data_complete['V_linfit'] = V_linfit
    data_complete['pbc_x'] = pbc_x
    data_complete['pbc_y'] = pbc_y

    return data_complete
