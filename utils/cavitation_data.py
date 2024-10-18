import os
import pickle
from glob import glob
import numpy as np
from sklearn import linear_model
from pathlib import Path
from MDAnalysis.auxiliary.XVG import XVGReader

from utils import read_density, error_est


def calculate_cavitation_data(folder, p_rate, p_start):
    file_list = glob(os.path.join(folder, "DECANE_-*.edr"))
    file_list.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True)
    Times, Volumes = read_density(file_list)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Times.reshape(-1, 1), Volumes.reshape(-1, 1))
    # Volumes_linfit = ransac.predict(Times.reshape(-1, 1)).ravel()
    inlier_mask = ransac.inlier_mask_
    t_star = Times[inlier_mask][-1]

    surf_tens = 230
    V_star = Volumes[inlier_mask][-1]
    p_star = t_star * p_rate + p_start
    r_bubble_star = -2*surf_tens/p_star
    V_bubble_star = 4/3 * np.pi * r_bubble_star**3

    return {"t_star": t_star, "V_star": V_star, "p_star": p_star,
            "r_bubble_star": r_bubble_star, "V_bubble_star": V_bubble_star}


def read_cavitation_data(folder, recalculate=False):
    if os.path.exists(os.path.join(folder, "RESULTS.pkl")) and recalculate is False:
        with open(os.path.join(folder, "RESULTS.pkl"), "rb") as pkl:
            data = pickle.load(pkl)
    else:
        with open(os.path.join(folder, "SYSTEM_CONFIGURATION.pickle"), "rb") as pkl:
            SYSTEM_CONFIGURATION = pickle.load(pkl)
        data = calculate_cavitation_data(folder, SYSTEM_CONFIGURATION['P_RATE'], SYSTEM_CONFIGURATION['P_START'])
        with open(os.path.join(folder, "RESULTS.pkl"), "wb") as pkl:
            pickle.dump(data, pkl)
    return data


def read_pbc_box_volume(file_list):
    file_list.sort(key=lambda x: float(os.path.basename(x).split("_")[1]), reverse=True)

    times = np.array([])
    volumes = np.array([])

    for file in file_list:
        file = os.path.join(Path(file))
        data = XVGReader(file)
        times_i = data._auxdata_values[:, 0]
        if len(times) != 0:
            times = np.concatenate((times, times_i + times[-1]+1))
        else:
            times = np.concatenate((times, times_i))
        volumes_i = data._auxdata_values[:, 1] * data._auxdata_values[:, 2] * data._auxdata_values[:, 3]
        volumes = np.concatenate((volumes, volumes_i))
        print(f"File {file} loaded")

    times = times / 10**3
    return times, volumes


def read_cav_rates_pressures(path):
    # LOAD DECANE DATA
    rate_folers = glob(f"{path}/p_rate_-*")
    rate_folers.sort(key=lambda x: float(os.path.basename(x).split("_")[-1]), reverse=True)

    P_RATES = np.zeros(len(rate_folers))
    P_STARS = np.zeros(len(rate_folers))
    P_STARS_ERR = np.zeros(len(rate_folers))
    P_STARS_ALL = []
    V_STARS = np.zeros(len(rate_folers))
    V_STARS_ERR = np.zeros(len(rate_folers))
    T_STARS = np.zeros(len(rate_folers))
    T_STARS_ERR = np.zeros(len(rate_folers))

    for i, rate_foler in enumerate(rate_folers):
        print(rate_foler)
        run_folders = glob(f"{rate_foler}/run_*")
        run_folders.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=False)

        P_RATES[i] = float(rate_foler.split("_")[-1])
        p_stars = []
        P_STARS_ALL.append(p_stars)
        v_stars = []
        t_stars = []
        for j, rf in enumerate(run_folders):
            if os.path.exists(os.path.join(rf, 'RESULTS.log')):
                print(rf)
                tmp = read_cavitation_data(rf, recalculate=False)
                p_stars.append(tmp['p_star'])
                v_stars.append(tmp['V_star'])
                t_stars.append(tmp['t_star'])

        P_STARS[i] = np.mean(p_stars)
        P_STARS_ERR[i] = error_est(p_stars)
        V_STARS[i] = np.mean(v_stars)
        V_STARS_ERR[i] = error_est(v_stars)
        T_STARS[i] = np.mean(t_stars)
        T_STARS_ERR[i] = error_est(t_stars)

    # convert to SI units
    P_RATES = P_RATES.copy() * 10**5 * 10**9
    P_STARS = P_STARS.copy() * 10**5
    P_STARS_ERR = P_STARS_ERR.copy() * 10**5

    return P_RATES, P_STARS, P_STARS_ERR
