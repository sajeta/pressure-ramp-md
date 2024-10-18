#%%
import os, sys
from pathlib import Path
from subprocess import run, CalledProcessError
import numpy as np
import argparse
from glob import glob
import matplotlib.pyplot as plt
from time import time, sleep
import shutil
import configparser
import pandas as pd
# from scipy.optimize import curve_fit
import pickle
import datetime
from utils import read_density
from sklearn import linear_model
from more_itertools import locate

# plt.style.use('ggplot')

#%%


parser = argparse.ArgumentParser(description='Perform pressure ramp molecular dynamics simulations.')
parser.add_argument('-wd', '--workdir', metavar='workdir', type=str, required=True)
parser.add_argument('-mdrun_args', '--mdrun_arguments', metavar='mdrun_args', type=str)
parser.add_argument('-cr', '--continue_run', metavar='continue_run', type=int, default=0, choices=[0, 1])
parser.add_argument('-an', '--analysis', action='store_true')
parser.add_argument('-log', '--log', action='store_true')
parser.add_argument('-config', '--config', metavar='config', type=str)
parser.add_argument('-r', '--restraint', metavar='restraint', type=str, choices=['None', 'Fixed', 'Moving'])

args = parser.parse_args()
args_dict = args.__dict__

WORKDIR = args.workdir
CONTINUE_RUN = bool(args.continue_run)
ANALYSIS = bool(args.analysis)
CONFIG_FILE = args.config
RESTRAINT_TYPE = args.restraint

MANUAL_STOP = os.environ.get('MANUAL_STOP') == 'True'


# P_STRING = 'ref_p                   = {} {}                ; reference pressure, in bar'
P_STRING = 'ref_p                   = {}                ; reference pressure, in bar'
GEN_VEL_STR = "gen-vel                  = {}"
GEN_SEED_STR = "gen_seed                 = {}"
CONTINUING_STR = "continuation             = {}"
NSTEPS_STR = "nsteps                   = {}    ; run time = dt * nsteps (ps)"

if CONFIG_FILE is not None:
    config = configparser.ConfigParser()
    config.read(CONFIG_FILE)
    P_RATE = eval(config['PRESSURE']['P_RATE'])
    DELTA_P = eval(config['PRESSURE']['DELTA_P'])
    P_MAX = eval(config['PRESSURE']['P_MAX'])
    P_START = eval(config['PRESSURE']['P_START'])
    dt = eval(config['PRESSURE']['dt'])
    DELTA_T = DELTA_P/P_RATE
    SYSTEM_CONFIGURATION = {"P_RATE": P_RATE, "DELTA_P": DELTA_P, "P_MAX": P_MAX, "P_START": P_START,
                            "dt": dt, "DELTA_T": DELTA_T}
SYS_CONF_PKL_NAME = "SYSTEM_CONFIGURATION.pickle"

PRINT_FLUSH = True
MAXWARN = 3


def a_starts_with_b(a, b):
    return b == a[:len(b)]


def identify_str(s):
    if a_starts_with_b(a=s, b=P_STRING.split(" ")[0]):
        return "P_STRING"
    elif a_starts_with_b(a=s, b=GEN_VEL_STR.split(" ")[0]):
        return "GEN_VEL_STR"
    elif a_starts_with_b(a=s, b=GEN_SEED_STR.split(" ")[0]):
        return "GEN_SEED_STR"
    elif a_starts_with_b(a=s, b=NSTEPS_STR.split(" ")[0]):
        return "NSTEPS_STR"
    elif a_starts_with_b(a=s, b=CONTINUING_STR.split(" ")[0]):
        return "CONTINUING_STR"
    else:
        return 0


def moving_avg(x, n):
    # cumsum = np.cumsum(np.insert(x, 0, 0))
    # return (cumsum[n:] - cumsum[:-n]) / float(n)
    return pd.Series(x).rolling(n, min_periods=1).mean().values


def block_avg(x, n_blocks):
    tmp = np.array_split(x, n_blocks)
    # avg = np.vectorize(lambda X: np.mean(X))(tmp)
    avg = np.array([np.mean(X) for X in tmp])
    return avg


def numeric_derivative(y, x):
    dx = np.diff(x)
    dy = np.diff(y)
    return dy/dx


def lin(x, a, b):
    return a*x + b


def make_md_file(prev_file, next_file, p, gen_vel=None, run_time=None, continuing=None):
    with open(os.path.join(Path(prev_file)), "r") as f:
        prev_md_file = f.read()
    prev_md_data = prev_md_file.split("\n")

    idxs = [identify_str(i) for i in prev_md_data]
    P_IDX, GEN_VEL_IDX, GEN_SEED_IDX, NSTEPS_IDX, CONTINUING_IDX = \
        idxs.index("P_STRING"), idxs.index('GEN_VEL_STR'), idxs.index('GEN_SEED_STR'), idxs.index('NSTEPS_STR'), idxs.index('CONTINUING_STR')

    next_md_data = prev_md_data.copy()

    pcoupltype_idx = list(locate(next_md_data, lambda x: x.startswith('pcoupltype')))
    pcoupltype = next_md_data[pcoupltype_idx[0]]

    if 'semiisotropic' in pcoupltype.split():
        P_STRING = 'ref_p                   = {} {}                ; reference pressure, in bar'
        next_md_data[P_IDX] = P_STRING.format('1.0', p)

    elif 'isotropic' in pcoupltype.split():
        P_STRING = 'ref_p                   = {}                ; reference pressure, in bar'
        next_md_data[P_IDX] = P_STRING.format(p)

    next_md_data[GEN_SEED_IDX] = GEN_SEED_STR.format(np.random.randint(low=1000, high=9999))

    if gen_vel is not None:
        if gen_vel:
            next_md_data[GEN_VEL_IDX] = GEN_VEL_STR.format('yes')
        else:
            next_md_data[GEN_VEL_IDX] = GEN_VEL_STR.format('no')

    if continuing is not None:
        if continuing:
            next_md_data[CONTINUING_IDX] = CONTINUING_STR.format('yes')
        else:
            next_md_data[CONTINUING_IDX] = CONTINUING_STR.format('no')

    if run_time is not None:
        nsteps = int(np.ceil(run_time / dt))
        next_md_data[NSTEPS_IDX] = NSTEPS_STR.format(nsteps)

    next_md_file = "\n".join(next_md_data)
    with open(os.path.join(Path(next_file)), "w") as f:
        f.write(next_md_file)


# noinspection DuplicatedCode
def run_em(folder):
    file_dict = {"deffnm": "EM"}
    folder = os.path.join(Path(folder))

    grompp_em_cmd = f"gmx grompp -f minim.mdp -c conf.gro -p topol.top -o {file_dict['deffnm']}.tpr -maxwarn {MAXWARN}"
    # gmx grompp -f minim.mdp -c conf.gro -p topol.top -o em.tpr -maxwarn 1 &> gromacs.log
    mdrun_em_cmd = f"gmx mdrun -nt 1 -deffnm {file_dict['deffnm']}"
    # gmx mdrun -nt 24 -deffnm em &>> gromacs.log

    print("Running command:", grompp_em_cmd, flush=PRINT_FLUSH)
    run(grompp_em_cmd, shell=True, check=True, cwd=folder)

    print("Running command:", mdrun_em_cmd, flush=PRINT_FLUSH)
    run(mdrun_em_cmd, shell=True, check=True, cwd=folder)


# noinspection DuplicatedCode
def run_nvt(folder):
    file_dict = {"deffnm": "NVT"}
    folder = os.path.join(Path(folder))

    grompp_nvt_cmd = f"gmx grompp -f nvt.mdp -c EM.gro -p topol.top -o {file_dict['deffnm']}.tpr -maxwarn {MAXWARN}"
    # gmx grompp -f nvt.mdp -c EM.gro -p topol.top -o NVT.tpr -maxwarn 1
    mdrun_nvt_cmd = f"gmx mdrun {args.mdrun_arguments} -deffnm {file_dict['deffnm']}"
    # gmx mdrun -nt 4 -deffnm NVT

    print("Running command:", grompp_nvt_cmd, flush=PRINT_FLUSH)
    run(grompp_nvt_cmd, shell=True, check=True, cwd=folder)

    print("Running command:", mdrun_nvt_cmd, flush=PRINT_FLUSH)
    run(mdrun_nvt_cmd, shell=True, check=True, cwd=folder)


# noinspection DuplicatedCode
def run_equilibration(folder):
    folder = os.path.join(Path(folder))

    file_dict = {"deffnm": "NPT"}
    grompp_nvt_cmd = f"gmx grompp -f npt.mdp -c EM.gro -p topol.top -o {file_dict['deffnm']}.tpr -maxwarn 1"
    # gmx grompp -f nvt.mdp -c EM.gro -p topol.top -o NVT.tpr -maxwarn 1
    mdrun_nvt_cmd = f"gmx mdrun {args.mdrun_arguments} -deffnm {file_dict['deffnm']}"
    # gmx mdrun -nt 4 -deffnm NVT

    print("Running command:", grompp_nvt_cmd, flush=PRINT_FLUSH)
    run(grompp_nvt_cmd, shell=True, check=True, cwd=folder)

    print("Running command:", mdrun_nvt_cmd, flush=PRINT_FLUSH)
    run(mdrun_nvt_cmd, shell=True, check=True, cwd=folder)

    file_dict = {"deffnm": "DECANE_BERENDSEN"}
    grompp_nvt_cmd = f"gmx grompp -f grompp_berendsten.mdp -c NPT.gro -p topol.top -o {file_dict['deffnm']}.tpr -maxwarn {MAXWARN}"
    # gmx grompp -f nvt.mdp -c EM.gro -p topol.top -o NVT.tpr -maxwarn 1
    mdrun_nvt_cmd = f"gmx mdrun {args.mdrun_arguments} -deffnm {file_dict['deffnm']}"
    # gmx mdrun -nt 4 -deffnm NVT

    print("Running command:", grompp_nvt_cmd, flush=PRINT_FLUSH)
    run(grompp_nvt_cmd, shell=True, check=True, cwd=folder)

    print("Running command:", mdrun_nvt_cmd, flush=PRINT_FLUSH)
    run(mdrun_nvt_cmd, shell=True, check=True, cwd=folder)


# noinspection DuplicatedCode
def run_md(file_dict, p, folder, extend_sim=False, gen_vel=False, continuing=True):
    folder = os.path.join(Path(folder))

    make_md_file(os.path.join(folder, "grompp_template.mdp"), os.path.join(folder, file_dict['grompp']), p,
                 gen_vel=gen_vel, run_time=DELTA_T, continuing=continuing)
    if extend_sim:
        grompp_md_cmd = "gmx grompp -f {grompp} -c {prev_deffnm}.gro -t {prev_deffnm}.cpt -p topol.top -o {deffnm}.tpr -maxwarn {maxwarn}"\
            .format(grompp=file_dict['grompp'], deffnm=file_dict['deffnm'], prev_deffnm=file_dict['prev_deffnm'], maxwarn=MAXWARN)
        # gmx grompp -f grompp.mdp -c em.gro -p topol.top -o decane.tpr -maxwarn 2 &>> gromacs.log
    else:
        grompp_md_cmd = "gmx grompp -f {grompp} -c {prev_deffnm}.gro -p topol.top -o {deffnm}.tpr -maxwarn {maxwarn}" \
            .format(grompp=file_dict['grompp'], prev_deffnm=file_dict['prev_deffnm'], deffnm=file_dict['deffnm'], maxwarn=MAXWARN)

    if RESTRAINT_TYPE != 'None':
        if RESTRAINT_TYPE == 'Moving':
            grompp_md_cmd += ' -r {}.gro'.format(file_dict['prev_deffnm'])
        elif RESTRAINT_TYPE == 'Fixed':
            grompp_md_cmd += ' -r NPT.gro'

    mdrun_md_cmd = f"gmx mdrun {args.mdrun_arguments} -deffnm {file_dict['deffnm']}"
    # gmx mdrun -nt 24 -deffnm decane_gromos &>> gromacs.log

    print("Running command:", grompp_md_cmd, flush=PRINT_FLUSH)
    run(grompp_md_cmd, shell=True, check=True, cwd=folder)

    print("Running command:", mdrun_md_cmd, flush=PRINT_FLUSH)
    run(mdrun_md_cmd, shell=True, check=True, cwd=folder)


def pbc_box_volume(wdir):
    file_list = glob(os.path.join(wdir, "DECANE_-*.edr"))
    file_list.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True)
    file_list = file_list[:-1]
    print(file_list)
    Times, Volumes = read_density(file_list)
    Pressures = Times * SYSTEM_CONFIGURATION['P_RATE'] + SYSTEM_CONFIGURATION['P_START']

    # start_idx = np.absolute(Times - 0.3).argmin()
    # end_idx = np.where(Volumes < Volumes[start_idx]*1.5)[0][-1]

    # popt, _ = curve_fit(lin, Times[start_idx:end_idx], Volumes[start_idx:end_idx])
    # Volumes_linfit = lin(Times, *popt)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Times.reshape(-1, 1), Volumes.reshape(-1, 1))
    Volumes_linfit = ransac.predict(Times.reshape(-1, 1)).ravel()
    inlier_mask = ransac.inlier_mask_
    t_star = Times[inlier_mask][-1]

    # p_critical_estimate = -185
    # idx = np.where(np.abs(Volumes - Volumes_linfit) < V_bubble_star)
    # t_star = Times[np.absolute(Times - Volumes[idx][-1]).argmin()]
    # V_star = Volumes[np.absolute(Volumes - Volumes[idx][-1]).argmin()]

    decane_surf_tens = 230 * 2
    V_star = Volumes[inlier_mask][-1]
    # p_star = t_star * SYSTEM_CONFIGURATION['P_RATE'] + SYSTEM_CONFIGURATION['P_START']
    p_star = Pressures[inlier_mask][-1]
    r_bubble_star = -2 * decane_surf_tens / p_star
    V_bubble_star = 4 / 3 * np.pi * r_bubble_star ** 3
    print("Cavitation detected at:"
          "\ntime =", t_star,
          "\nvolume =", V_star,
          "\npressure =", p_star,
          "\nradius =", r_bubble_star,
          "\nbubble volume =", V_bubble_star,
          "\n", flush=PRINT_FLUSH)
    with open(os.path.join(wdir, "RESULTS.log"), "w") as f:
        print("Cavitation detected at:"
              "\ntime =", t_star,
              "\nvolume =", V_star,
              "\npressure =", p_star,
              "\nradius =", r_bubble_star,
              "\nbubble volume =", V_bubble_star,
              "\n", file=f, flush=True)

    result = {"t_star": t_star, "V_star": V_star, "p_star": p_star,
              "r_bubble_star": r_bubble_star, "V_bubble_star": V_bubble_star}
    with open(os.path.join(wdir, "RESULTS.pkl"), "wb") as pkl:
        pickle.dump(result, pkl)

    title = "PBC box volume and pressure"

    fig = plt.figure(dpi=150)
    fig.suptitle(title)

    plt.plot(Times, Volumes, color="c")
    plt.plot(Times, Volumes_linfit, color="red", alpha=0.8)
    plt.ylim(Volumes[0]*0.98, V_star*1.1)
    plt.axvline(x=t_star, color="black", linestyle="dotted")
    plt.hlines(V_star, xmin=0, xmax=t_star, color="c", linestyle="dotted")
    plt.legend(['V Raw Data', 'V Linear fit', 'Cavitation time'], bbox_to_anchor=(1.12, 1), loc="upper left", fontsize=8)
    plt.xlabel('Time ($ns$)')
    plt.ylabel(r'Volume ($nm^3$)')
    plt.title('Volume and Pressure')
    ax0_2 = plt.twinx()
    ax0_2.plot(Times, Pressures, color="orange")
    ax0_2.grid(False)
    ax0_2.hlines(p_star, xmin=0, xmax=t_star, color="orange", linestyle="dotted")
    ax0_2.legend(['Pressure'], bbox_to_anchor=(1.12, 0.82), loc="upper left", fontsize=8)
    ax0_2.set_ylabel(r'Pressure ($bar$)')

    plt.tight_layout()
    plt.savefig(os.path.join(wdir, "RESULTS_pbc_box_volume.png"), dpi='figure', format='png')
    plt.close(fig)


def trjcat(wdir, pattern):
    trr_paths = glob(os.path.join(wdir, pattern))
    trr_paths.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True)
    trr_files = [str(os.path.basename(i)) for i in trr_paths][-50:-1]

    # gmx trjcat -f DECANE_0.trr DECANE_-10.trr DECANE_-20.trr DECANE_-30.trr DECANE_-40.trr DECANE_-50.trr -o DECANE.trr
    cmd = ("gmx trjcat -cat -keeplast -f" + " {}"*len(trr_files) + " -o DECANE.xtc").format(*trr_files)
    # cmd = ("gmx trjcat -f" + " {}"*len(trr_files) + " -o DECANE.trr").format(*trr_files)
    run(cmd, shell=True, check=True, cwd=wdir)


# noinspection DuplicatedCode
def one_run():
    t1 = time()

    print("Preparing files...", flush=PRINT_FLUSH)

    if os.path.isdir(os.path.join(WORKDIR, f"p_rate_{P_RATE}")):
        previous_runs = glob(os.path.join(WORKDIR, f"p_rate_{P_RATE}", "run_*"))
        if len(previous_runs) == 0:
            workdir = os.path.join(WORKDIR, f"p_rate_{P_RATE}", "run_1")
            # os.makedirs(workdir)
        else:
            previous_runs.sort(key=lambda x: float(os.path.basename(x).split("_")[1]))
            m = int(os.path.basename(previous_runs[-1]).split("_")[-1])
            workdir = os.path.join(WORKDIR, f"p_rate_{P_RATE}", f"run_{m + 1}")
            # os.makedirs(workdir)
    else:
        workdir = os.path.join(WORKDIR, f"p_rate_{P_RATE}", "run_1")
        # os.makedirs(workdir)

    shutil.copytree(os.path.join(WORKDIR, "templates"), workdir)

    with open(os.path.join(workdir, SYS_CONF_PKL_NAME), "wb") as pkl:
        pickle.dump(SYSTEM_CONFIGURATION, pkl)

    shutil.copy2(CONFIG_FILE, os.path.join(workdir, os.path.basename(CONFIG_FILE)))

    print("WORKDIR:", workdir, flush=PRINT_FLUSH)

    pressures = np.arange(P_MAX, P_START + np.abs(DELTA_P), np.abs(DELTA_P))[::-1]

    with open(os.path.join(workdir, 'NPT.gro'), 'r') as f:
        conf_data = f.read().strip().split('\n')
        INITIAL_PBC_SIZE = [float(i) for i in conf_data[-1].strip().split()]

    # run_em(folder=workdir)
    # run_equilibration(folder=workdir)
    run_md({"grompp": f"GROMPP_{pressures[0]}.mdp", "deffnm": f"DECANE_{pressures[0]}", "prev_deffnm": "NPT"}, folder=workdir, p=pressures[0], gen_vel=False, continuing=True)

    for P in pressures[1:]:
        Files = {"grompp": "GROMPP_{}.mdp".format(P), "deffnm": "DECANE_{}".format(P), "prev_deffnm": "DECANE_{}".format(P - DELTA_P)}

        with open(os.path.join(workdir, "DECANE_{}.gro".format(P - DELTA_P)), 'r') as f:
            conf_data = f.read().strip().split('\n')
            CURRENT_PBC_SIZE = [float(i) for i in conf_data[-1].strip().split()]
        if MANUAL_STOP and CURRENT_PBC_SIZE[-1] > INITIAL_PBC_SIZE[-1] * 10:
            pbc_box_volume(workdir)
            trjcat(workdir, "DECANE_-*.xtc")
            break

        try:
            run_md(Files, folder=workdir, p=P, gen_vel=False, continuing=True)
        except CalledProcessError:
            # cavitation = pbc_box_volume(workdir)
            pbc_box_volume(workdir)
            # if not cavitation:
            #     raise err
            trjcat(workdir, "DECANE_-*.xtc")
            break

    t2 = time()
    print("Run time:", round(t2 - t1, 2), flush=PRINT_FLUSH)


# noinspection DuplicatedCode
def continue_run(wdir):
    t1 = time()

    previous_sims = glob(os.path.join(wdir, "DECANE_*.gro"))
    previous_sims.sort(key=lambda x: float(os.path.basename(x).split("_")[1].split(".")[0]), reverse=True)
    previous_max_p = int(os.path.basename(previous_sims[-1]).split("_")[1].split(".")[0])
    pressures = np.arange(P_MAX, P_START + np.abs(DELTA_P), np.abs(DELTA_P))[::-1]

    for P in pressures:
        print("Continuing from", previous_max_p, "bar", flush=PRINT_FLUSH)
        Files = {"grompp": "GROMPP_{}.mdp".format(P), "deffnm": "DECANE_{}".format(P), "prev_deffnm": "DECANE_{}".format(P - DELTA_P)}
        run_md(Files, folder=wdir, p=P)

    pbc_box_volume(wdir)
    trjcat(wdir, "DECANE_-*.xtc")

    t2 = time()
    print("Run time:", round(t2 - t1, 2), flush=PRINT_FLUSH)


def main():
    if args.log:
        original_stdout = sys.stdout
        original_stderr = sys.stderr

        log_file = f"VARIABLE_PRESSURE_SIM_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.log"
        LOG = open(os.path.join(WORKDIR, log_file), "w")
        sys.stdout = LOG
        sys.stderr = LOG

    start_time = time()

    if CONTINUE_RUN:
        continue_run(WORKDIR)

    elif ANALYSIS:
        with open(os.path.join(WORKDIR, SYS_CONF_PKL_NAME), "rb") as pkl:
            global SYSTEM_CONFIGURATION
            SYSTEM_CONFIGURATION = pickle.load(pkl)

        pbc_box_volume(WORKDIR)
        # trjcat(WORKDIR, "DECANE_-*.xtc")

    else:
        one_run()

    end_time = time()
    print("Total run time:", round(end_time-start_time, 2), flush=PRINT_FLUSH)

    if args.log:
        # noinspection PyUnboundLocalVariable
        sys.stdout = original_stdout
        # noinspection PyUnboundLocalVariable
        sys.stderr = original_stderr
        # noinspection PyUnboundLocalVariable
        LOG.close()


if __name__ == '__main__':
    sleep(np.random.randint(31))
    main()
