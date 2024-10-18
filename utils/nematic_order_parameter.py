import numpy as np
import mdtraj as md
from more_itertools import locate
from operator import itemgetter


def calculate_decane_nematic_order(file_xtc, file_gro, bin_num=25, return_frames='all'):
    traj = md.load(file_xtc, top=file_gro)

    RADII = []
    ANGLES = []
    if return_frames == 'all':
        frames = traj.xyz
    else:
        frames = itemgetter(*return_frames)(traj.xyz)

    for frame in frames:
        decane_idx = list(locate(traj.topology.atoms, lambda x: 'DEC' in str(x.residue)))
        decane_atoms = itemgetter(*decane_idx)(list(traj.topology.atoms))
        decane_atoms_xyz = frame[decane_idx]

        residues = np.unique([str(x.residue) for x in decane_atoms]).tolist()
        decane_molecules = dict([(i, []) for i in residues])
        decane_molecules_xyz = dict([(i, []) for i in residues])
        for atom, atom_xyz in zip(decane_atoms, decane_atoms_xyz):
            decane_molecules[str(atom.residue)].append(atom)
            decane_molecules_xyz[str(atom.residue)].append(atom_xyz)

        decane_molecules_com = np.array(
            [
                np.average(
                    np.array(decane_molecules_xyz[residue]),
                    axis=0,
                    weights=np.array([i.element.mass for i in decane_molecules[residue]]))
                for residue in residues
            ])

        centre_of_mass = np.average(decane_atoms_xyz, axis=0, weights=np.array([i.element.mass for i in decane_atoms]))

        C1_idx = list(locate(decane_molecules[residues[0]], lambda x: str(x) == 'DEC1-C1'))[0]
        C9_idx = list(locate(decane_molecules[residues[0]], lambda x: str(x) == 'DEC1-C9'))[0]

        cos_thetas = []
        for i, _ in enumerate(decane_molecules_com):
            r1 = decane_molecules_com[i] - centre_of_mass
            r2 = decane_molecules_xyz[residues[i]][C9_idx] - decane_molecules_xyz[residues[i]][C1_idx]
            cos_theta = np.dot(r1, r2) / (np.linalg.norm(r1)*np.linalg.norm(r2))
            cos_thetas.append(cos_theta)
        cos_thetas = np.array(cos_thetas)

        distances = np.linalg.norm(decane_molecules_com - centre_of_mass, axis=1)
        distance_max = np.max(distances)
        distance_min = 0
        Vs = np.linspace(distance_min**2, distance_max**2, num=bin_num)
        bins = np.power(Vs, 1/2)

        counts, edges = np.histogram(distances, bins=bins)
        radius = (edges[:-1]+edges[1:])/2
        # shell_volumes = 4/3*np.pi * (edges[1:]**3 - edges[:-1]**3)
        # density = counts / shell_volumes

        bins = edges[1:-1]
        ndx = np.digitize(distances, bins=bins)
        assert all(counts == np.bincount(ndx))

        cos_thetas_dist = [[] for _ in range(len(np.bincount(ndx)))]
        for n, cos_theta in zip(ndx, cos_thetas):
            cos_thetas_dist[n].append(cos_theta)

        radii = []
        order_params = []
        for cos_theta_dist, r in zip(cos_thetas_dist, radius):
            if len(cos_theta_dist) > 2:
                radii.append(r)
                cos = np.array(cos_theta_dist)
                S = np.mean((3*cos**2 - 1) / 2)
                order_params.append(S)
        radii = np.array(radii)
        angles = np.array(order_params)
        RADII.append(radii)
        ANGLES.append(angles)

    return RADII, ANGLES


def calculate_distances_angles_full_traj(
        file_xtc,
        file_gro,
        last_frame=-1,
):
    traj = md.load(file_xtc, top=file_gro)

    DISTANCES = []
    COS_THETAS = []

    N = len(traj.xyz[:last_frame])
    print('Number of frames:', N)
    for frame in traj.xyz[:last_frame]:
        decane_idx = list(locate(traj.topology.atoms, lambda x: 'DEC' in str(x.residue)))
        decane_atoms = itemgetter(*decane_idx)(list(traj.topology.atoms))
        decane_atoms_xyz = frame[decane_idx]

        residues = np.unique([str(x.residue) for x in decane_atoms]).tolist()
        decane_molecules = dict([(i, []) for i in residues])
        decane_molecules_xyz = dict([(i, []) for i in residues])
        for atom, atom_xyz in zip(decane_atoms, decane_atoms_xyz):
            decane_molecules[str(atom.residue)].append(atom)
            decane_molecules_xyz[str(atom.residue)].append(atom_xyz)

        decane_molecules_com = np.array(
            [
                np.average(
                    np.array(decane_molecules_xyz[residue]),
                    axis=0,
                    weights=np.array([i.element.mass for i in decane_molecules[residue]]))
                for residue in residues
            ])

        centre_of_mass = np.average(decane_atoms_xyz, axis=0, weights=np.array([i.element.mass for i in decane_atoms]))

        C1_idx = list(locate(decane_molecules[residues[0]], lambda x: str(x) == 'DEC1-C1'))[0]
        C9_idx = list(locate(decane_molecules[residues[0]], lambda x: str(x) == 'DEC1-C9'))[0]

        cos_thetas = []
        for i, _ in enumerate(decane_molecules_com):
            r1 = decane_molecules_com[i] - centre_of_mass
            r2 = decane_molecules_xyz[residues[i]][C9_idx] - decane_molecules_xyz[residues[i]][C1_idx]
            cos_theta = np.dot(r1, r2) / (np.linalg.norm(r1)*np.linalg.norm(r2))
            cos_thetas.append(cos_theta)
        cos_thetas = np.array(cos_thetas)
        COS_THETAS.append(cos_thetas)

        distances = np.linalg.norm(decane_molecules_com - centre_of_mass, axis=1)
        DISTANCES.append(distances)

    COS_THETAS = np.concatenate(COS_THETAS)
    DISTANCES = np.concatenate(DISTANCES)
    return DISTANCES, COS_THETAS


def bin_distances_angles_full_traj(
        DISTANCES,
        COS_THETAS,
        bin_num=1000,
        order_param_func=lambda x: (3*x**2 - 1) / 2
):
    distance_max = np.max(DISTANCES)
    distance_min = 0
    Vs = np.linspace(distance_min**2, distance_max**2, num=bin_num)
    bins = np.power(Vs, 1/2)

    counts, edges = np.histogram(DISTANCES, bins=bins)
    radius = (edges[:-1]+edges[1:])/2
    shell_volumes = 4/3*np.pi * (edges[1:]**3 - edges[:-1]**3)
    density = counts / shell_volumes

    bins = edges[1:-1]
    ndx = np.digitize(DISTANCES, bins=bins)
    assert all(counts == np.bincount(ndx))

    cos_thetas_dist = [[] for _ in range(len(np.bincount(ndx)))]
    for n, cos_theta in zip(ndx, COS_THETAS):
        cos_thetas_dist[n].append(cos_theta)

    radii = []
    order_params = []
    for cos_theta_dist, r in zip(cos_thetas_dist, radius):
        if len(cos_theta_dist) > 2:
            radii.append(r)
            cos = np.array(cos_theta_dist)
            S = np.mean(order_param_func(cos))
            order_params.append(S)
    radii = np.array(radii)
    angles = np.array(order_params)

    r_ = radius[np.where(density < 1e-1)[0][0]]
    idx = np.abs(radii - r_).argmin()

    return radii[:idx], angles[:idx]


def calculate_decane_nematic_order_full_traj(
        file_xtc,
        file_gro,
        bin_num=1000,
        last_frame=-1,
        order_param_func=lambda x: (3*x**2 - 1) / 2
):
    distances, cos_thetas = calculate_distances_angles_full_traj(file_xtc, file_gro, last_frame)
    radii, angles = bin_distances_angles_full_traj(distances, cos_thetas, bin_num, order_param_func)
    return radii, angles
