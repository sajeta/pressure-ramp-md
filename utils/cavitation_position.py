import os
os.environ['GMX_MAXBACKUP'] = '-1'

import numpy as np
import mdtraj as md
import panedr
from sklearn import linear_model


RHO_H2O = 132.80154
RHO_HC = 98.263216
RHO_H2O_MASS = 997.62655
RHO_HC_MASS = 725.52476


def density_from_vectors(vectors, bins):
    distances = np.linalg.norm(vectors, axis=1)
    distance_max = np.max(distances)
    # distance_min = np.min(distances)
    distance_min = 0

    Vs = np.linspace(distance_min**2, distance_max**2, num=bins)
    Rs = np.power(Vs, 1/2)

    # Take a histogram:
    counts, lengths = np.histogram(distances, bins=Rs)
    # calculate the volume of each spherical shell:
    shell_volumes = 4/3*np.pi * (lengths[1:]**3 - lengths[:-1]**3)
    # normalize 'counts' by the volume of each shell:
    density = counts / shell_volumes
    radius = (lengths[:-1]+lengths[1:])/2

    return radius, density


def mass_density_from_vectors(vectors, masses, bins):
    distances = np.linalg.norm(vectors, axis=1)
    distance_max = np.max(distances)
    # distance_min = np.min(distances)
    distance_min = 0

    Vs = np.linspace(distance_min**2, distance_max**2, num=bins)
    Rs = np.power(Vs, 1/2)

    # Take a histogram:
    counts, lengths = np.histogram(distances, bins=Rs)
    # calculate the volume of each spherical shell:
    shell_volumes = 4/3*np.pi * (lengths[1:]**3 - lengths[:-1]**3)
    # normalize 'counts' by the volume of each shell:
    # density = counts / shell_volumes
    radius = (lengths[:-1]+lengths[1:])/2

    bins = lengths[1:-1]
    ndx = np.digitize(distances, bins=bins)
    assert all(counts == np.bincount(ndx))

    mass_distribution = [[] for _ in range(len(np.bincount(ndx)))]
    for n, mass in zip(ndx, masses):
        mass_distribution[n].append(mass)

    radii = []
    mass_density = []
    for mass_dist, r in zip(mass_distribution, radius):
        radii.append(r)
        mass_density.append(np.sum(mass_dist))
    radii = np.array(radii)
    mass_density = np.array(mass_density) / shell_volumes * 1.66

    # r_ = radius[np.where(density < 1e-1)[0][0]]
    # idx = np.abs(radii - r_).argmin()

    # return radii[:idx], mass_density[:idx]
    return radii, mass_density


def calculate_radial_densities(file_xtc, file_gro):
    traj = md.load(file_xtc, top=file_gro)

    decane = traj.top.select('resname DEC')
    system = traj.top.select('all')
    water = traj.top.select('water')

    # calculate masses of all atoms:
    masses = np.array([i.element.mass for i in traj.topology.atoms])

    # iterate through all frames and save the distances of each atom to the centre of mass:
    radii = []
    densities = []
    radii_dec = []
    densities_dec = []
    radii_water = []
    densities_water = []
    R_max = []

    for frame in traj.xyz:
        decane_coords = frame[decane]
        all_coords = frame[system]
        water_coord = frame[water]
        # centre_of_mass = decane_coords.mean(0)
        centre_of_mass = np.average(decane_coords, axis=0, weights=masses[decane])

        # idx = np.where(np.linalg.norm(all_coords - centre_of_mass, axis=1) < np.max(all_coords/2))
        # vectors = (all_coords[idx] - centre_of_mass)
        vectors = (all_coords - centre_of_mass)
        radius, density = density_from_vectors(vectors, bins=200)
        radii.append(radius)
        densities.append(density)

        # idx = np.where(np.linalg.norm(decane_coords - centre_of_mass, axis=1) < np.max(all_coords/2))
        # vectors = (decane_coords[idx] - centre_of_mass)
        vectors = (decane_coords - centre_of_mass)
        radius, density = density_from_vectors(vectors, bins=100)
        radii_dec.append(radius)
        densities_dec.append(density)

        # idx = np.where(np.linalg.norm(water_coord - centre_of_mass, axis=1) < np.max(all_coords/2))
        # vectors = (water_coord[idx] - centre_of_mass)
        vectors = (water_coord - centre_of_mass)
        radius, density = density_from_vectors(vectors, bins=100)
        radii_water.append(radius)
        densities_water.append(density)

        idx = np.where(np.linalg.norm(all_coords - centre_of_mass, axis=1) <= np.max(all_coords)/2)
        r_max = np.max(all_coords[idx] - centre_of_mass, axis=0)
        r_min = np.abs(np.min(all_coords[idx] - centre_of_mass, axis=0))
        R = np.concatenate((r_max, r_min))
        R_max.append(np.min(R))

    return_data = {
        'radii': radii,
        'densities': densities,
        'radii_dec': radii_dec,
        'densities_dec': densities_dec,
        'radii_water': radii_water,
        'densities_water': densities_water,
        'R_max': R_max,
    }
    return return_data


def calculate_radial_densities_full_traj(file_xtc, file_gro, bin_num=1000, last_frame=-1):
    traj = md.load(file_xtc, top=file_gro)

    decane = traj.top.select('resname DEC')
    system = traj.top.select('all')
    water = traj.top.select('water')

    # calculate masses of all atoms:
    masses = np.array([i.element.mass for i in traj.topology.atoms])

    vectors_all = []
    vectors_decane = []
    vectors_water = []
    R_max = []

    N = len(traj.xyz[:last_frame])
    print('Number of frames:', N)
    for frame in traj.xyz[:last_frame]:
        decane_coords = frame[decane]
        all_coords = frame[system]
        water_coord = frame[water]
        centre_of_mass = np.average(decane_coords, axis=0, weights=masses[decane])
        vectors = (all_coords - centre_of_mass)
        vectors_all.append(vectors)
        del vectors

        vectors_dec = (decane_coords - centre_of_mass)
        vectors_decane.append(vectors_dec)
        del vectors_dec

        vectors_w = (water_coord - centre_of_mass)
        vectors_water.append(vectors_w)
        del vectors_w

        idx = np.where(np.linalg.norm(all_coords - centre_of_mass, axis=1) <= np.max(all_coords)/2)
        r_max = np.max(all_coords[idx] - centre_of_mass, axis=0)
        r_min = np.abs(np.min(all_coords[idx] - centre_of_mass, axis=0))
        R = np.concatenate((r_max, r_min))
        R_max.append(np.min(R))
    del traj; del masses

    radius_all, density_all = density_from_vectors(np.concatenate(vectors_all), bins=bin_num)
    radius_decane, density_decane = density_from_vectors(np.concatenate(vectors_decane), bins=bin_num)
    radius_water, density_water = density_from_vectors(np.concatenate(vectors_water), bins=bin_num)
    r_max = np.min(R_max)
    del vectors_all; del vectors_decane; del vectors_water; del R_max

    return_data = {
        'radius_all': radius_all,
        'density_all': density_all / N,
        'radius_decane': radius_decane,
        'density_decane': density_decane / N,
        'radius_water': radius_water,
        'density_water': density_water / N,
        'r_max': r_max,
    }
    del radius_all; del density_all
    del radius_decane; del density_decane
    del radius_water; del density_water

    return return_data


def calculate_radial_mass_densities_full_traj(file_xtc, file_gro, bin_num=1000, last_frame=-1):
    traj = md.load(file_xtc, top=file_gro)

    decane = traj.top.select('resname DEC')
    system = traj.top.select('all')
    water = traj.top.select('water')

    # calculate masses of all atoms:
    masses = np.array([i.element.mass for i in traj.topology.atoms])

    vectors_all = []
    vectors_decane = []
    vectors_water = []
    R_max = []

    masses_all = []
    masses_decane = []
    masses_water = []

    N = len(traj.xyz[:last_frame])
    print('Number of frames:', N)
    for frame in traj.xyz[:last_frame]:
        decane_coords = frame[decane]
        all_coords = frame[system]
        water_coord = frame[water]
        centre_of_mass = np.average(decane_coords, axis=0, weights=masses[decane])
        vectors = (all_coords - centre_of_mass)
        vectors_all.append(vectors)
        del vectors
        masses_all.append(masses)

        vectors_dec = (decane_coords - centre_of_mass)
        vectors_decane.append(vectors_dec)
        del vectors_dec
        masses_decane.append(masses[decane])

        vectors_w = (water_coord - centre_of_mass)
        vectors_water.append(vectors_w)
        del vectors_w
        masses_water.append(masses[water])

        idx = np.where(np.linalg.norm(all_coords - centre_of_mass, axis=1) <= np.max(all_coords)/2)
        r_max = np.max(all_coords[idx] - centre_of_mass, axis=0)
        r_min = np.abs(np.min(all_coords[idx] - centre_of_mass, axis=0))
        R = np.concatenate((r_max, r_min))
        R_max.append(np.min(R))
    del traj; del masses

    radius_all, density_all = mass_density_from_vectors(
        np.concatenate(vectors_all),
        np.concatenate(masses_all),
        bins=bin_num
    )
    radius_decane, density_decane = mass_density_from_vectors(
        np.concatenate(vectors_decane),
        np.concatenate(masses_decane),
        bins=bin_num
    )
    radius_water, density_water = mass_density_from_vectors(
        np.concatenate(vectors_water),
        np.concatenate(masses_water),
        bins=bin_num
    )
    r_max = np.min(R_max)
    del vectors_all; del vectors_decane; del vectors_water; del R_max

    return_data = {
        'radius_all': radius_all,
        'density_all': density_all / N,
        'radius_decane': radius_decane,
        'density_decane': density_decane / N,
        'radius_water': radius_water,
        'density_water': density_water / N,
        'r_max': r_max,
    }
    del radius_all; del density_all
    del radius_decane; del density_decane
    del radius_water; del density_water

    return return_data


def find_cavitation_frame(radii_dec, densities_dec):
    ndx = np.where(densities_dec[0] > RHO_HC*0.9)[0]
    indx = ndx[:int(len(ndx)*0.5)][-1]

    frame_i = None; R = None
    for i, density in enumerate(densities_dec):
        idx = np.where(density[:indx] < RHO_HC*0.2)[0]
        # idx = np.where(density[:len(density)//4] < RHO_HC*0.1)[0]
        if len(idx) > 0:
            frame_i = i
            R = radii_dec[frame_i][idx[0]]
            break
    return_data = {
        'frame_i': frame_i,
        'R': R
    }
    return return_data


def load_system_volume_density(file_edr):
    data = panedr.edr_to_df(file_edr)
    Times = data['Time'].values[len(data['Time'].values)//20:] / 1e3
    Volumes = data['Volume'].values[len(data['Volume'].values)//20:]
    print(Times.shape, Volumes.shape)

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Times.reshape(-1, 1), Volumes.reshape(-1, 1))
    Volumes_linfit = ransac.predict(Times.reshape(-1, 1)).ravel()
    inlier_mask = ransac.inlier_mask_
    t_star = Times[inlier_mask][-1]
    V_star = Volumes[inlier_mask][-1]

    Density = data['Density'].values[len(data['Volume'].values)//20:]

    ransac = linear_model.RANSACRegressor()
    ransac.fit(Times.reshape(-1, 1), Density.reshape(-1, 1))
    Density_linfit = ransac.predict(Times.reshape(-1, 1)).ravel()
    inlier_mask = ransac.inlier_mask_
    # t_star = Times[inlier_mask][-1]
    rho_star = Density[inlier_mask][-1]

    return_data = {
        'Time': Times,
        'Volume': Volumes,
        'V_linfit': Volumes_linfit,
        'V_star': V_star,
        'rho': Density,
        'rho_linfit': Density_linfit,
        'rho_star': rho_star,
        't_star': t_star
    }
    return return_data


def find_empty_voxels(file_xtc, file_gro, first_frame=None, last_frame=-1, stride=1):
    print(f'Loading file {file_xtc}', flush=True)
    traj = md.load(file_xtc, top=file_gro, stride=stride)

    decane = traj.top.select('resname DEC')
    system = traj.top.select('all')
    # water = traj.top.select('water')

    # calculate masses of all atoms:
    masses = np.array([i.element.mass for i in traj.topology.atoms])

    COMs = []
    cav_positions = []
    cav_Rs = []

    if first_frame is None:
        first_frame = int(len(traj.xyz)*0.9)

    frames = traj.xyz[first_frame:last_frame]
    del traj

    print(f'Number of frames to read: {len(frames)}', flush=True)
    for frame in frames:
        decane_coords = frame[decane]
        all_coords = frame[system]
        # water_coord = frame[water]

        centre_of_mass = np.average(decane_coords, axis=0, weights=masses[decane])
        COMs.append(centre_of_mass)

        N = int(np.max(all_coords)/0.8)
        H, edges = np.histogramdd(all_coords, bins=N)
        idx = np.where(H == 0)

        xs = edges[0][idx[0]]
        ys = edges[1][idx[1]]
        zs = edges[2][idx[2]]
        cav_pos = np.array([[i, j, k] for i, j, k in zip(xs, ys, zs)])
        cav_positions.append(cav_pos)

        if len(cav_pos) > 0:
            cav_R = np.linalg.norm(cav_pos - centre_of_mass, axis=1)
        else:
            cav_R = []
        cav_Rs.append(cav_R)

    # cav_frame_i = None
    # for i, cav_pos in enumerate(cav_positions):
    #     if len(cav_pos) > 0:
    #         if all([len(q) for q in cav_positions[i:i+5]]):
    #             cav_frame_i = i
    #             break

    cav_frame_i = None
    for i, cav_pos in enumerate(cav_positions):
        if len(cav_pos) > 5:
            cav_frame_i = i
            break

    return_data = {
        'cav_frame_i': cav_frame_i,
        'COMs': COMs,
        'cav_positions': cav_positions,
        'cav_Rs': cav_Rs,
    }
    return return_data
