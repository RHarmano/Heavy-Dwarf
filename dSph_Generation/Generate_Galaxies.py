#!/usr/bin/python
"""
Generate_Galaxies.py

By: Roman Harman
Date: Oct. 20th, 2020

A sample script calling the dSph_Model class to produce a galaxy with parallelization
"""
import numpy as np
import pandas as pd
import scipy.stats.distributions as st
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import norm
from dSph_Model import dSph_Model
import matplotlib.pyplot as plt

from mpi4py import MPI

DM_distribution_model = 'Combination'
exit_toggle = False

# establish how many cores we are working on
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# main body of the simulation software which establishes the initial parameters and generates the csv files
if __name__ == '__main__':

    diskfig = plt.figure()
    ax = diskfig.gca(projection='3d')

    if size == 1:
        print("It looks like you're running this on one core. It will take significantly longer for a large number of galaxies (~ 10 hours for 1000 galaxies)")
    # Parameter Definition
    # number of galaxies to produce
    num_galaxy = 1

    # Burkert profile parameters
    r_core_array = np.linspace(50, 2500, 10000)
    rho_central_array = np.linspace(10 ** -1.5, 10 ** 1.5, 10000)

    #NFW profile parameters
    M_200_array = np.linspace(7.2e7, 5.2e15, 1000000)
    c_param_array = np.linspace(10, 50, 40)

    # general parameters
    beta_array = np.linspace(-3.36,0.33,10000)
    D_array = np.logspace(4.07, 5.2, 100000)
    i = 0

    # Anisotropy parameters
    beta = 0  # np.random.choice(beta_array)  # Orbital anisotropy

    # stellar density parameters
    N = 1000  # Number of stars
    L = 2.7e5  # Solar Luminosities

    r_eval = np.logspace(1, 3, 100)  # radius array
    D_array = np.logspace(4.07, 5.2, 10000)  # array from aprox 20-120 kpc
    z_array = np.linspace(0.1, 1e3, 100) # array of z values for cylindrical co-ordinates in the galactic plane

    if size > 1:
        num_galaxy_processor = num_galaxy//size
        for size_index in range(size):
            if rank == size_index:
                dSph_project =  pd.DataFrame(columns=["x", "y", "z",
                                    "x_velocity", "y_velocity", 
                                    "z_velocity", "theta_x", 
                                    "theta_y"])
                for galaxy in range(num_galaxy_processor):
                    # Burkert profile parameters
                    r_core = float(np.random.choice(r_core_array, 1))  # [pc]
                    rho_central = float(np.random.choice(rho_central_array, 1))  # [m_solar/pc^3]
                    r_half = r_core * 0.4  # Radius containing half the luminosity

                    # NFW profile parameters
                    M_200 = float(np.random.choice(M_200_array, size=1))
                    c_param = float(np.random.choice(c_param_array, size=1))

                    # Call the class to build a dSph model from given parameters
                    dSph_galaxy_model = dSph_Model(r_eval, r_core, r_half, rho_central, L, beta, M_200, c_param)

                    # Produce the galaxy distribution and add it to the dataframe
                    # dSph_project.append(dSph_galaxy_model.produce_dSph(D_array, 'Burkert'), ignore_index=True)
                    
                    # Generate the mass of the galaxy
                    mass = []
                    for r in r_eval:
                        mass.append(dSph_galaxy_model.mass_from_density(r))
                    DM_mass = sum(mass)
                    
                    print("{} galaxies generated.".format(str(galaxy)))
                comm.send(dSph_project, dest=0, tag=8228)
            if rank == 0:
                dSph_project.append(comm.recv(source = size_index, tag = 8228))
    else:
        dSph_project =  pd.DataFrame(columns=["x", "y", "z",
                    "x_velocity", "y_velocity", 
                    "z_velocity", "theta_x", 
                    "theta_y"])
        for galaxy in range(num_galaxy):
            # Burkert profile parameters
            r_core = float(np.random.choice(r_core_array, 1))  # [pc]
            rho_central = float(np.random.choice(rho_central_array, 1))  # [m_solar/pc^3]
            r_half = r_core * 0.4  # Radius containing half the luminosity

            # NFW profile parameters
            M_200 = float(np.random.choice(M_200_array, size=1))
            c_param = float(np.random.choice(c_param_array, size=1))

            # Call the class to build a dSph model from given parameters
            dSph_galaxy_model = dSph_Model(r_eval, r_core, r_half, rho_central, L, beta, M_200, c_param)

            # Produce the galaxy distribution and add it to the dataframe
            # dSph_project.append(dSph_galaxy_model.produce_dSph(D_array, 'Burkert'), ignore_index=True)
            # disk_array = dSph_galaxy_model.gen_disk_foreground(r_eval, z_array)
            R, Z = np.meshgrid(r_eval, z_array)
            # surf = ax.plot_surface(R, Z, disk_array)
            # diskfig.colorbar(surf)
            # plt.show()
            # bulge_array = dSph_galaxy_model.gen_bulge_foreground(r_eval)
            halo_array = dSph_galaxy_model.gen_halo_foreground(R, Z)
            surf = ax.plot_surface(R, Z, halo_array)
            diskfig.colorbar(surf)
            plt.show()




            # Generate the mass of the galaxy
            # mass = []
            # for r in r_eval:
            #     mass.append(dSph_galaxy_model.mass_from_density(r))
            # DM_mass = sum(mass)
            