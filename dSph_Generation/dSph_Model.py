# -*- coding: utf-8 -*-

#################################################
# 
# Dwarf Spheroidal Satellite Galaxy Simulation Code
# 
# Author: Roman Harman
# Dated: September 29, 2020
#
# Originally contained in Genesis_Burkert.py,
# Author: Daniel Friedland
# Dated: March 7, 2019 and March 22 2019
#
# Edited and Compiled by: Robert Faraday
# Dated: March 8, 2020
#################################################

import numpy as np
import pandas as pd
import scipy.stats as stat
import scipy.stats.distributions as st
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.special import kv
import os
import datetime

class disk_gen(st.rv_continuous):
    """Volumetric mass density of stars within the Disk"""
    def _pdf(self, r, z):
        H = 2.75e3 # length scale of the disk [pc]
        solar_mass = 2e30 # mass of the sun, common mass scale in astrophysics [kg]
        rho_0_disk = 0.0493*solar_mass # central disk density [kg/pc^2]
        beta_disk = 0.565 # orbital anisotropy of stars in the disk []
        h1 = 270 # [pc]
        h2 = 440 # [pc]
        R_0 = 8e3 # ~radius where bulge -> disk [pc]

        nu_disk = np.array([])
        try:
            for i in range(len(r)):
                nu_disk.append(np.max([r[i]/9025+0.114, 0.670]))
        except Exception:
            nu_disk = [np.max([r/9025+0.114, 0.670])]
        R, Z = np.meshgrid(r, z) # need a meshgrid for cylindrical distribution
        nu_disk = np.array(nu_disk)
        hypsec = stat.hypsecant.pdf(Z/nu_disk*h1)
        disk_density = ((rho_0_disk/nu_disk)*np.exp(-(R-R_0)/H)*
                        ((1-beta_disk)*hypsec**2+beta_disk*np.exp(-abs(Z)/(nu_disk*h2)))) # [solar masses/pc^3]
        return disk_density

class bulge_gen(st.rv_continuous):
    """Volumetric mass density of stars within the Bulge"""
    def _pdf(self, s):
        piecewise_shift = 938 # [pc]

        s_inner = s[s < piecewise_shift]
        s_outer = s[s >= piecewise_shift]

        inner_bulge_dist = 1.04e6*(s_inner/0.482)**(-1.85) # [solar masses/pc^3]
        outer_bulge_dist = 3.53*kv(0, (s_outer/667)) # [solar masses/pc^3]
        bulge_density = np.append(inner_bulge_dist, outer_bulge_dist)
        return bulge_density

class halo_gen(st.rv_continuous):
    """
    Number density of stars within the "halo" of the galaxy
    simplified as an ellipsoidal model.

    Values exist more as ranges for best fit parameters, so typical ranges will
    be provided next to them.

    https://arxiv.org/pdf/astro-ph/0510520.pdf
    """
    def _pdf(self, r, z):
        q_H = 0.6 # ~0.5-1 for common values, most recent as of above source for MW was 0.6; halo ellipticity
        R_solar = 2.25461e-8 # astronomical constant
        n_H = 2.6 # ~2.5-3.0 fit parameter
        f_H = 1e-3 # halo normalization relative to the thin disk, i.e. rho_D(r=R_solar, z=0)
        disk_object = disk_gen()
        rho_D = disk_object._pdf(r=R_solar, z=0)
        return rho_D*f_H*(R_solar/(r**2+(z/q_H)**2)**(1/2))**(n_H)


class dSph_Model():
    """
    A class to encapsulate the generation of a dSph galaxy using both
    Burkert and NFW DM distribution methods, allowing for the
    parallelization of the simulation to achieve better generation
    times for larger galaxies.

    Parameters
    ----------
    r_eval : array-like variable\n
        Radius from centre of galaxy (units).\n
    r_core : float\n
        Core radius where half of luminosity is contained (units).\n
    r_half : float\n
        Radius where half of luminosity is contained (units).\n
    rho_central : float\n
        Central density.\n
    L : float\n
        Total Luminosity (units).\n
    beta : float or array\n
        The orbital anisotropy parameter; use 0 for simplification.\n
    M_200 : float\n
        Mass of the dark matter halo used to calculate the 
        radius of the dark matter halo (virial mass).\n
    c_param : float\n
        Concentration parameter which determines the characteristic
        radius and the characteristic overdensity.
    """

    # constructor
    def __init__(self, r_eval, r_core, r_half, rho_central, 
                    L, beta, M_200, c_param):
        self.r_eval = r_eval
        self.L = L
        self.r_half = r_half
        self.r_core = r_core
        self.rho_central = rho_central
        self.beta = beta
        self.M_200 = M_200
        self.c_param = c_param    

    # dSph functions
    def plummer(self, r, L, r_half):
        """ 
        Returns the Stellar number density given a Plummer Profile with
        some total luminosity and half radius.

        Parameters
        -----------
        r : array-like variable\n
            Radius from centre of galaxy (units)\n
        L : float\n
            Total Luminosity (units)\n
        r_half : float\n
            Radius where half of luminosity is contained (units)
        """
        
        return ( 
                3 * L *(4 * np.pi * r_half ** 3) ** (-1) *
                (1 + r ** 2 / r_half **2 ) ** (-5/2)
                )

    def burkert(self, r):
        """
        Returns a Dark Matter density distribution based on a Burkert DM
        density profile.

        Parameters
        -----------
        r : array-like variable\n
            Radius from centre of galaxy (units)\n
        r_core : float\n
            Core Radius Radius where half of luminosity is contained (units)\n
        rho_central : float\n
            central density

        Returns
        ----------
        (sigma_p)^2 : array-like variable
        """

        return (
                    (self.rho_central * (self.r_core ** 3)) /
                    ((r + self.r_core) * (r ** 2 + self.r_core ** 2))
                )

    def NFW(self, r):
        """ 
        Returns a Dark Matter density distribution based on a NFW DM
        density profile

        Parameters
        -----------
        r : array-like variable\n
            Radius from centre of galaxy (units)\n
        """
        # constants 
        RHO_C = 2.77536627e-7*0.674**2 # critical density of the universe [m_sun/pc]

        #calculations
        delta_c = ((200/3)*(self.c_param**3)/
        (np.log(1+self.c_param)-self.c_param/(1+self.c_param))) # characteristic overdensity [arb. units]
        
        r_200 = (3*self.M_200/(200*RHO_C*4*np.pi))**(1/3) # virial radius [pc]

        r_char = r_200/self.c_param # characeristic radius [pc]

        return ((RHO_C*delta_c)/((r/r_char)*(1+r/r_char)**2))

    def mass_from_density(self, r):
        """
        Calculates the mass of DM from the underlying DM distribution function.

        Parameters
        ----------
        r : array-like\n
            Radius from the centre of the galaxy (units).
        """
        mass_integral =  quad(func=lambda s: (s ** 2) * self.rho_dist(r), a=0, b=r)
        mass = 4 * np.pi * mass_integral[0]
        return mass

    def jeans_ivp(self,r,U):
        """ 
        Solves an initial value problem Jeans Equation. See salucci et al (2014)
        
        Parameters
        -----------
        r : array-like variable\n
            Radius from centre of galaxy (units)\n
        U : array like\n
            Radial Velocity Dispersion * nu. 
            What is being solved for by the ode Solver.\n
        """
        
        nu = self.plummer(r, self.L, self.r_half)
        M = self.mass_from_density(r)
        G = 4.302e-3  # pc*m_sun-1*(km/s)^2
        # jeans equation rearranged for U
        dUdr = (- ((G * M * nu / r**2) - (2 * self.beta * U / r)))
        return dUdr

    # define the function to solve the initial value problem for the jeans equation using the RK23 method
    def solve_jeans(self):
        """ 
        Solves an initial value problem for Jeans Equation. 
        See salucci et al (2014) for background.
        """
        sol = solve_ivp(lambda r, y: self.jeans_ivp(r,y),
                        [self.r_eval[-1],self.r_eval[0]],
                        y0 = [0],
                        method = 'RK23',
                        t_eval=self.r_eval[::-1])
        
        return sol

    def angular_dispersion_from_radial(self, var_r):
        """
        Calculates the angular dispersion of the galaxy as it relates
        to the radial dispersion.

        Parameters
        ----------
        beta : float\n
            The orbital anisotropy parameter\n
        var_r : float\n
            The radial dispersion of the galaxy
        """
        var_theta = (1-self.beta)*var_r
        
        return var_theta    

    def dispersions_from_sol(self):
        """
        Returns the dispersion of the galaxy as a result of group motion of stars
        within the galaxy governed by Jeans' equation(s).

        Parameters
        ----------
        sol : type(scipy.integrate.solve_ivp(...))\n
            Solution to the IVP for Jeans equation for the galaxy\n
        nu : array-like\n
            Number density distribution according to plummer model
        """
        
        radial_disp = self.solve_jeans().y[0][::-1]/self.nu

        theta_disp = self.angular_dispersion_from_radial(radial_disp)
        phi_disp = theta_disp

        return (radial_disp,theta_disp,phi_disp)

    def select_random_dist(self):
        """
        Randomly selects between Burkert and NFW DM distribution for when
        DM_dist is set to 'Combination' for produce_dSph.
        """
        random_int = np.random.randint(low=1, high=2, size=1, dtype=int)
        if random_int % 2 == 0:
            return self.burkert
        elif random_int % 2 != 0:
            return self.NFW

    def nu_plummer(self, r):
        self.nu_function = self.plummer
        self.normal_const = quad(self.nu_function,
                                     0, np.inf,
                                     args=(self.L, self.r_half)) 

        return ((1/self.normal_const[0]) * 
                self.nu_function(r, self.L, self.r_half)
                )

    def produce_dSph(self, D_array, DM_dist_type):
        """
        Creates a dSpheroidal galaxy of N stars governed by the above functions
        with DM distribution according to the input of DM_dist.

        Parameters
        ----------
        N : int\n
            Number of stars in the given galaxy.\n
        DM_dist : string\n
            DM distribution method; valid inputs are 'Burkert', 'NFW' or 
            'Combination', for a Burkert, NFW, or random mix of 
            distributions, respectively.
        """

        self.nu = self.nu_plummer(self.r_eval)
        N = len(self.r_eval)
        self.DM_dist_type = DM_dist_type
        DM_dist_dict = {
                        'Burkert' : self.burkert,
                        'NFW' : self.NFW,
                        'Combination' : self.select_random_dist
                        }

        self.rho_dist = DM_dist_dict[DM_dist_type]

        radial_disp, theta_disp, phi_disp = self.dispersions_from_sol()
    
        # Stellar Profile to get star positions
        cos_theta_dist = st.uniform(scale=2)
        phi_dist = st.uniform(scale=2 * np.pi)
        
        # phi, theta grid
        theta = np.linspace(0, np.pi, N),
        phi = np.linspace(0, 2 * np.pi, N)
        
        # sample phi and theta equally in 3d space
        theta_samp = np.arccos(cos_theta_dist.rvs(size=N) - 1)
        phi_samp = phi_dist.rvs(size=N)

        # create interpolation instances so we can get a dispersion value for a given r_samp 
        # r_samp is a different shape than r_eval so need to interp
        rad_disp_interp = interp1d(self.r_eval, radial_disp, kind='cubic', fill_value="extrapolate")
        theta_disp_interp = interp1d(self.r_eval, theta_disp, kind='cubic', fill_value="extrapolate")
        phi_disp_interp = interp1d(self.r_eval, phi_disp, kind='cubic', fill_value="extrapolate")

        vel_radial = np.empty_like(self.nu)
        vel_theta = np.empty_like(self.nu)
        vel_phi = np.empty_like(self.nu)

        for i, r in enumerate(self.nu):
            vel_dist_r = stat.norm(scale=np.sqrt(abs(rad_disp_interp(r))))
            vel_radial[i] = vel_dist_r.rvs(1)

            vel_dist_theta = stat.norm(scale=np.sqrt(abs(theta_disp_interp(r))))
            vel_theta[i] = vel_dist_theta.rvs(1)

            vel_dist_phi = stat.norm(scale=np.sqrt(abs(phi_disp_interp(r))))
            vel_phi[i] = vel_dist_phi.rvs(1)

        x = self.nu * np.sin(theta_samp) * np.cos(phi_samp)
        y = self.nu * np.sin(theta_samp) * np.sin(phi_samp)
        z = self.nu * np.cos(theta_samp)

        vel_x = (vel_radial*np.sin(theta_samp)*np.cos(phi_samp) +
             vel_theta*np.cos(theta_samp)*np.cos(phi_samp) -
             vel_phi*np.sin(phi_samp))

        vel_y = (vel_radial*np.sin(theta_samp)*np.sin(phi_samp) +
                vel_theta*np.cos(theta_samp)*np.sin(phi_samp) +
                vel_phi*np.cos(phi_samp))

        vel_z = (vel_radial*np.cos(theta_samp) -
                vel_theta*np.sin(theta_samp))

        D = float(np.random.choice(D_array, size=1))

        theta_x = np.arctan(x / D)
        theta_y = np.arctan(y / D)

        dSph_project = pd.DataFrame({"x": x, "y":y,"z":z,
                                    "x_velocity": vel_x, "y_velocity": vel_y, 
                                    "z_velocity": vel_z, "theta_x": theta_x, 
                                    "theta_y": theta_y})

        return dSph_project

    def gen_disk_foreground(self, r, z):
        """
        Generates a distribution of stars in the disk of the Milky Way that
        obstruct the view of dSph galaxies as foreground.

        See Stellar Contribution to the Galactic Bulge Microlensing Optical Depth
        by Cheongho Han and Andrew Gould (and their sources cited) for more information and
        deeper explanation of constants/parameters.

        Parameters
        ----------
        z : array-like of floats\n
        Distance from the galactic plane for ehich the disk stars are sampled with z = 0 being the galactic plane.\n

        Returns
        -------
        disk_dist : 2-D array of floats\n
        A sample star distribution based on the r_eval array passed to __init__ and the z array passed to this method.\n
        disk_dist.shape = (len(r_eval), len(z))
        """

        disk_density = disk_gen()
        disk_dist = disk_density._pdf(r, z)
        
        return disk_dist
        
    def gen_bulge_foreground(self, s):
        """
        Generates a distribution of stars in the "Bulge" around the centroid of the Milky Way 
        to obstruct dSph galaxies beyond the opposite side of the Milky Way.

        Shell structure.

        The radius at which Bulge becomes disc is ~3e3 pc

        Parameters
        ----------
        s : array-like of floats\n
        Radius from the galactic centre.\n

        Returns
        -------
        bulge_dist : 2-D array of floats\n
        """

        bulge_density = bulge_gen()
        bulge_dist = bulge_density._pdf(s)

        return bulge_dist

    def gen_halo_foreground(self, r, z):
        """
        Generates foreground stars that occur in halos using best fit parameters as of 2002 (in the process of looking
        for updated parameters to see if the model is still valid).
        For more information, see the halo_gen class at the top of the file.

        Parameters
        ----------
        r : array-like \n
        Radius from the galactic centre. \n
        z : array-like \n
        Distance from the galactic plane, or more specifically from the central plane of the thin disk.

        Returns
        -------
        bulge_dist : 2-D array of floats\n
        """

        halo_density = halo_gen()
        halo_dist = halo_density._pdf(r, z)

        return halo_dist