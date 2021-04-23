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
import math
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import scipy.stats.distributions as st
from scipy.integrate import quad, dblquad, solve_ivp
from scipy.interpolate import interp1d
from scipy.special import kv
import scipy as sci
import os
import datetime
import random

class disk_gen(st.rv_continuous):
    """
    Volumetric mass density of stars within the Disk
    
    Parameters
    ----------
    r : numpy array \n
        Mesh grid of radii from the galactic centre. \n
    z : numpy array \n
        Mesh grid of distance from the galactic plane. \n

    Returns
    -------
    disk_density : aray-like \n
        The local stellar mass density contribution from the disk.
    """
    def _pdf(self, r, z):
        H = 2.75e3 # length scale of the disk [pc]
        solar_mass = 1 # mass of the sun, common mass scale in astrophysics [solar mass]
        rho_0_disk = 0.0493*solar_mass # central disk density [M_odot/pc^2]
        beta_disk = 0.565 # orbital anisotropy of stars in the disk []
        h1 = 270 # [pc]
        h2 = 440 # [pc]
        R_0 = 8e3 # ~radius where bulge -> disk [pc]
        # print(r)

        # need to check to see if r is a single value or multiple
        if hasattr(range, '__len__') or isinstance(r, list):
            r = np.asarray(r) # cast to array if list
            if len(r.shape) > 1:
                nu_disk = np.fromiter(map(lambda x : np.maximum(x[0] / 9025 + 0.114, 0.670), r), np.float64)
            else:
                nu_disk = np.maximum(r / 9025 + 0.114, 0.670)
        else:
            nu_disk = np.maximum(r / 9025 + 0.114, 0.670)

        hypsec = stat.hypsecant.pdf(z/nu_disk*h1)
        disk_density = ((rho_0_disk/nu_disk)*np.exp(-(r-R_0)/H)*
                        ((1-beta_disk)*hypsec**2+beta_disk*np.exp(-abs(z)/(nu_disk*h2)))) # [solar masses/pc^3]
        return disk_density

class bulge_gen(st.rv_continuous):
    """
    Volumetric mass density of stars within the Bulge
    
    Parameters
    ----------
    r : aray-like \n
        Mesh grid of radii from the galactic centre. \n
    z : aray-like \n
        Mesh grid of distance from the galactic plane. \n

    Returns
    -------
    bulge_density : aray-like \n
        The local stellar mass density contribution from the bulge.
    """
    def inner_dist(self, s):
        inner_bulge_dist = 1.04e6*(s/0.482)**(-1.85) # [solar masses/pc^3]    
        return inner_bulge_dist
    def outer_dist(self, s):
        outer_bulge_dist = 3.53*kv(0, (s/667)) # [solar masses/pc^3]
        return outer_bulge_dist
    def _pdf(self, r, z):
        piecewise_shift = 938 # [pc]
        s = np.sqrt(r**2 + z**2)
        if hasattr(s, '__len__') or isinstance(s, list):
            s = np.asarray(s)
            inner_density = list(map(lambda x : self.inner_dist(x), s[s < piecewise_shift]))
            outer_density = list(map(lambda x : self.outer_dist(x), s[s >= piecewise_shift]))
            bulge_density = np.asarray(inner_density + outer_density)
        else:
            if s < piecewise_shift:
                bulge_density = self.inner_dist(s)
            else:
                bulge_density = self.outer_dist(s)
        return bulge_density

class NormalVelocityDistribution(st.rv_continuous):
    """
    Normal curve for describing a velocity dispersion pdf,
    as described by the dispersion (variance) about the mean velocity (experimentally determined
    and filled in by the user).
    """
    def __init__(self, momtype=1, a=0, b=1e2, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)

    def _pdf(self, v, mean, disp):
        coeff = 1/np.sqrt(disp*2*np.pi)
        exponent = -0.5*((v-mean)/disp**0.5)**2
        return coeff*np.exp(exponent)

    def _rvs(self, mean, disp):
        xbounds = [self.a, self.b]
        pmax = self._pdf(mean, mean, disp)
        while True:
            x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
            y = np.random.rand(1)*pmax
            if y<=self._pdf(x, mean, disp):
                return x


class chabrier_imf(st.rv_continuous):
    """
    IMF distribution.

    See https://en.wikipedia.org/wiki/Initial_mass_function for the gist 
    or https://sites.astro.caltech.edu/~ccs/ay124/chabrier03_imf.pdf for primary sources.

    Sets the minimum allowable sample value with a and maximum with b, both representative of the smallest observable
    masses found and setting the upper limit due to greatly diminshing relative abundance.

    """
    def __init__(self, momtype=1, a=10**(-1), b=10**2, xtol=1e-14, badvalue=None, name=None, longname=None, shapes=None, extradoc=None, seed=None):
        super().__init__(momtype=momtype, a=a, b=b, xtol=xtol, badvalue=badvalue, name=name, longname=longname, shapes=shapes, extradoc=extradoc, seed=seed)
        self.__maxval__ = self.b
        self.__minval__ = self.a

    def _pdf(self, m):
        M_C = np.log10(0.079) # another fit constant
        sigma2 = 0.69**2    # it's based on a statistical distribution, so of course it has sigma = 0.69!
        alpha = 1.3    # based on the Salpeter distribution
        A_1 = 0.158     # fit constant
        A_2 = 4.43*10**(-2) # fit constant
        A_bound = np.exp(-(M_C ** 2) / 2.0 / sigma2) # value of the function at the piecewise boundary
        C_norm = 0.2171675868106134 # integral over the given bounds, used for normalization

        a1 = A_1 / C_norm * np.exp(-((np.log10(m) - M_C) ** 2) / 2.0 / sigma2)
        a2 = A_2 / C_norm * m ** (-alpha)
        return np.where(m <= 1, a1, a2)

    def _rvs(self):
        xbounds = [self.a, self.b]
        pmax = self._pdf(self.a)
        while True:
            x = np.random.rand(1)*(xbounds[1]-xbounds[0])+xbounds[0]
            y = np.random.rand(1)*pmax
            if y<=self._pdf(x):
                return x

# class mw_num_density(st.rv_continuous):
#     """
#     Combined number density of halo, thin and thick disk stars. The bulge is a
#     bit trickier, but a 'synthetic bulge' approach is considered, as seen in:

#     https://arxiv.org/pdf/1308.0593v1.pdf
#     """

#     def _pdf(self, r, z, Z_solar, rho_D_0, L1, H1, f, L2, H2, f_H):
#         """
#         The disk & halo  number density, parameters given in Table 3
        
#         https://faculty.washington.edu/ivezic/Publications/tomographyI.pdf
#         """
        
#         q_H = 0.6 # ~0.5-1 for common values, most recent as of above source for MW was 0.6; halo ellipticity
#         R_solar = 2.25461e-8 # astronomical constant
#         n_H = 2.6 # ~2.5-3.0 fit parameter
#         halo_num_density = rho_D_0*f_H*(R_solar/(r**2+(z/q_H)**2)**(1/2))**(n_H)
#         thin_disk_num_density = rho_D_0*np.exp(R_solar/L1)*np.exp(-abs(r)/L1)*np.exp(-abs(z+Z_solar)/H1)
#         thick_disk_num_density = rho_D_0*np.exp(R_solar/L2)*np.exp(-abs(r)/L2)*np.exp(-abs(z+Z_solar)/H2)
#         return halo_num_density + thin_disk_num_density + f*thick_disk_num_density

class GenerateDSPH():
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

    def cylRay2Sph(self, r, z):
        """
        Converts a 1D or 2D array of cylindrical r and z co-ordinates to their spherical radial
        equivalent. (Meant for internal class use)

        Parameters
        ----------
        r : array-like \n
            Radius array, either 1D or 2D.
        z : array-like \n
            Distance from galactic plane array, either 1D or 2D.

        Returns
        -------
        s : array-like \n
            1D array of the spherical radial component corresponding to the
            same index r and z values.
        """
        if r.shape != z.shape:
            print("r and z dimensions do not match shape of r is: "+r.shape+ " and shape of z is: "+z.shape)
            return None
        if r.shape[1] == 1:
            s = np.zeros(shape=(len(r), len(z)))
            for ri in range(0, len(r)):
                for zi in range(0, len(z)):
                    s[ri, zi] = np.sqrt(r[ri]**2 + z[zi]**2) # every r applied with every z
        if r.shape[1] > 1:
            r = r[0, :] # every unique r
            z = z[0, :] # every unique z
            s = np.zeros(shape=(len(r), len(z)))
            for ri in range(0, len(r)):
                for zi in range(0, len(z)):
                    s[ri, zi] = np.sqrt(r[ri]**2 + z[zi]**2) # every r applied with every z

        s.reshape(1, s.size) # reshape the spherically symmetric radius so it's compatible with the velocity calculation functions

        return s

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

        return (radial_disp, theta_disp, phi_disp)

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

        return (
                (1/self.normal_const[0]) * 
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
        theta = np.linspace(0, np.pi, N)
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

class GenerateForeground():
    """

    """
    def GeV2Modot(self, val):
        """
        Converts the value in Gev/cm^3 to M_odot/pc^3
        """
        return 0.008/0.3*val

    def cylRay2Sph(self, r, z):
        """
        Converts a 1D or 2D array of cylindrical r and z co-ordinates to their spherical radial
        equivalent. (Meant for internal class use)

        Parameters
        ----------
        r : array-like \n
            Radius array, either 1D or 2D.
        z : array-like \n
            Distance from galactic plane array, either 1D or 2D.

        Returns
        -------
        s : array-like \n
            1D array of the spherical radial component corresponding to the
            same index r and z values.
        """
        if r.shape != z.shape:
            print("r and z dimensions do not match shape of r is: "+r.shape+ " and shape of z is: "+z.shape)
            return None
        if r.shape[1] == 1:
            s = np.zeros(shape=(len(r), len(z)))
            for ri in range(0, len(r)):
                for zi in range(0, len(z)):
                    s[ri, zi] = np.sqrt(r[ri]**2 + z[zi]**2) # every r applied with every z
        if r.shape[1] > 1:
            r = r[0, :] # every unique r
            z = z[0, :] # every unique z
            s = np.zeros(shape=(len(r), len(z)))
            for ri in range(0, len(r)):
                for zi in range(0, len(z)):
                    s[ri, zi] = np.sqrt(r[ri]**2 + z[zi]**2) # every r applied with every z

        s.reshape(1, s.size) # reshape the spherically symmetric radius so it's compatible with the velocity calculation functions

        return s

    def anotherDMdensityFunction(self, r, g=0, rs=14.45e3, rho0=0.43, R0=8.2e3, profile='NFW'):
        """
        Determines the local DM density for a given profile, fit parameters and
        spherical radius.

        Parameters
        ----------
        r : spherical radius [pc]\n
        rs : scale radius [pc]\n
        g : parameterized slope, defaulting to 0 for a purely spherical distribution\n
        rho0 : parameterized central DM mass density [GeV/cm^3]\n
        R0 : another radius scaling parameter [pc]\n
        profile : the DM profile being used, often NFW for MW

        Returns
        -------
        rho : DM denisty at spherical radius r for a given profile and parameters 
        """
        if (profile == 'NFW'):
            a = 1;
            b = 3;
            rhos  = rho0/(2**((b-g)/a))*(R0/rs)**g*(1+(R0/rs)**a)**((b-g)/a); # DM Density at r        
            rho = rhos*(2**((b-g)/a))/((r/rs)**g)/((1+(r/rs)**a)**((b-g)/a));
            return rho

        elif (profile == 'Einasto'):
            rhos = rho0*np.exp(2/g*((R0/rs)**g-1));
            rho = rhos*np.exp(-2/g*((r/rs)**g-1));
            return rho

        elif (profile == 'Burkert'):
            u = r/rs;
            rhos=rho0*(1+u)*(1+u**2);
            rho = rhos/(1+u)/(1+u**2);
            return rho
        else:
            return 

    def gen_disk_pdf(self, r, z):
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
        
    def gen_bulge_pdf(self, r, z):
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
        bulge_dist = bulge_density._pdf(r, z)

        return bulge_dist

    def gen_halo_pdf(self, r, z):
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

    def stellar_mass_density(self, r, z):
        """
        Add together the arrays of local stellar densities
        """
        return self.gen_disk_pdf(r,z) + self.gen_bulge_pdf(r,z) #+ self.gen_halo_pdf(r,z)

    def tdSphVec(self, gVec, eVec):
        """
        Function to convert galactocentric triplet vectors (cyl) to an earth-centric vector (cyl)
        for determination of the "vision cone" of a dSph from Earth, taking the galactocentric vector as reference.

        [r, theta, z]
        """
        v = np.sqrt(gVec[0]**2+eVec[0]**2-gVec[0]*eVec[0]*np.cos(gVec[1])) # radial distance from e to dsph
        return np.array([v, eVec[1], gVec[2]-eVec[2]])

    # def e2GC(self, eVec, rs):
    #     """
    #     Converts from co-ordinates reached by rs in Earth's frame of reference to the galactocentric co-ordinate represntation
    #     """

    #     alpha = math.atan(rs[2] / rs[0])
    #     phi = math.atan(-eVec[2] / (rs[0] * math.cos(alpha)))

    #     return np.array([eVec[0] * math.cos(np.pi/2 - phi) + rs[0] * math.cos(phi), np.pi/2 - alpha, rs[2] - eVec[2]])

    def mCarloStar(self, p):
        """
        Uses the Monte Carlo method to determine whether a star will be created for a given
        number probability. Used by star_cone for N < 1 for a given control volume.
        
        Parameters
        ----------
        p : float
            Probability that a star exists within the CV, or simply N for N < 1.
        """
        if p <= np.random.rand(1):
            return True
        else:
            return False

    def phidm(self, s, z, g=0, rs=14.45e3, rho0=0.43, R0=8.2e3, profile='NFW'):
        """
        
        """
        r = np.sqrt(s**2+z**2)
        self.cm = 1e-2
        self.GeV = 1.78e-27 # kg
        rho_integrand = lambda x : x**2*self.anotherDMdensityFunction(x, g=0, rs=14.45e3, rho0=0.43, R0=8.2e3, profile='NFW')*self.GeV/(self.cm**3)
        if isinstance(r, np.ndarray) or isinstance(r, list):
            phi = np.zeros(shape=(1,len(r)))
            for i in range(len(r)):
                phi[i] = -4*np.pi*self.kpc*self.GN/r[i]**2*quad(rho_integrand, 0, r[i])[0]
        else:
            integration = quad(rho_integrand, 0, r) # need an additional line ot access first element
            phi = -4*np.pi*self.kpc*self.GN/r**2*integration[0]
        return phi

    def bulgeDphiDr(self, s, z):
        """

        """
        r = np.sqrt(s**2+z**2)
        co = 0.6 # kpc
        dphidr = -1/(r+co)**2
        return dphidr

    def diskDphiDr(self, s, z):
        """

        """
        r = np.sqrt(s**2+z**2)
        bd = 4 #kpc
        dphidr = -1/r**2*(1-np.exp(-r/bd)) + 1/(r*bd)*np.exp(-r/bd)
        return dphidr

    def MWDispersions(self, s, z, g=0, rs=14.45e3, rho0=0.43, R0=8.2e3, profile='NFW'):
        """
        Calculation of velocity dispersions using the Jeans relationship for the disk, bulge and DM in the MW
        """
        self.Msun = 2e30 # kg
        self.Mb = 1.5e10 * self.Msun # kg Bulge mass
        self.Md = 7e10 * self.Msun # kg Disk mass
        self.GN = 6.67e-11 # m^3 kg^-1 s^-2
        self.kpc = 3.086e19 # m
        self.GeV = 1.78e-27 # kg

        # cylindrical position to spherical position
        r = np.sqrt(s**2+z**2)
        dm_disp = np.zeros(shape=(1,len(r)))
        bulge_disp = np.zeros(shape=(1,len(r)))
        disk_disp = np.zeros(shape=(1,len(r)))

        phi_bulge = lambda x,y : self.GN*self.Mb/self.kpc**2*self.bulgeDphiDr(x, y)
        phi_disk = lambda x,y : self.GN*self.Md/self.kpc**2*self.diskDphiDr(x, y)
        rhodm = self.stellar_mass_density(s, z) # need stellar mass density
        # halo and bulge spherically symetric but disk not -> important distinction!
        # z comp. of the disk -> assume DM doesn't vary as much

        DMintegrand = lambda x,y : self.stellar_mass_density(x, y)*self.phidm(x, y, profile, rs, g, rho0, R0)
        BulgeIntegrand = lambda x,y : self.stellar_mass_density(x, y)*phi_bulge(x, y)
        DiskIntegrand = lambda x,y : self.stellar_mass_density(x, y)*phi_disk(x, y)

        if isinstance(r, (np.ndarray, list)):
            for i in range(len(r)):
                dm_disp[i] = -1/rhodm[i]*dblquad(DMintegrand, r[i], np.inf, lambda x: np.sqrt(r[i]**2+x**2), lambda x: np.inf)[0]*self.kpc
                bulge_disp[i] = -1/rhodm[i]*dblquad(BulgeIntegrand, r[i], np.inf, lambda x: np.sqrt(r[i]**2+x**2), lambda x: np.inf)[0]*self.kpc
                disk_disp[i] = -1/rhodm[i]*dblquad(DiskIntegrand, r[i], np.inf, lambda x: np.sqrt(r[i]**2+x**2), lambda x: np.inf)[0]*self.kpc
        else:
            dm_disp = -1/rhodm*dblquad(DMintegrand, r, np.inf, lambda x: np.sqrt(r**2+x**2), lambda x: np.inf)[0]*self.kpc
            bulge_disp = -1/rhodm*dblquad(BulgeIntegrand, r, np.inf, lambda x: np.sqrt(r**2+x**2), lambda x: np.inf)[0]*self.kpc
            disk_disp = -1/rhodm*dblquad(DiskIntegrand, r, np.inf, lambda x: np.sqrt(r**2+x**2), lambda x: np.inf)[0]*self.kpc
        
        total_disp = dm_disp + bulge_disp + disk_disp

        return total_disp

    def changeFrame(self, v, rotationmat, translationmat=np.array([[0.0],[0.0],[0.0]]), direction="forward"):
        """
        Applies a frame rotation and/or translation to vector v
        """
        if direction == 'forward':
          modv = v - translationmat
          return np.matmul(rotationmat, modv)
        if direction == 'inverse':
          invrot = np.linalg.inv(rotationmat)
          return np.matmul(invrot, v) + translationmat

    def genRandomPointMass(self, stellar_component, N, rcv, rn, rs_max, rdSph, mu_d=20e3, mu_b=10e3, cut_excess=True, excess_ratio=100):
        """
        Generates point masses with positions contained within the control volume in question,
        using the geometry of the dSph problem to provide galactocentric co-ordinates.

        Parameters
        ----------
        N : int/float\n
            Number of stars expected within the CV. If it isn't a whole number, the remainder is
            converted into a probability for an additional star to be generated.\n
        rcv : (r, theta, z) numpy array\n
            Vector that goes from the Earth's reference frame to the CV point.\n
        rn : (r, theta, z) numpy array
            Vector from the CV point to the outer boundary layer.\n
        rs_max : (r, theta, z) numpy array \n
            The vector that reaches the final outer bound, corresponding to the edge of dSph.\n
        rdSph : float \n
            Radius of the dwarf spheroidal galaxy being evaluated.\n
        alpha : float\n
            Angle at which z rises corresponding to r, see default case.\n
        cut_excess : boolean\n
            If N is much greater than 1 at ratio excess_ratio, the probability calculation
            just slows the calculation for marginal impact.
        """

        # need an IMF!
        cIMF = chabrier_imf()
        # Gaussian velocity distribution
        vgdist = NormalVelocityDistribution()
        # temporary array to be appended to our final mass array and then saved to an excel spreadsheet later
        mass_df = pd.DataFrame(columns=["r", "theta", "z", "m", "component", "v_theta"])
        # assign the correct component
        if stellar_component == 'bulge' or stellar_component == 'disk' or stellar_component == 'halo':
          component = stellar_component
        else:
          component = 'undefined'
        # needed locally as well
        alpha = math.atan(rcv[2]/rcv[0])
        # get the line-of-sight vector for more realistic representation of foreground stars
        los_rs = self.changeFrame(rcv.reshape(3,1), self.LoS_rotation)
        # vector that goes from the CV point to the outer bound
        rn = np.array([[rn],[0.0],[0.0]])
        nmax = 1

        # local number is less than 1, use the monte carlo method to randomly determine if a star will be generated
        if (N < 1):
            if (self.mCarloStar(N)):
                m = cIMF._rvs()
                n = np.random.rand(1) * (nmax - (-nmax)) - nmax # random position along the axial vector
                ax_pos = los_rs + rn*n
                h_frac = np.sqrt(ax_pos[0]**2+ax_pos[2]**2)/rs_max
                s_max = h_frac * rdSph # cone boundary based on problem symmetry
                theta = np.random.rand(1) * 2 * np.pi # random angle within the cone
                r = np.random.rand(1) * s_max # random value along the radial vector between 0 and s_max
                z = ax_pos[2]
                starpos = np.ones(shape=(3,1))
                starpos[0] = r
                starpos[1] = theta
                starpos[2] = z
                starpos = np.array([[r],[theta],[z]]) # position of the star in LoS frame, need to change for velocity 
                hc_starpos = self.changeFrame(v=starpos.reshape(3,1), rotationmat=self.LoS_rotation, direction='inverse')
                gc_starpos = self.changeFrame(v=hc_starpos.reshape(3,1), rotationmat=self.G2E_rotation, translationmat=self.G2E_translation, direction='inverse')
                dispersion = self.MWDispersions(s=gc_starpos[0], z=gc_starpos[2])
                if component == 'disk':
                    v_theta = vgdist._rvs(mean=mu_d, disp=dispersion)
                elif component == 'bulge':
                    v_theta = vgdist._rvs(mean=mu_b, disp=dispersion)
                else:
                    v_theta = np.array([0.0])
                mass_df=mass_df.append({"r":r[0],'theta':theta[0],'z':z[0],'m':m[0],'component':component, 'v_theta':v_theta[0]}, ignore_index=True)
            return mass_df
        # include the extra as a percentage to generate another star
        if (N >= 1 and N < N * excess_ratio):
            for i in range(int(N)):
                m = cIMF._rvs()
                n = np.random.rand(1) * (nmax - (-nmax)) - nmax # random position along the axial vector
                ax_pos = los_rs + rn*n
                h_frac = np.sqrt(ax_pos[0]**2+ax_pos[2]**2)/rs_max
                s_max = h_frac * rdSph # cone boundary based on problem symmetry
                theta = np.random.rand(1) * 2 * np.pi # random angle within the cone
                r = np.random.rand(1) * s_max # random value along the radial vector between 0 and s_max
                z = ax_pos[2]
                starpos = np.ones(shape=(3,1))
                starpos[0] = r
                starpos[1] = theta
                starpos[2] = z
                starpos = np.array([[r],[theta],[z]]) # position of the star in LoS frame, need to change for velocity 
                hc_starpos = self.changeFrame(v=starpos.reshape(3,1), rotationmat=self.LoS_rotation, direction='inverse')
                gc_starpos = self.changeFrame(v=hc_starpos.reshape(3,1), rotationmat=self.G2E_rotation, translationmat=self.G2E_translation, direction='inverse')
                dispersion = self.MWDispersions(s=gc_starpos[0], z=gc_starpos[2])
                if component == 'disk':
                    v_theta = vgdist._rvs(mean=mu_d, disp=dispersion)
                elif component == 'bulge':
                    v_theta = vgdist._rvs(mean=mu_b, disp=dispersion)
                else:
                    v_theta = np.array([0.0])
                mass_df=mass_df.append({"r":r[0],'theta':theta[0],'z':z[0],'m':m[0],'component':component, 'v_theta':v_theta[0]}, ignore_index=True)
            if (self.mCarloStar(N-int(N))):
                m = cIMF._rvs()
                n = np.random.rand(1) * (nmax - (-nmax)) - nmax # random position along the axial vector
                ax_pos = los_rs + rn*n

                h_frac = np.sqrt(ax_pos[0]**2+ax_pos[2]**2)/rs_max
                s_max = h_frac * rdSph # cone boundary based on problem symmetry
                theta = np.random.rand(1) * 2 * np.pi # random angle within the cone
                r = np.random.rand(1) * s_max # random value along the radial vector between 0 and s_max
                z = ax_pos[2]
                starpos = np.ones(shape=(3,1))
                starpos[0] = r
                starpos[1] = theta
                starpos[2] = z
                starpos = np.array([[r],[theta],[z]]) # position of the star in LoS frame, need to change for velocity 
                hc_starpos = self.changeFrame(v=starpos.reshape(3,1), rotationmat=self.LoS_rotation, direction='inverse')
                gc_starpos = self.changeFrame(v=hc_starpos.reshape(3,1), rotationmat=self.G2E_rotation, translationmat=self.G2E_translation, direction='inverse')
                dispersion = self.MWDispersions(s=gc_starpos[0], z=gc_starpos[2])
                if component == 'disk':
                    v_theta = vgdist._rvs(mean=mu_d, disp=dispersion)
                elif component == 'bulge':
                    v_theta = vgdist._rvs(mean=mu_b, disp=dispersion)
                else:
                    v_theta = np.array([0.0])
                mass_df=mass_df.append({"r":r[0],'theta':theta[0],'z':z[0],'m':m[0],'component':component, 'v_theta':v_theta[0]}, ignore_index=True)
            return mass_df
        # many stars, no need to including percentage chance for this CV
        if (N > excess_ratio * N):
            for i in range(int(N)):
                m = cIMF._rvs()
                n = np.random.rand(1) * (nmax - (-nmax)) - nmax # random position along the axial vector
                ax_pos = los_rs + rn*n
                h_frac = np.sqrt(ax_pos[0]**2+ax_pos[2]**2)/rs_max
                s_max = h_frac * rdSph # cone boundary based on problem symmetry
                theta = np.random.rand(1) * 2 * np.pi # random angle within the cone
                r = np.random.rand(1) * s_max # random value along the radial vector between 0 and s_max
                z = ax_pos[2] # it should be along the line of sight
                starpos = np.ones(shape=(3,1))
                starpos[0] = r
                starpos[1] = theta
                starpos[2] = z
                starpos = np.array([[r],[theta],[z]]) # position of the star in LoS frame, need to change for velocity 
                hc_starpos = self.changeFrame(v=starpos.reshape(3,1), rotationmat=self.LoS_rotation, direction='inverse')
                gc_starpos = self.changeFrame(v=hc_starpos.reshape(3,1), rotationmat=self.G2E_rotation, translationmat=self.G2E_translation, direction='inverse')
                dispersion = self.MWDispersions(s=gc_starpos[0], z=gc_starpos[2])
                if component == 'disk':
                    v_theta = vgdist._rvs(mean=mu_d, disp=dispersion)
                elif component == 'bulge':
                    v_theta = vgdist._rvs(mean=mu_b, disp=dispersion)
                else:
                    v_theta = np.array([0.0])
                mass_df=mass_df.append({"r":r[0],'theta':theta[0],'z':z[0],'m':m[0],'component':component, 'v_theta':v_theta[0]}, ignore_index=True)
            return mass_df

    def starCone(self, name="star_data", rG=10e3, zG=80e3, r_dSph=1e3, theta=np.pi/8, rE=8e3, zE=43, ncv=32):
        """
        Creates a cone of stars within the field of view of a dwarf spheroidal galaxy as foreground noise
        relative to the position of earth from galactocentric coordinates.

        Parameters
        ----------
        rG : float
            Galactocentric radius for which the centre of the viewed dSph exists [pc]
        zG : float
            Distance above the galactic plane where the core the core of the dSph resides [pc]
        rdSph : float
            Radius of the dSph [pc]
        phi : float
            Angle from the galactic normal vector to the anti-norml vector [rad]
        theta : float
            Angle in the galactic plane relative to the radial vector to earth [rad]
        rE : float
            Current galactocentric radius at which Earth exists in the galactic plane [pc]
        zE : float
            Distance above the galactic plane at which Earth exists [pc]

        Returns
        -------
        star_cone : dataframe
            Generated star masses and their positions {[pc],[pc],[pc],[solar masses]}
        """

        # IMF used, more could be implemented later
        imf = chabrier_imf()
        bulge_mass_dist = bulge_gen()
        disk_mass_dist = disk_gen()
        vdist = NormalVelocityDistribution()
        # star dataframe
        starcone = pd.DataFrame()
        # integration bounds for number density normalization
        m_min = 10**(-1)
        m_max = 10**2

        gVec = np.array([rG, 0, zG]) # galaxy core to core of dSph
        eVec = np.array([rE, theta, zE]) # galaxy core to Earth
        rsVec = self.tdSphVec(gVec, eVec) # Earth to core of dSph
        
        rs = rsVec[0] # radial distance from Earth to dSph core
        zs = rsVec[2] # perpendicular distance from the plane of Earth parallel to the galactic plane
        alpha = math.atan(rsVec[2] / rs) # angle of climb of the central axis of the "view cone"
        beta = np.pi/2 - alpha # z axis rotation for line of sight
        phi = math.atan(-eVec[2] / rs) # angle of Earth's radial vector projection (rs) relative to the GC's (rG)
        rs_mag = np.sqrt(rs**2 + zs**2) # magnitude the Earth to dSph vector
        rs_hat = rsVec / rs_mag # unit vector for Earth to dSph
        radial_slope = r_dSph / rs_mag # scaling parameter for CV calculations

        # rotation matrix going from galactocentric co-ordinates to earth-centric co-ordinates with parallel
        # z-axes
        self.G2E_rotation = np.array([[np.cos(theta), -np.sin(theta), 0.0],
                                [np.sin(theta), np.cos(theta), 0.0],
                                [0.0, 0.0, 1.0]])
        # translation from galactocentric frame to heliocentric frame
        self.G2E_translation = np.array([[rE*np.cos(theta)],[rE*np.sin(theta)],[zE]])
        # rotation from MW cylindrical co-ordinates to line-of sight frame
        self.LoS_rotation = np.array([[np.cos(beta), 0.0, -np.sin(beta)],
                                [0.0, 1.0, 0.0],
                                [np.sin(beta), 0.0, np.cos(beta)]])

        # volume of a cone section, using the relationship for a total cone radius and height. Takes smaller radius and 
        # cone section height; a byproduct of a new unused method
        CV_func = lambda h,r : np.pi/3*(3*r**2*h + 3*radial_slope*r*h**2 + radial_slope*2*h**3) # volume of a cone section
        # generate log spaced lines to subdivide the cone with greater resolution near Earth
        outer_bound = np.logspace(0, math.log(rs_mag - r_dSph, 10), num=ncv+1)
        # enclose these as control volumes
        inner_bound = np.hstack(([0.0], [outer_bound[i] for i in range(len(outer_bound)-1)]))
        # find the centre point of the vector spanning the height of the control volume
        cv_point = np.fromiter(map(lambda i : (outer_bound[i] - outer_bound[i-1])/2 + outer_bound[i-1], range(1, len(outer_bound))), np.float64)
        # determine the radius of the vision cone for a given distance along the line of sight vector
        bound_r = np.hstack((np.fromiter(map(lambda i : inner_bound[i]*radial_slope, range(len(inner_bound))), np.float64), r_dSph))
        # match up the volume of each CV to the corresponding radius
        deltaV = np.fromiter(map(lambda i : CV_func(outer_bound[i] - outer_bound[i-1], bound_r[i-1]), range(1,len(outer_bound))), np.float64)
        # stellar number normaliztion numerator
        [intNum, numErr] = quad(lambda m : imf._pdf(m), a=m_min, b=m_max)
        # stellar number normaliztion denominator
        [intDenom, denomErr] = quad(lambda m : imf._pdf(m)*m, a=m_min, b=m_max)
        # stellar number normalization for the Chabrier IMF with 
        n_normalization = intNum / intDenom
        # get the galactocentric co-ordinates for the equivalent Earth reference frame vector along rs
        rsEarth = np.array([cv_point * math.cos(alpha), np.zeros(len(cv_point)), cv_point * math.sin(alpha)]).T
        # eVec2D = np.array([eVec[0]*np.ones(len(cv_point)), eVec[1]*np.ones(len(cv_point)), eVec[2]*np.ones(len(cv_point))]).T
        axial_vector = np.zeros(shape=rsEarth.shape)
        for i in range(len(cv_point)):
            v = rsEarth[i,:].reshape(3,1)
            val = self.changeFrame(v, rotationmat=self.G2E_rotation, translationmat=self.G2E_translation, direction="inverse")
            axial_vector[i, :] = np.transpose(val)
        bulge_mass = bulge_mass_dist._pdf(axial_vector[:,0], axial_vector[:,2])
        disk_mass = disk_mass_dist._pdf(axial_vector[:,0], axial_vector[:,2])
        # number of stars in a given control volume
        N_bulge = n_normalization * bulge_mass * deltaV
        N_disk = n_normalization * disk_mass * deltaV
        # distance between each bound and the CV point
        rn = np.fromiter(map(lambda i: (outer_bound[i+1]-outer_bound[i])/2, range(0, len(outer_bound)-1)), np.float64)

        """Uncomment for position testing"""
        # print("N_bulge: "+str(len(N_bulge)))
        # print("rsEarth: "+str(len(rsEarth)))
        # print("rn: "+str(len(rn)))
        # N_bulge = np.ones(shape=(len(cv_point), 1))
        # N_disk = np.ones(shape=(len(cv_point), 1))
        
        for i in range(len(cv_point)):
            print("Working on CV {} of {}".format(i+1, len(cv_point)))
            starcone=starcone.append(self.genRandomPointMass('bulge', N_bulge[i], rsEarth[i], rn[i], rs_mag, r_dSph), ignore_index=True)
            starcone=starcone.append(self.genRandomPointMass('disk', N_disk[i], rsEarth[i], rn[i], rs_mag, r_dSph), ignore_index=True)
        xlsxfile = name + ".xls"
        starcone.to_excel(xlsxfile)
        # for a number less than one, apply that as the probability to generate a star