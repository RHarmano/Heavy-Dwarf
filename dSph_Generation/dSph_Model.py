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
import scipy.stats as stat
import scipy.stats.distributions as st
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.special import kv
import os
import datetime
import random

class disk_gen(st.rv_continuous):
    """Volumetric mass density of stars within the Disk"""
    def _pdf(self, r, z):
        H = 2.75e3 # length scale of the disk [pc]
        solar_mass = 1 # mass of the sun, common mass scale in astrophysics [solar mass]
        rho_0_disk = 0.0493*solar_mass # central disk density [M_odot/pc^2]
        beta_disk = 0.565 # orbital anisotropy of stars in the disk []
        h1 = 270 # [pc]
        h2 = 440 # [pc]
        R_0 = 8e3 # ~radius where bulge -> disk [pc]

        try:
            nu_disk = np.zeros(shape=(len(r), 1))
            for i in range(len(r)):
                nu_disk[i,1] = np.maximum(r[i]/9025+0.114, 0.670)
        except:
            nu_disk = np.array([np.maximum(r/9025+0.114, 0.670)])
        R, Z = np.meshgrid(r, z) # need a meshgrid for cylindrical distribution
        hypsec = stat.hypsecant.pdf(Z/nu_disk*h1)
        disk_density = ((rho_0_disk/nu_disk)*np.exp(-(R-R_0)/H)*
                        ((1-beta_disk)*hypsec**2+beta_disk*np.exp(-abs(Z)/(nu_disk*h2)))) # [solar masses/pc^3]
        return disk_density

class bulge_gen(st.rv_continuous):
    """Volumetric mass density of stars within the Bulge"""
    def inner_dist(self, s):
        inner_bulge_dist = 1.04e6*(s/0.482)**(-1.85) # [solar masses/pc^3]    
        return inner_bulge_dist
    def outer_dist(self, s):
        outer_bulge_dist = 3.53*kv(0, (s/667)) # [solar masses/pc^3]
        return outer_bulge_dist
    def _pdf(self, r, z):
        piecewise_shift = 938 # [pc]
        s = (r**2+z**2)**0.5
        bulge_density = np.zeros(shape=s.shape)
        for i in range(r.shape[0]):
            for j in range(r.shape[1]):
                if s[i,j] < piecewise_shift:
                    bulge_density[i,j] = self.inner_dist(s[i,j])
                else:
                    bulge_density[i,j] = self.outer_dist(s[i,j])
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

class chabrier_imf(st.rv_continuous):
    """
    IMF distribution.

    See https://en.wikipedia.org/wiki/Initial_mass_function for the gist 
    or https://sites.astro.caltech.edu/~ccs/ay124/chabrier03_imf.pdf for primary sources.

    Assumes the mass array is 10**(m)

    """

    def _pdf(self, m):
        A_1 = 0.158 # fit constant
        M_C = np.log10(0.079) # another fit constant
        sigma2 = 0.69**2    # it's based on a statistical distribution, so of course it has sigma = 0.69!
        alpha = -2.3    # based on the Salpeter distribution

        a1 = A_1 * np.exp(-((m - M_C) ** 2) / 2.0 / sigma2)
        a2 = 2 * (10.0 ** m) ** alpha
        return np.where(m <= 0, a1, a2)

class mw_num_density(st.rv_continuous):
    """
    Combined number density of halo, thin and thick disk stars. The bulge is a
    bit trickier, but a 'synthetic bulge' approach is considered, as seen in:

    https://arxiv.org/pdf/1308.0593v1.pdf
    """

    def _pdf(self, r, z, Z_solar, rho_D_0, L1, H1, f, L2, H2, f_H):
        """
        The disk & halo  number density, parameters given in Table 3
        
        https://faculty.washington.edu/ivezic/Publications/tomographyI.pdf
        """
        
        q_H = 0.6 # ~0.5-1 for common values, most recent as of above source for MW was 0.6; halo ellipticity
        R_solar = 2.25461e-8 # astronomical constant
        n_H = 2.6 # ~2.5-3.0 fit parameter
        halo_num_density = rho_D_0*f_H*(R_solar/(r**2+(z/q_H)**2)**(1/2))**(n_H)
        thin_disk_num_density = rho_D_0*np.exp(R_solar/L1)*np.exp(-abs(r)/L1)*np.exp(-abs(z+Z_solar)/H1)
        thick_disk_num_density = rho_D_0*np.exp(R_solar/L2)*np.exp(-abs(r)/L2)*np.exp(-abs(z+Z_solar)/H2)
        return halo_num_density + thin_disk_num_density + f*thick_disk_num_density

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

    def gen_mw_foreground_num_density(self, r, z):
        """
        Generates the number density distribution using fit parameters for the disk & halo using:

        https://faculty.washington.edu/ivezic/Publications/tomographyI.pdf
        parameters are in Table 3.

        Parameters are determined by r-i bins.
        TABLE3Best-Fit Values (Joint Fits, Bright Parallax Relation)2Bin(R;0)L1H1fL2H2fH1.61.......   1:3<r-i<1:4  0.0058  2150 245 0.13 3261 743...1:2<ri<1:3  0.00541:1<ri<1:2  0.00461:0<ri<1:1  0.00381.70.......   0:9<ri<1:0  0.0032  2862 251 0.12 3939 647 0.005070:8<ri<0:9  0.00270:7<ri<0:8  0.00240:65<ri<0:7  0.0011
        """

        num_density_mw_bodies  = mw_num_density()
        rhoD_0 = 1e-3*np.array([5.8, 5.4, 4.6, 3.8, 3.2, 2.7, 2.4, 1.1])
        L1 = np.array([2150, 2862])
        L2 = np.array([3261, 3939])
        H1 = np.array([245, 251])
        H2 = np.array([743, 647])
        f = np.array([0.13, 0.12])
        f_H = np.array([0.0, 0.00507])
        Z_solar = 10 # 10-50 [pc] depending on paper consulted

        density_matrix = num_density_mw_bodies._pdf(r, z, Z_solar, rhoD_0[0], L1[0], H1[0], f[0], L2[0], H2[0], f_H[0])
        # parameters switch to include stars bright enough to be visible in the halo, around 1.0 > r-i > 0.9
        for red_index in range(1, len(rhoD_0)):
            if red_index >= 4:
                density_matrix += num_density_mw_bodies._pdf(r, z, Z_solar, rhoD_0[red_index], L1[1], H1[1], f[1], L2[1], H2[1], f_H[1])
            else:
                density_matrix += num_density_mw_bodies._pdf(r, z, Z_solar, rhoD_0[red_index], L1[0], H1[0], f[0], L2[0], H2[0], f_H[0])
        
        return density_matrix

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

    def chabrier_sample(self, m):
        """
        Provides the change in number of stars per mass bin with respect to the mass in question
        """

        # IMPORTANT: this is dN/dm and needs the offset still!
        imf_pdf = chabrier_imf(name='chabrier_imf')._pdf
        if len(m) > 1:
            chabrier_array = np.array([])
            for mass in m:
                chabrier_array = np.append(chabrier_array, imf_pdf(mass))
            return chabrier_array
        else:
            return imf_pdf(m)

    def imf(self, local_density, mmin=0.01, mmax=100, Mcm=10000, imf_type='Chabrier'):
        """
        Generates a sample distribution for the Chabrier IMF
        """
        mmin_log = np.log10(mmin)
        mmax_log = np.log10(mmax)

        chunksize = 10
        result = np.array([], dtype=np.float64)

        while result.sum() < local_density:
            m = np.random.uniform(mmin_log, mmax_log, size=chunksize)
            x = np.random.uniform(0, 1, size=chunksize)
            result = np.hstack((result, 10 ** m[x < self.chabrier_sample(m)]))

        return result[result.cumsum() < local_density]

    def generate_stars_position(self, r, z):
        """
        Creates stars distributed at random intervals on the cylinder using finite control volumes
        at the given radii and distances from the galactic plane.
        
        Parameters
        ----------
        r : array_like \n
        Radii from the galactic centre (MUST NOT INCLUDE 0). \n
        z : array_like \n
        Distance perpendicular to the galactic plane (not disk).
        """
        [R, Z] = np.meshgrid(r, z) # matrix needed for random generation
        theta = list(map(lambda x : np.transpose(np.linspace(-1/x, 1/x)), r)) # generates a theta map such that the volume of the CV is held at 1 pc^3
        phi = np.arange(0, 2*np.pi, np.pi/100) # intervals around the galactic centre, function of r to prevent overlap of CVs
        for ri in range(len(r)):
            for zi in range(len(z)):
                

        return stars_frame

    def anotherDMdensityFunction(self, r, profile, rs, g, rho0, R0):
        """
        DM density functions by profile given
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

    def phidm(self, r, profile, rs, g, rho0, R0):
        """
        
        """
        self.cm = 1e-2;
        phi = np.zeros(shape=(1,len(r)))
        rho_integrand = lambda x : x**2*self.anotherDMdensityFunction(x, profile, rs, g, rho0, R0)*self.GeV/(self.cm**3)
        for i in range(len(r)):
            phi[i] = -4*np.pi*self.kpc*self.GN/r[i]**2*quad(rho_integrand, 0, r[i])
        return phi

    def bulge_omega(self, r):
        """

        """
        co = 0.6 # kpc
        dphidr = -1/(r+co)**2
        return dphidr

    def disk_omega(self, r):
        """

        """
        bd = 4 #kpc
        dphidr = -1/r**2*(1-np.exp(-r/bd)) + 1/(r*bd)*np.exp(-r/bd)
        return dphidr

    def MW_velocity_dispersions(self, r, profile, rs, g, rho0, R0):
        """
        Calculation of velocity dispersions using the Jeans relationship for the disk, bulge and DM in the MW
        """
        self.Msun = 2e30 # kg
        self.Mb = 1.5e10*Msun # kg Bulge mass
        self.Md = 7e10*Msun # kg Disk mass
        self.GN = 6.67e-11 # SI units
        self.kpc = 3.086e19 # m
        self.GeV = 1.78e-27 # kg

        dm_disp, bulge_disp, disk_disp = np.zeros(shape=(1,len(r)))

        phi_bulge = lambda r : self.GN*self.Mb/self.kpc**2*self.bulge_omega(r)
        phi_disk = lambda r : self.GN*self.Md/self.kpc**2*self.disk_omega(r)
        rhodm = lambda x : self.anotherDMdensityFunction(x, profile, rs, g, rho0, R0)

        DMintegrand = lambda x : rhodm(x)*self.phidm(x, profile, rs, g, rho0, R0)
        BulgeIntegrand = lambda x : self.rhodm(x)*phi_bulge(x)
        DiskIntegrand = lambda x : self.rhodm(x)*phi_disk(x)

        for i in range(len(r)):
            dm_disp[i] = 1/rhodm(r[i])*quad(DMintegrand, np.inf, r[i])*self.kpc
            bulge_disp[i] = 1/rhodm(r[i])*quad(BulgeIntegrand, np.inf, r[i])*self.kpc
            disk_disp[i] = 1/rhodm(r[i])*quad(DiskIntegrand, np.inf, r[i])*self.kpc
        
        total_disp = dm_disp + bulge_disp + disk_disp

        return [total_disp, dm_disp, bulge_disp, disk_disp]