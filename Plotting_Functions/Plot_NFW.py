# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 17:34:08 2019

@author: Daniel
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:42:11 2019

@author: Daniel
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
#mpl.rcParams['lines.markersize'] = 10
import numpy as np
import pandas as pd
import scipy.stats.distributions as st
#from distribution_selection import plummer_num_density, burkert_density
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import  norm
import datetime
import os
import sys
import csv


plt.style.use('seaborn-poster')
class num_density(st.rv_continuous):  
    
    """
    A Normalized stellar number density given a Plummer Profile with
    some total luminosity and half radius
    
    
    Notes
    -------
    Is a subclass of scipy.stats.distributions.rv_continuous. By redifining _pdf
    we can also make use of all of the other built in functions of rv_continuous
    such as rvs which allows us to produce random variates of the underlying
    distribution
    
    See: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html#scipy.stats.rv_continuous"""
   
    def _argcheck(self, *args):
        """Default check for correct values on args and keywords.

        Returns condition array of 1's where arguments are correct and
         0's where they are not.

        """
        cond = 1
        
        for arg in args[:]:
            cond = np.logical_and(cond, (np.asarray(arg) > 0 ))
            
        return cond
    
    
    
    def _pdf(self, r, L, r_half):
        #print(args)
        function_mappings = {
        'plummer': plummer,
        }
        self.nu_function = function_mappings[self.name]  
        #print(self.nu_function)
        
        if not hasattr(self,'normal_const'):
            self.normal_const = quad(self.nu_function,
                                     0, np.inf,
                                     args=(L,r_half)) 

        return ((1/self.normal_const[0]) * 
                self.nu_function(r, L,r_half)
                )


def plummer(r,L,r_half):
    """ Returns the Stellar number density given a Plummer Profile with
    some total luminosity and half radius

    Attributes
    -----------
    r : array-like variable
        Radius from centre of galaxy (units)
    L : float
        Total Luminosity (units)
    r_half : float
        Radius where half of luminosity is contained (units)

    Returns
    ----------
    nu : array-like number density"""
    # L = args[0]
    # r_half = args[1]
    
    return (3 * L *
            (4 * np.pi * r_half ** 3) ** (-1) *
            (1 + r ** 2 / r_half **2 ) ** (-5/2))

def burkert(r,*args):

    """ Returns a Dark Matter density distribution based on a Burkert DM
    density

    Attributes
    -----------
    r : array-like variable
        Radius from centre of galaxy (units)
    r_core : float
        Core Radius Radius where half of luminosity is contained (units)
    rho_central : float
        central density
    """
    r_core =args[0]
    #print('rcore:',args[0])

    rho_central = args[1]
    #print('rho_central:',args[1])

    return ((rho_central * (r_core ** 3)) /
            ((r + r_core) * (r ** 2 + r_core ** 2))
            )

# define the function to calculate the the dark matter distribution for an NFW profile
def NFW(r, *args):

    """ Returns a Dark Matter density distribution based on a NFW DM
    density profile

    Attributes
    -----------
    r : array-like variable
        Radius from centre of galaxy (units)
    c_param : float
        Concentration parameter which determines the characteristic radius and the characteristic overdensity
    M_200 : float
        Mass of the dark matter halo used to calculate the radius of the dark matter halo (virial mass)
    """
    
    c_param = args[0]
    #print('c_param:', args[0])

    M_200 = args[1]
    #print('M_200:', args[1])

    rho_crit = 2.77536627e-7*0.674**2 # critical density of the universe [m_sun/pc]

    delta_c = (200/3)*(c_param**3)/(np.log(1+c_param)-c_param/(1+c_param)) # characteristic overdensity [arb. units]
    
    r_200 = (3*M_200/(200*rho_crit*4*np.pi))**(1/3) # virial radius [pc]

    r_char = r_200/c_param # characeristic radius [pc]

    return ((rho_crit*delta_c)/((r/r_char)*(1+r/r_char)**2))

def mass_from_density(r, rho_args):
    
    rho_dist = rho_args[0]
    out =  quad(lambda s: (s ** 2) *rho_dist(r, *rho_args[1:]), 0, r)  # rho_args is a tuple of (rho_dist, r_core,rho_central)
    mass = 4 * np.pi * out[0]

    return mass
def U_fun(r,beta,nu_args, rho_dist, rho_args):
    #G = 6.67408e-11 #m3/kg*s2
    G = 4.302e-3 #pc*m_sun-1 (km/s)^2
    (L, r_half) = nu_args
   # (r_core, rho_central) = M_args
   # nu = nu_fun(r,L,r_half)
   # M = M_fun(r, r_core, rho_central)

    integrand = lambda r:r**((2*beta)-2)*nu_fun(r,L,r_half)*Mass_fun(r,NFW,rho_args)
    I = quad(integrand,r.min(),r.max())[0]
    U = G*r**(-2*beta)*I

    return U

def jeans_ivp(r,U,beta,nu_args,rho_args):
    """ Solves an initial value problem Jeans Equation. See salucci et al (2014)
    Attributes
    -----------
    r : array-like variable
        Radius from centre of galaxy (units)
    U : array like
        Radial Velocity Dispersion * nu. What is being solved for by the ode Solver
    nu_args : tuple
        tuple of argument to be fed into nu_fun. Of form: (L, r_half)
    M_args : tuple
        tuple of argument to be fed into M_fun. Of form: (r_core, rho_central
    """
    # unpacking
    (nu_dist, L, r_half) = nu_args
    (rho_dist, c_param, M_200) = rho_args

    nu = nu_dist(r, nu_args[1], nu_args[2])
    M = mass_from_density(r, rho_args)
    # M = M_fun(r, r_core, rho_central)

    # G = 6.67408e-11 #m3/kg*s2
    G = 4.302e-3  # pc*m_sun-1*(km/s)^2
    # jeans equation rearranged for U
    dUdr = (- ((G * M * nu) / r**2) -
            (2 * beta * U / r)
            )
    return dUdr
def solve_jeans(r_eval,beta, nu_args, rho_args):
    sol = solve_ivp(lambda r, y: jeans_ivp(r,y,beta,nu_args,rho_args),
                    [r_eval[-1],r_eval[0]],
                    y0 = [0],
                    method = 'RK23',
                    t_eval=r_eval[::-1])
    return sol

def angular_dispersion_from_radial(beta,var_r):
    var_theta = (1-beta)*var_r
    return var_theta

def dispersions_from_sol(sol,nu,beta):
    
    radial_disp = sol.y[0][::-1]/nu

    theta_disp = angular_dispersion_from_radial(beta,radial_disp)
    phi_disp = theta_disp
    
    

    return(radial_disp,theta_disp,phi_disp)
#%% Main Body
if __name__ == "__main__":
    # Parameter Definition
    # stellar density parameters
    M_200_list = [8e7, 10e10, 10e12, 5e15]
    dSph_list = []
    dSph_project_list = []
    for M_200 in M_200_list:
            
        rho_dist = NFW
        #r_core = 100, 618, 1200,  # [pc]
        r_core = 618
        c_param = 30 # [m_solar/pc^3]
        rho_args = (rho_dist, c_param, M_200)
        # Other parameters
        beta = - 0.1 # Orbital anisotropy
        
        nu_dist = plummer
        N = 1000  # Number of stars
        L = 2.7e5  # Solar Luminosities
        r_half =  0.4*r_core  # Radius containing half the luminosity
        nu_args = (nu_dist, L, r_half)
        # DM profile parameters
        
        r_eval = np.logspace(-3,6,10000)  # radius array
        
    
        #%% Calculate M and nu for plotting later if needed
        nu = nu_dist(r_eval,nu_args[1],nu_args[2])
        M = []
        for r in r_eval:
            M.append(mass_from_density(r, rho_args))
        M = np.asarray(M)
        #%% Solve Jeans Equation for velocity dispersion
        sol = solve_jeans(r_eval,beta,nu_args,rho_args)
        radial_disp,theta_disp, phi_disp = dispersions_from_sol(
               sol,nu,beta)
        # plt.figure(0)
        # plt.loglog((sol.t[::-1]),np.sqrt((radial_disp[i][:])), label = r'$\sigma_{{r}},  Beta = {}$'.format(beta))
        #%% Stellar Profile to get star positions
        cos_theta_dist = st.uniform(scale = 2)
        phi_dist = st.uniform(scale = 2* np.pi)
        r_dist = num_density(a=0, name = 'plummer')
        #phi, theta grid
        theta = np.linspace(0, np.pi, N),
        phi = np.linspace(0, 2* np.pi, N)
        # sample phi and theta equally in 3d space
        theta_samp = np.arccos(cos_theta_dist.rvs(size = N) - 1)
        phi_samp = phi_dist.rvs(size = N)
        # sample r based on steller profile
        r_samp  = r_dist.rvs(L = nu_args[1],r_half = nu_args[2],size = N)
       
        
     
        # create interpolation instances so we can get a dispersion value for a given r_samp 
        # r_samp is a different shape than r_eval so need to interp
        rad_disp_interp = interp1d(r_eval,radial_disp, kind = 'cubic')
        theta_disp_interp = interp1d(r_eval,theta_disp, kind = 'cubic')
        phi_disp_interp = interp1d(r_eval,phi_disp, kind = 'cubic')
        
        # radial_dispersion = rad_disp_interp(r_samp)
        # theta_dispersion = theta_disp_interp(r_samp)
        # phi_dispersion = phi_disp_interp(r_samp)
        
        # now can produce boltzmann distribution for each r and variance
        vel_radial = np.empty_like(r_samp)
        vel_theta = np.empty_like(r_samp)
        vel_phi = np.empty_like(r_samp)
        
        for i, r in enumerate(r_samp):
            vel_dist_r = norm(scale =np.sqrt(rad_disp_interp(r)))
            vel_radial[i] = vel_dist_r.rvs(1)
            
            vel_dist_theta = norm(scale =np.sqrt(theta_disp_interp(r)))
            vel_theta[i] = vel_dist_theta.rvs(1)
            
            vel_dist_phi = norm(scale =np.sqrt(phi_disp_interp(r)))
            vel_phi[i] = vel_dist_phi.rvs(1)
        #%%
        # constuct a dataframe of physical variable
        x = r_samp*np.sin(theta_samp)*np.cos(phi_samp)
        y = r_samp*np.sin(theta_samp)*np.sin(phi_samp)
        z = r_samp*np.cos(theta_samp) 
        
        vel_x = (vel_radial*np.sin(theta_samp)*np.cos(phi_samp) + 
                 vel_theta*np.cos(theta_samp)*np.cos(phi_samp) -
                 vel_phi*np.sin(phi_samp))
        
        vel_y = (vel_radial*np.sin(theta_samp)*np.sin(phi_samp) +
                 vel_theta*np.cos(theta_samp)*np.sin(phi_samp) +
                 vel_phi*np.cos(phi_samp))
        
        vel_z = (vel_radial*np.cos(theta_samp) -
                vel_theta*np.sin(theta_samp))
        
        D_array = np.logspace(4.07,5.2,100000) # array from aprox 20-120 kpc
        
        D = np.random.choice(D_array, size = 1)
        
        theta_x = np.arctan(x/D)
        theta_y = np.arctan(y/D)
        
        #theta_obs = np.arctan(np.sqrt(x**2+y**2) / D)
        
        # dSph_sphere = pd.DataFrame({"radius": r_samp, "theta":theta_samp, "phi":phi_samp,
        #  "radial_velocity": vel_radial, "theta_velocity": vel_theta, "phi_velocity": vel_phi})
                
        dSph_list.append(pd.DataFrame({"x": x, "y":y,"z":z,
                             "x_velocity": vel_x, "y_velocity": vel_y, "z_velocity": vel_z}))
        
        dSph_project_list.append(pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y,
                                     "z_velocity": vel_z}))
    #%% plotting
    
    plt.style.use('seaborn-whitegrid')
    mpl.rcParams.update({'font.size': 25,'figure.titlesize': 35,'xtick.major.size':25, 'ytick.major.size':25})
    
    #plt.style.use('default')
    out_keys = ['r_core', 'rho_central', 'beta', "D"]
    plt.close('all')
    fig = plt.figure()
    fig.set_size_inches(15, 15)
    #axx = [fig.add_subplot(1,1,1+i, projection='3d') for i in range(1)]
    for i, (dSph) in enumerate(dSph_list):
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111,projection='3d')
        #fig.gca(projection='3d')
        ax.set_aspect("auto", anchor = None)
        radius = np.array(np.sqrt(dSph['x']**2 + dSph['y']**2 + dSph['z']**2))
        #ax.plot_wireframe(x, y, z, color = 'b', rstride=1, cstride=1, alpha = 0.2)
        ax.scatter(dSph['x'].values,dSph['y'].values,dSph['z'].values, c = radius,cmap = 'hot',edgecolor = 'dimgrey',s = 40)
        ax.set_title("$M_200 = {}$".format(M_200_list[i]), size = 30)
        ax.set_xlabel('X [Pc]', labelpad =20,size = 25)
        ax.set_ylabel('Y [Pc]', labelpad =20,size = 25)
        ax.set_zlabel('Z [Pc]',labelpad = 20, size = 25)
        ax.tick_params('x',which='major', labelsize=20)
        ax.tick_params('y',which='major', labelsize=20)
        ax.tick_params('z',which='major', labelsize=20)
        plt.suptitle('Star Position for dSph Galaxies Simulated with {} Stars'.format(N),size = 35)
        sm = plt.cm.ScalarMappable(cmap='hot', norm=None)
        sm.set_array([])
        plt.tight_layout()
        cbar = plt.colorbar(sm, ticks=np.linspace(0,int(max(radius)),11, endpoint = True),
                        boundaries=(np.linspace(min(radius),max(radius))))
        cbar.set_label( r'Radius from Centre [pc]',size = 25)
        plt.tight_layout()
        plt.savefig('plummer_position3D_NFW{}.pdf'.format(M_200_list[i]), format='pdf', dpi=1200, transparent = True, bbox_inches = 'tight', pad_inches = 0)
        
    #%%  
    plt.close('all')
    plt.style.use('seaborn-paper')
    mpl.rcParams.update({'font.size': 25,'figure.titlesize': 35,'xtick.major.size':30, 'ytick.major.size':30})
    
    #plt.style.use('default')
    out_keys = ['r_core', 'rho_central', 'beta', "D"]

    fig = plt.figure()
    fig.set_size_inches(15, 15)
    #axx = [fig.add_subplot(1,1,1+i, projection='3d') for i in range(1)]
    for i, (dSph) in enumerate(dSph_project_list):
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111,projection=None)
        #fig.gca(projection='3d')
        
        
        velocityz = np.array(dSph['z_velocity'].values)
        arc_x = dSph['theta_x'].values * (60 * 180)/np.pi
        arc_y = dSph['theta_y'] *(60*180)/np.pi
        ax.scatter(arc_x,arc_y.values,s = 30 ,
                    c = velocityz, cmap = 'viridis', edgecolor = 'black')
        ax.tick_params('both',which='major', labelsize=20)
        ax.set_title("2D Projected dSph Galaxy Simulated with {} Stars According to Plummer Profile".format(N), size = 35)
        ax.set_xlabel('$\Theta_{x} = \mathrm{arctan}(x/D)}$  [arcmin]',size = 25)
        ax.set_ylabel('$\Theta_{y} = \mathrm{arctan}(y/D)$  [arcmin]',size = 25)
        sm2 = plt.cm.ScalarMappable(cmap='viridis', norm=None)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ticks=np.linspace(int(min(velocityz)),int(max(velocityz)),11, endpoint = True),
                            boundaries=(np.linspace(min(velocityz),max(velocityz))))
        cbar2.set_label( r'Stellar Line of Sight Velocity, $V = V_{z}$ [Km/s]',size = 25)
        plt.tight_layout()
        plt.savefig('plummer_position2D_NFW{}.pdf'.format(M_200_list[i]), format='pdf', dpi=1200, transparent = True, bbox_inches = 'tight', pad_inches = 0)

#%%
    plt.close('all')
    plt.style.use('seaborn-paper')
    mpl.rcParams.update({'font.size': 25,'figure.titlesize': 35,'xtick.major.size':30, 'ytick.major.size':30})
    
    #plt.style.use('default')
    out_keys = ['r_core', 'rho_central', 'beta', "D"]

    fig = plt.figure()
    fig.set_size_inches(15, 15)
    #axx = [fig.add_subplot(1,1,1+i, projection='3d') for i in range(1)]
    for i, (dSph) in enumerate(dSph_project_list):
        fig = plt.figure()
        fig.set_size_inches(20, 15)
        ax = fig.add_subplot(111,projection=None)
        #fig.gca(projection='3d')
        
        
        velocityz = np.array(dSph['z_velocity'].values)
        arc_x = dSph['theta_x'].values * (60 * 180)/np.pi
        arc_y = dSph['theta_y'] *(60*180)/np.pi
        ax.scatter(arc_x,arc_y.values,s = 30 ,
                    c = velocityz, cmap = 'viridis', edgecolor = 'black')
        ax.tick_params('both',which='major', labelsize=20)
        ax.set_title("2D Projected dSph Galaxy Simulated with {} Stars According to Plummer Profile".format(N), size = 35)
        ax.set_xlabel('$\Theta_{x} = \mathrm{arctan}(x/D)}$  [arcmin]',size = 25)
        ax.set_ylabel('$\Theta_{y} = \mathrm{arctan}(y/D)$  [arcmin]',size = 25)
        sm2 = plt.cm.ScalarMappable(cmap='viridis', norm=None)
        sm2.set_array([])
        cbar2 = plt.colorbar(sm2, ticks=np.linspace(int(min(velocityz)),int(max(velocityz)),11, endpoint = True),
                            boundaries=(np.linspace(min(velocityz),max(velocityz))))
        cbar2.set_label( r'Stellar Line of Sight Velocity, $V = V_{z}$ [Km/s]',size = 25)
        plt.tight_layout()
        plt.savefig('plummer_position2D_NFW{}.pdf'.format(M_200_list[i]), format='pdf', dpi=1200, transparent = True, bbox_inches = 'tight', pad_inches = 0)

        
