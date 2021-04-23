# -*- coding: utf-8 -*-

#################################################
# ENPH 455
# Dwarf Spheroidal Satellite Galaxy Simulation Code
# 
# Originally Two Scripts: phase_space.py and genesis.py
# Author: Daniel Friedland
# Dated: March 7, 2019 and March 22 2019
#
# Edited and Compiled by: Robert Faraday
# Dated: March 8, 2020
#################################################

# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats.distributions as st
import seaborn as sns
from scipy.integrate import quad, solve_ivp
from scipy.interpolate import interp1d
from scipy.stats import norm
import datetime
import os
import sys
import csv

# set plot style
plt.style.use('seaborn-poster')


# establish the number density class
class num_density(st.rv_continuous):  
    
    """
    A Normalized stellar number density given a Plummer Profile with
    some total luminosity and half radius
    
    
    Notes
    -------
    Is a subclass of scipy.stats.distributions.rv_continuous. By redifining _pdf
    we can also make use of all of the othr built in functions of rv_continuous
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


# define the function to generate the steallar number density for a Plummer profile
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
    
    return (3 * L * (4 * np.pi * r_half ** 3) ** (-1) * (1 + r ** 2 / r_half **2 ) ** (-5/2))


# define the function to return the dark matter density distribution for a Burkert profile
def burkert(r,*args):

    """ Returns a Dark Matter density distribution based on a Burkert DM
    density profile

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


# define the function to calculate the mass of the dark matter distribution based on the density
def mass_from_density(r, rho_args):
    
    rho_dist = rho_args[0]
    out =  quad(lambda s: (s ** 2) *rho_dist(r, *rho_args[1:]), 0, r)  # rho_args is a tuple of (rho_dist, r_core,rho_central)
    mass = 4 * np.pi * out[0]

    return mass


# define the function to generate the initial value problem for the jeans equation
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
    rho_args : tuple
        tuple of argument to be fed into mass_from_density. Of form: (rho_dist, r_core, rho_central)
    """
    # unpacking
    (nu_dist, L, r_half) = nu_args
    (rho_dist, r_core, rho_central) = rho_args

    nu = nu_dist(r, nu_args[1], nu_args[2])
    M = mass_from_density(r, rho_args)
    # M = M_fun(r, r_core, rho_central)

    # G = 6.67408e-11 #m3/kg*s2
    G = 4.302e-3  # pc*m_sun-1*(km/s)^2
    # jeans equation rearranged for U
    dUdr = (- ((G * M * nu) / r**2) - (2 * beta * U / r))
    
    return dUdr


# define the function to solve the initial value problem for the jeans equation using the RK23 method
def solve_jeans(r_eval,beta, nu_args, rho_args):
    
    """ Solves an initial value problem Jeans Equation at the . See salucci et al (2014)
    Attributes
    -----------
    r_eval : array-like variable\n
        Array of radii values at which the initial value problem is solved\n
    beta : array like\n
        Radial Velocity Dispersion * nu. What is being solved for by the ODE Solver\n
    nu_args : tuple\n
        tuple of argument to be fed into nu_fun. Of form: (L, r_half)\n
    M_args : tuple\n
        tuple of argument to be fed into M_fun. Of form: (r_core, rho_central)
    """
    sol = solve_ivp(lambda r, y: jeans_ivp(r,y,beta,nu_args,rho_args),
                    [r_eval[-1],r_eval[0]],
                    y0 = [0],
                    method = 'RK23',
                    t_eval=r_eval[::-1])
    
    return sol


# define the function to calculate the angular dispersion of the galaxy from the radial dispersion
def angular_dispersion_from_radial(beta,var_r):
    var_theta = (1-beta)*var_r
    
    return var_theta    


# define the function to calculate the dispersions from the solution of the jeans equation
def dispersions_from_sol(sol,nu,beta):
    
    radial_disp = sol.y[0][::-1]/nu

    theta_disp = angular_dispersion_from_radial(beta,radial_disp)
    phi_disp = theta_disp

    return(radial_disp,theta_disp,phi_disp)



# function to generate the dSph galaxies, main function which calls the necessary functions to generate the galaxy data
def produce_dSph(*args):
    rho_args, beta, nu_dist, N, L, r_half, nu_args, r_eval, D = args
    nu = nu_dist(r_eval, nu_args[1], nu_args[2])

    sol = solve_jeans(r_eval, beta, nu_args, rho_args)
    radial_disp, theta_disp, phi_disp = dispersions_from_sol(
        sol, nu, beta)
   
    # Stellar Profile to get star positions
    cos_theta_dist = st.uniform(scale=2)
    phi_dist = st.uniform(scale=2 * np.pi)
    r_dist = num_density(a=0, name='plummer')
    
    # phi, theta grid
    theta = np.linspace(0, np.pi, N),
    phi = np.linspace(0, 2 * np.pi, N)
    
    # sample phi and theta equally in 3d space
    theta_samp = np.arccos(cos_theta_dist.rvs(size=N) - 1)
    phi_samp = phi_dist.rvs(size=N)
    
    # sample r based on stellar profile
    r_samp = r_dist.rvs(L=nu_args[1], r_half=nu_args[2], size=N)

    # create interpolation instances so we can get a dispersion value for a given r_samp 
    # r_samp is a different shape than r_eval so need to interp
    rad_disp_interp = interp1d(r_eval, radial_disp, kind='cubic', fill_value="extrapolate")
    theta_disp_interp = interp1d(r_eval, theta_disp, kind='cubic', fill_value="extrapolate")
    phi_disp_interp = interp1d(r_eval, phi_disp, kind='cubic', fill_value="extrapolate")

    vel_radial = np.empty_like(r_samp)
    vel_theta = np.empty_like(r_samp)
    vel_phi = np.empty_like(r_samp)

    for i, r in enumerate(r_samp):
        vel_dist_r = norm(scale=np.sqrt(abs(rad_disp_interp(r))))
        vel_radial[i] = vel_dist_r.rvs(1)

        vel_dist_theta = norm(scale=np.sqrt(abs(theta_disp_interp(r))))
        vel_theta[i] = vel_dist_theta.rvs(1)

        vel_dist_phi = norm(scale=np.sqrt(abs(phi_disp_interp(r))))
        vel_phi[i] = vel_dist_phi.rvs(1)

    x = r_samp * np.sin(theta_samp) * np.cos(phi_samp)
    y = r_samp * np.sin(theta_samp) * np.sin(phi_samp)


    vel_z = (vel_radial * np.cos(theta_samp) -
             vel_theta * np.sin(theta_samp))

    D_array = np.logspace(4.07, 5.2, 100000)  # array from aprox 20-120 kpc

    D = np.random.choice(D_array, size=1)

    theta_x = np.arctan(x / D)
    theta_y = np.arctan(y / D)

    dSph_project = pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y,
                                 "z_velocity": vel_z})

    return (dSph_project)



# main body of the simulation software which establishes the initial parameters and generates the csv files
if __name__ == '__main__':
    # Parameter Definition
    # number of galaxies to produce
    num_galaxy = 10000

    # free parameter array to randomly choose from
    
    # Burkert profile parameters
    r_core_array = np.linspace(50, 2500, 10000)
    rho_central_array = np.linspace(10 ** -1.5, 10 ** 1.5, 10000)

    # NFW profile parameters
    c_param_array = np.linspace(10, 50, 40)
    M_200_array = np.linspace(7.2e7, 5.2e15, 1000000)

    # general parameters
    beta_array = 0  # np.linspace(-3.36,0.33,10000)
    D_array = np.logspace(4.07, 5.2, 100000)
    i = 0
    for galaxy in range(num_galaxy):
        
        # Anisotropy parameters
        beta = 0  # np.random.choice(beta_array)  # Orbital anisotropy

        # stellar density parameters
        nu_dist = plummer
        N = 1000  # Number of stars
        L = 2.7e5  # Solar Luminosities
        r_half = 4 #r_core * 0.4  # Radius containing half the luminosity
        nu_args = (nu_dist, L, r_half)

        r_eval = np.logspace(-3, 6, 10000)  # radius array
        D = float(np.random.choice(D_array, size=1))
        
        # randomly select DM profile parameters
        rand_select = np.random.randint(2, size=1)
        if rand_select == 0:
            rho_dist = burkert
        
        if rand_select == 1:
            rho_dist = NFW

        if rho_dist == burkert:
            # Burkert profile parameters
            r_core = float(np.random.choice(r_core_array, 1))  # [pc]
            rho_central = float(np.random.choice(rho_central_array, 1))  # [m_solar/pc^3]
            rho_args = (rho_dist, r_core, rho_central)

            # call function to produce dSph galaxies
            dSph_project = produce_dSph(rho_args, beta, nu_dist, N, L, r_half, nu_args,
                                    r_eval, D)

            dSph_keys = ['nu_dist', 'N', 'L', 'r_half', 'DM_dist',
                     'r_core', 'rho_central', 'beta', 'D']
            
            dSph_values = [nu_dist.__name__, N, L, r_half, rho_dist.__name__,
                       r_core, rho_central, beta, D]

        if rho_dist == NFW:   
            # NFW profile parameters
            c_param = float(np.random.choice(c_param_array, 1)) #[arb. units]
            M_200 = float(np.random.choice(M_200_array, 1))
            rho_args = (rho_dist, c_param, M_200)

            # call function to produce the dSph galaxies
            dSph_project = produce_dSph(rho_args, beta, nu_dist, N, L, r_half, nu_args,
                                    r_eval, D)

            dSph_keys = ['nu_dist', 'N', 'L', 'r_half', 'DM_dist',
                     'c_param', 'M_200', 'beta', 'D']
            
            dSph_values = [nu_dist.__name__, N, L, r_half, rho_dist.__name__,
                       c_param, M_200, beta, D]

        
        dSph_dict = dict(zip(dSph_keys, dSph_values))
        currentDT = datetime.datetime.now()
        
        # create save location folder if does not already exist
        save_path = "./dSph_data/{}_stars_Mix".format(str(N))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        # write dSph metadata to top of the file
        currentDT = currentDT.strftime("%Y_%m_%d_%H_%M_%S_%f")
        save_location = os.getcwd() + '\\dSph_data\\{}_stars_Mix\\dSph_'.format(N) + currentDT + '.csv'

        with open(save_location, 'w') as f:
            for key in dSph_dict.keys():
                f.write("%s,%s\n" % (key, dSph_dict[key]))

        dSph_project.to_csv(save_location, mode='a+')
        print("Created Galaxy Number {} with {} Parameter".format(i, dSph_dict))
        i = i + 1


# # Main Body for plotting galaxies at given core radii values and dark matter densities for demonstration
# if __name__ == "__main__":
#     # Parameter Definition
#     # stellar density parameters
#     r_core_list = [100, 618, 1200]
#     rho_dist = burkert
#     r_core = 618  # [pc]
#     rho_central = 0.80 # [m_solar/pc^3]
#     rho_args = (rho_dist, r_core, rho_central)
#     # Other parameters
#     beta = - 0.1 # Orbital anisotropy
#
#     nu_dist = plummer
#     N = 1000  # Number of stars
#     L = 2.7e5  # Solar Luminosities
#     r_half =  0.4*r_core  # Radius containing half the luminosity
#     nu_args = (nu_dist, L, r_half)
#     # DM profile parameters
#
#     r_eval = np.logspace(-3,6,10000)  # radius array
#
#
#     #%% Calculate M and nu for plotting later if needed
#     nu = nu_dist(r_eval,nu_args[1],nu_args[2])
#     M = []
#     for r in r_eval:
#         M.append(mass_from_density(r, rho_args))
#     M = np.asarray(M)
#     #%% Solve Jeans Equation for velocity dispersion
#     sol = solve_jeans(r_eval,beta,nu_args,rho_args)
#     radial_disp,theta_disp, phi_disp = dispersions_from_sol(
#            sol,nu,beta)
#     # plt.figure(0)
#     # plt.loglog((sol.t[::-1]),np.sqrt((radial_disp[i][:])), label = r'$\sigma_{{r}},  Beta = {}$'.format(beta))
#     #%% Stellar Profile to get star positions
#     cos_theta_dist = st.uniform(scale = 2)
#     phi_dist = st.uniform(scale = 2* np.pi)
#     r_dist = num_density(a=0, name = 'plummer')
#     #phi, theta grid
#     theta = np.linspace(0, np.pi, N),
#     phi = np.linspace(0, 2* np.pi, N)
#     # sample phi and theta equally in 3d space
#     theta_samp = np.arccos(cos_theta_dist.rvs(size = N) - 1)
#     phi_samp = phi_dist.rvs(size = N)
#     # sample r based on steller profile
#     r_samp  = r_dist.rvs(L = nu_args[1],r_half = nu_args[2],size = N)
#
#
#
#     # create interpolation instances so we can get a dispersion value for a given r_samp
#     # r_samp is a different shape than r_eval so need to interp
#     rad_disp_interp = interp1d(r_eval,radial_disp, kind = 'cubic')
#     theta_disp_interp = interp1d(r_eval,theta_disp, kind = 'cubic')
#     phi_disp_interp = interp1d(r_eval,phi_disp, kind = 'cubic')
#
#     # radial_dispersion = rad_disp_interp(r_samp)
#     # theta_dispersion = theta_disp_interp(r_samp)
#     # phi_dispersion = phi_disp_interp(r_samp)
#
#     # now can produce boltzmann distribution for each r and variance
#     vel_radial = np.empty_like(r_samp)
#     vel_theta = np.empty_like(r_samp)
#     vel_phi = np.empty_like(r_samp)
#
#     for i, r in enumerate(r_samp):
#         vel_dist_r = norm(scale =np.sqrt(rad_disp_interp(r)))
#         vel_radial[i] = vel_dist_r.rvs(1)
#
#         vel_dist_theta = norm(scale =np.sqrt(theta_disp_interp(r)))
#         vel_theta[i] = vel_dist_theta.rvs(1)
#
#         vel_dist_phi = norm(scale =np.sqrt(phi_disp_interp(r)))
#         vel_phi[i] = vel_dist_phi.rvs(1)
#     #%%
#     # constuct a dataframe of physical variable
#     x = r_samp*np.sin(theta_samp)*np.cos(phi_samp)
#     y = r_samp*np.sin(theta_samp)*np.sin(phi_samp)
#     z = r_samp*np.cos(theta_samp)
#
#     vel_x = (vel_radial*np.sin(theta_samp)*np.cos(phi_samp) +
#              vel_theta*np.cos(theta_samp)*np.cos(phi_samp) -
#              vel_phi*np.sin(phi_samp))
#
#     vel_y = (vel_radial*np.sin(theta_samp)*np.sin(phi_samp) +
#              vel_theta*np.cos(theta_samp)*np.sin(phi_samp) +
#              vel_phi*np.cos(phi_samp))
#
#     vel_z = (vel_radial*np.cos(theta_samp) -
#             vel_theta*np.sin(theta_samp))
#
#     D_array = np.logspace(4.07,5.2,100000) # array from aprox 20-120 kpc
#
#     D = np.random.choice(D_array, size = 1)
#
#     theta_x = np.arctan(x/D)
#     theta_y = np.arctan(y/D)
#
#     #theta_obs = np.arctan(np.sqrt(x**2+y**2) / D)
#
#     # dSph_sphere = pd.DataFrame({"radius": r_samp, "theta":theta_samp, "phi":phi_samp,
#     #  "radial_velocity": vel_radial, "theta_velocity": vel_theta, "phi_velocity": vel_phi})
#
#     dSph = pd.DataFrame({"x": x, "y":y,"z":z,
#                          "x_velocity": vel_x, "y_velocity": vel_y, "z_velocity": vel_z})
#
#     dSph_project = pd.DataFrame({"theta_x": theta_x, "theta_y": theta_y,
#                                  "z_velocity": vel_z})
#     #%% plotting
#     plt.style.use('default')
#     mpl.rcParams.update({'font.size': 25,'figure.titlesize': 35,'xtick.major.size':25, 'ytick.major.size':25})
#
#     plt.close('all')
#     fig = plt.figure()
#     fig.set_size_inches(15, 15)
#     ax = fig.gca(projection='3d')
#     #ax.set_aspect("equal")
#     velocity = np.sqrt(dSph['x_velocity']**2 + dSph['y_velocity']**2 + dSph['z_velocity']**2)
#     #ax.plot_wireframe(x, y, z, color = 'b', rstride=1, cstride=1, alpha = 0.2)
#     ax.scatter(dSph['x'].values,dSph['y'].values,dSph['z'].values,s = 30 , c = velocity, cmap = 'viridis', edgecolor = None)
#     plt.suptitle("dSph Galaxy Simulated with {} Tracer Stars \n According to Plummer Profile".format(N), size = 30)
#     ax.set_xlabel('X [Pc]', labelpad = 18,size = 25)
#     ax.set_ylabel('Y [Pc]', labelpad = 18,size = 25)
#     ax.set_zlabel('Z [Pc]',labelpad = 18, size = 25)
#     ax.set_title('$r_h$ = {:.0f}'.format(0.4*r_core))
#     #norm = mpl.colors.Normalize(vmin=0,vmax=2)
#     sm = plt.cm.ScalarMappable(cmap='viridis', norm=None)
#     sm.set_array([])
#     cbar = plt.colorbar(sm, ticks=np.linspace(0,int(max(velocity)),11, endpoint = True),
#                         boundaries=(np.linspace(min(velocity),max(velocity))))
#
#     #plt.colorbar( label='digit value')
#     #cbar = plt.colorbar()
#     cbar.set_label( r'Star Speed, $V = \sqrt{V_{x}^2 +V_{y}^2 + V_{z}^2}$ [Km/s]',size = 20)
#     #plt.tight_layout()
#     plt.savefig('3Dplummer_speed_rh{}.png'.format(0.4*r_core), transparent = True, bbox_inches = 'tight', pad_inches = 0)
#
#     # project plot
#
#     if fig2 is not None:
#         plt.close(fig2)
#     fig2 = plt.figure()
#     ax2 = fig2.gca()
#
#     velocityz = np.array(dSph_project['z_velocity'].values)
#     arc_x = dSph_project['theta_x'].values * (60 * 180)/np.pi
#     arc_y = dSph_project['theta_y'] *(60*180)/np.pi
#     ax2.scatter(arc_x,arc_y.values,s = 35 ,
#                 c = velocityz, cmap = 'bwr', edgecolor = 'black')
#
#     ax2.set_title("2D Projected dSph Galaxy Simulated with {} Stars ".format(N), size = 35)
#     ax2.set_xlabel('$\Theta_{x} = \mathrm{arctan}(x/D)}$  [arcmin]', labelpad = 18,size = 25)
#     ax2.set_ylabel('$\Theta_{y} = \mathrm{arctan}(y/D)$  [arcmin]', labelpad = 18,size = 25)
#     sm2 = plt.cm.ScalarMappable(cmap='bwr', norm=None)
#     sm2.set_array([])
#     plt.tight_layout()
#     cbar2 = plt.colorbar(sm2, ticks=np.linspace(int(min(velocityz)),int(max(velocityz)),11, endpoint = True),
#                         boundaries=(np.linspace(min(velocityz),max(velocityz))))
#     cbar2.set_label( r'Stellar Line of Sight Velocity, $V = V_{z}$ [Km/s]',size = 25)
#     plt.savefig('2Dplummer_speed_rh{}.png'.format(0.4*r_core), transparent = True, bbox_inches = 'tight', pad_inches = 0)
#
#
#     # saving
#
#     vel_dist_r_long = maxwell(scale =np.sqrt(radial_disp[0]))
#     norm_const = quad(plummer,0,np.inf,args = (L,r_half))[0]
#     r_samp2 = np.random.choice(r_eval,N,p=nu/norm_const)
#     plt.loglog(r_eval,nu)
#     plt.hist(r_samp)