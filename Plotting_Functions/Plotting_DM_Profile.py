"ENPH 455 Plotting of Core-Cusp Profiles"


import os
import numpy as np
import matplotlib.pyplot as plt

if not os.path.isdir("./Final_Figs"):
    os.mkdir("./Final_Figs")

# set up matplotlib text parameters
plt.rc('xtick', labelsize=14)    # fontsize of the x tick labels
plt.rc('ytick', labelsize=14)    # fontsize of the y tick labels
plt.rc('axes', labelsize=18)     # fontsize of the axes labels
plt.rc('axes', titlesize=18)     # fontsize for title
plt.rc('xtick.major', pad=8)     # pad between x tick labels and axis labels
plt.rc('ytick.major', pad=8)     # pad between y tick labels and axis labels


# define the function to return the dark matter density distribution for a Burkert profile
def burkert(r, r_core, rho_central):

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

    return ((rho_central * (r_core ** 3)) /
            ((r + r_core) * (r ** 2 + r_core ** 2))
            )


# define the function to calculate the the dark matter distribution for an NFW profile
def NFW(r, c_param, M_200):

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
    
    rho_crit = 2.77536627e-7*0.674**2 # critical density of the universe [m_sun/pc]

    delta_c = (200/3)*(c_param**3)/(np.log(1+c_param)-c_param/(1+c_param)) # characteristic overdensity [arb. units]
    
    r_200 = (3*M_200/(200*rho_crit*4*np.pi))**(1/3) # virial radius [pc]

    r_char = r_200/c_param # characeristic radius [pc]

    return ((rho_crit*delta_c)/((r/r_char)*(1+r/r_char)**2))


def mid(x):
    mid = int(len(x)/2)
    return mid

def plotting(r_eval, data, xlim, filename):
    fig , ax = plt.subplots(figsize = (8,6))

    ax.set(xlabel='Galaxy Radius [Pc]', ylabel='Dark Matter Density [M/$pc^3$]', xlim=xlim, title=filename)
    
    ax.plot(r_eval, data[:,0], color='r', lw=2)
    ax.plot(r_eval, data[:,1], color='g', lw=2)
    ax.plot(r_eval, data[:,2], color='b', lw=2)

    ax.legend(['Min', 'Mid', 'Max'], loc = 'upper right', fancybox = True, fontsize = 'x-large', framealpha = 1)


    fig.tight_layout()
    plt.savefig('./Final_Figs/' + filename + '.pdf', format='pdf', dpi=1200, bbox_inches='tight')
    plt.show()

# Burkert profile parameters
r_core_array = np.linspace(50, 2500, 10000)
rho_central_array = np.linspace(10 ** -1.5, 10 ** 1.5, 10000)

# NFW profile parameters
c_param_array = np.linspace(10, 50, 40)
M_200_array = np.linspace(7.2e7, 5.2e15, 1000000)

# set evaluation radius
r_eval = np.logspace(-3, 6, 10000)  # radius array

# select the burkert profile parameters
r_core_samp = (np.min(r_core_array), r_core_array[mid(r_core_array)], np.max(r_core_array))
rho_central_samp = (np.min(rho_central_array), rho_central_array[mid(rho_central_array)], np.max(rho_central_array))

# select the NFW profile parameters
c_param_samp = (np.min(c_param_array), c_param_array[mid(c_param_array)], np.max(c_param_array))
M_200_samp = (np.min(M_200_array), M_200_array[mid(M_200_array)], np.max(M_200_array))

burkert_profile = np.empty([len(r_eval), 3])
NFW_profile = np.empty([len(r_eval), 3])
count = 0
for r in r_eval:
    burkert_profile[count][0] = burkert(r, r_core_samp[0], rho_central_array[0])
    burkert_profile[count][1] = burkert(r, r_core_samp[1], rho_central_array[1])
    burkert_profile[count][2] = burkert(r, r_core_samp[2], rho_central_array[2])

    NFW_profile[count][0] = NFW(r, c_param_samp[0], M_200_samp[0])
    NFW_profile[count][1] = NFW(r, c_param_samp[1], M_200_samp[1])
    NFW_profile[count][2] = NFW(r, c_param_samp[2], M_200_samp[2])

    count +=1


plotting(r_eval, burkert_profile, (-50, 10000), 'Burkert DM Density Profile')
plotting(r_eval, NFW_profile, (-0.001, 0.1), 'NFW DM Density Profile')