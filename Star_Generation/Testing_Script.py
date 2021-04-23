import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from dSph_Model import NormalVelocityDistribution, chabrier_imf, GenerateForeground
import scipy.stats.distributions as st
import scipy as sci
import seaborn as sns

def cyl2cart(v):
    """
    Assumes array of shape (r, theta, z)
    """
    r = v[0]
    theta = v[1]
    z = v[2]
    return np.array([r*np.cos(theta), r*np.sin(theta), z])

if __name__ == "__main__":
    """PARAMETERS"""
    name = "sample_foreground_star_masspos"
    label_kwargs = {'size':18, 
                    'weight':'bold'}
    title_kwargs = {'size':24,
                    'weight':'bold'}

    # name = "sample_foreground_star_masspos"
    label_kwargs = {'size':18}
    title_kwargs = {'size':24}
    df = pd.read_excel(io=name+'.xls')
    starframe = df.loc[:, df.columns[1:]]

    """Generating Data"""
    dSph_galaxy_model = GenerateForeground()
    dSph_galaxy_model.starCone(name=name)         
    # """Check the data"""
    starxy = pd.DataFrame(columns=['x','y','z','m','component','v_theta'])
    for i in range(len(starframe.index)):
        cylco = np.array(starframe.loc[i, ['r','theta','z']])
        xyzco = cyl2cart(cylco)
        starxy = starxy.append({'x':xyzco[0],'y':xyzco[1],'z':cylco[2],'m':starframe.loc[i,'m'],'component':starframe.loc[i,'component'],'v_theta':starframe.loc[i,'v_theta']}, ignore_index=True)
    
    """Checking the Co-ordinate Conversion"""
    data = pd.read_excel(io=name+'.xls')

    """Plotting Velocity Dispersion"""
    # mbvd = NormalVelocityDistribution()
    # v = np.linspace(0, 300, 301)
    # fig = plt.figure()
    # plt.plot(v, mbvd._pdf(v, 150, 100))
    # plt.show()
    # plt.pause(1)
    

    """Line of Sight Stars"""
    fig = plt.figure(figsize=(10,10));
    sns.set_style('white')
    sns.scatterplot(data=starxy, x='x',y='y', size='m', hue='v_theta');
    plt.xlabel('x [pc]', **label_kwargs);
    plt.ylabel('y [pc]', **label_kwargs);
    plt.title('Distribution of Generated Stars Along \nLine of Sight Axis', **title_kwargs);
    matplotlib.rc('font', family='Serif') 
    matplotlib.rc('font', serif='Times') 
    plt.xscale('symlog')
    plt.yscale('symlog')
    # plt.xlim([np.min(starxy.loc[:,'x']), np.max(starxy.loc[:,'y'])])
    plt.show()

    """Conical Distribution"""
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111, projection='3d')
    x = starxy['x']
    y = starxy['y']
    z = starxy['z']
    gca = ax.scatter(xs=x,ys=y,zs=np.log10(z),cmap='hot',c=starxy['v_theta'])
    plt.xlabel('x [pc]', **label_kwargs)
    plt.ylabel('y [pc]', **label_kwargs)
    ax.set_zlabel('log$_{10}$(z) [log$_{10}$(pc)]', **label_kwargs)
    plt.title('Distribution of Stars in Heliocentric \nCartesian Coordinates', **title_kwargs)
    m = matplotlib.cm.ScalarMappable(norm=gca.norm, cmap='hot')
    m.set_array(starxy['v_theta'])
    cb = plt.colorbar(m, fraction=0.05, shrink=0.5)
    cb.ax.set_xlabel(r'$v_\theta$ [km/s]', **label_kwargs)
    plt.show()

    """Chabrier_IMF"""
    imf = chabrier_imf()
    m_array = np.array([])
    sizeRVS = 10000
    for i in range(sizeRVS):
        m_array= np.hstack((m_array, imf._rvs()))
    m_sample = np.logspace(np.log10(imf.__minval__), np.log10(imf.__maxval__), sizeRVS)
    logbins = np.logspace(np.log10(imf.a), np.log10(imf.b), 50) 
    fig, ax = plt.subplots(figsize=(10,10));
    plt.hist(m_array, density=True, bins=logbins);
    plt.plot(m_sample, imf._pdf(m_sample));
    plt.xlabel('m [M$_\odot$]', **label_kwargs);
    plt.ylabel('Normalized Count', **label_kwargs);
    plt.title('Random Variate Distribution Along the \nChabrier IMF for $10^{%d}$ Stars' % np.log10(sizeRVS), **title_kwargs);
    plt.xscale('log');
    plt.yscale('log');
    plt.xlim([imf.__minval__, imf.__maxval__]);
    plt.legend(['Chabrier IMF PDF','RVS Generation for $10^{%d}$ Stars' % np.log10(sizeRVS)])
    plt.show()

    """Velocity Distribution"""
    disp = 50
    mu_b = 10
    mu_d = 20
    gvb = NormalVelocityDistribution()
    vb = np.array([])
    vd = np.array([])
    for i in range(sizeRVS):
        vb = np.hstack((vb, gvb._rvs(mu_b, disp)))
        vd = np.hstack((vd, gvb._rvs(mu_d, disp)))
    vb_samp = np.linspace(np.min(vb), np.max(vb), sizeRVS)
    vd_samp = np.linspace(np.min(vd), np.max(vd), sizeRVS)
    pdf_b = gvb._pdf(vb_samp, mu_b, disp)
    pdf_d = gvb._pdf(vd_samp, mu_d, disp)
    fig = plt.figure(figsize=(10,10))
    plt.hist(vb, bins=sizeRVS//500, density=True, alpha=0.5)
    plt.plot(vb_samp, pdf_b)
    plt.hist(vd, bins=sizeRVS//500, density=True, alpha=0.5)
    plt.plot(vd_samp, pdf_d)
    plt.title('Random Variate Sampling of Orbital Velocities', **title_kwargs)
    plt.xlabel(r'Orbital Velocity, $v_\theta$ [km/s]', **label_kwargs)
    plt.ylabel('Normalized Count', **label_kwargs)
    plt.legend(['Gaussian Probability Distribution of Bulge Stars',
                'Gaussian Probability Distribution of Disk Stars',
                r'Normalized Distribution of $10^{%d}$ Bulge Stars' % np.log10(sizeRVS),
                r'Normalized Distribution of $10^{%d}$ Disk Stars' % np.log10(sizeRVS)])
    plt.show()