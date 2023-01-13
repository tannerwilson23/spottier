import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sci
from scipy import stats
import math
import random as random
from isochrones.mist import MIST_EvolutionTrack
from astropy.io import fits
from scipy import interpolate
import dill as pkl
import bottleneck
import os.path
from tqdm import tqdm
import scipy.optimize as op
np.random.seed(80085)

#the number of spectra you create will be smaller than this number. fails for some, just make sure you make enough to run the number you want
size =100
wl = 10**(6e-6*np.arange(8575)+4.179)

def sampleFromSalpeter(N, alpha, M_min, M_max):
    # Convert limits from M to logM.
    log_M_Min = math.log(M_min)
    log_M_Max = math.log(M_max)
    # Since Salpeter SMF decays, maximum likelihood occurs at M_min
    maxlik = math.pow(M_min, 1.0 - alpha)

    # Prepare array for output masses.
    Masses = []
    # Fill in array.
    while (len(Masses) < N):
        # Draw candidate from logM interval.
        logM = random.uniform(log_M_Min,log_M_Max)
        M    = math.exp(logM)
        # Compute likelihood of candidate from Salpeter SMF.
        likelihood = math.pow(M, 1.0 - alpha)
        # Accept randomly.
        u = random.uniform(0.0,maxlik)
        if (u < likelihood):
            Masses.append(M)
    return Masses


def sampleMet(scale):
    alpha, beta = (10, 2)
    min_metallicity, max_metallicity = (-2, 0.5)

    z = stats.beta.rvs(alpha, beta, size=size)

    mode = (alpha - 1)/(alpha + beta - 2)
    scale = max_metallicity - min_metallicity

    fe_h_draws = scale * (z - mode)
    return fe_h_draws


def sampleMIST(scale,mass_draws,eep_draws,fe_h_draws):
    models = []
    mist_track = MIST_EvolutionTrack()

    for i in range(scale):
        pars = [mass_draws[i], eep_draws[i], fe_h_draws[i]]
        #print(mass,age,feh)
        models.append(mist_track.interp_value(pars,['mass','Teff','logg','feh','logL', 'radius', 'star_age'])) # "accurate=True" makes more accurate, but slower
    return models



def sampleFromModels(models,plot=False):
    chosen = []
    j = 0
    for i in range(size):
        if (models[i][-1]<4.5e9):
            chosen.append(models[i])
            j=j+1
    if (plot == True):
        plt.xlim(8000,3000)
        plt.xlabel('Teff (K)')
        plt.ylabel('Log L')
    return chosen

#import rossby data and create interpolator
ros = fits.open('rosby_data.fits')
ros_ros = ros[1].data
ros_ros_trans = np.zeros([149,3])
for i in range(149):
    ros_ros_trans[i] = np.append(np.array(ros_ros[i])[0:2],np.array(ros_ros[i])[-2])

interpolator_ros = interpolate.LinearNDInterpolator(
    ros_ros_trans[:,0:2],
    ros_ros_trans[:,-1],
    rescale=False
)
#A = model["coefficients"].T
def interpolate_conv_turn_number(mass,log_10age):
    return (interpolator_ros([mass, log_10age]))[0]

#interpolating rotating grid of isochrones
spada_isos = pd.read_csv('spada_rot_isochrones.csv')
spada_isos = np.array(spada_isos)
spada_ages = spada_isos[0,2:]
spada_masses = spada_isos.T[0,1:]
spada_rots = spada_isos[1:,2:]
spada_func = sci.interpolate.interp2d(spada_ages,spada_masses,spada_rots)

def spada_val(age,mass):
    spada = spada_func(age,mass)
    return spada[0]


def tamb_from_teff(teff,fspot,xspot):
    tamb = teff*((1-fspot) + fspot*(xspot**4))**(-1/4)
    return tamb
def tspot_from_tamb(tamb,xspot):
    tspot = tamb*xspot
    return tspot


def approximate_log10_microturbulence(log_g):
    """
    Approximate the log10(microturbulent velocity) given the surface gravity.
    :param log_g:
        The surface gravity.
    :returns:
        The log base-10 of the microturbulent velocity, vt: log_10(vt).
    """

    coeffs = np.array([0.372160, -0.090531, -0.000802, 0.001263, -0.027321])
    # I checked with Holtz on this microturbulence relation because last term is not used.
    DM = np.array([1, log_g, log_g**2, log_g**3, 0])
    return DM @ coeffs

#this will either check whether the interpolator grid is in your current directory or will create the interpolator and pickle it. saves ~20mins
interp_pkl = "interp_grid.pkl"
if os.path.isfile(interp_pkl):
    with open("interp_grid.pkl" ,'rb') as f:
        interpolator = pkl.load(f)
else:
    with open("convolved_grid_with_vsini.pkl", "rb") as fp:
        model = pkl.load(fp)

    interpolator = interpolate.LinearNDInterpolator(
        model["parameter_values"],
        model["node_flux"],
        rescale=False
    )
    with open(interp_pkl ,'wb') as f:
        pkl.dump(interpolator,f)


def interpolate_apogee_spectrum(teff, logg, microturbulence, m_h,log10_vsini):
    return (interpolator([teff, logg, microturbulence, m_h, log10_vsini]))[0]

def generate_spec(wl,xspot,fspot,teff,logg,meh,log10_vsini):
    micro = 10**approximate_log10_microturbulence(logg)
    tamb = tamb_from_teff(teff,fspot,xspot)
    tspot = tspot_from_tamb(tamb,xspot)
    amb_spec = interpolate_apogee_spectrum(tamb,logg,micro,meh,log10_vsini)
    spotted_spec = interpolate_apogee_spectrum(tspot,logg,micro,meh,log10_vsini)
    #Eqn 2 in Gao +  Pinsonneault 2022
    spectra = fspot * spotted_spec + (1-fspot)*amb_spec
    spectra = spectra/bottleneck.move_mean(spectra,30)
    spectra[0:29] = 1
    return spectra

def generate_spec_non_spot(wl,teff,logg,meh,log10_vsini):
    micro = 10**approximate_log10_microturbulence(logg)
    spectra = interpolate_apogee_spectrum(teff,logg,micro,meh,log10_vsini)
    spectra = spectra/bottleneck.move_mean(spectra,30)
    spectra[0:29] = 1
    return spectra

def rossby_from_prot(prot,mass,age):
    log10_age = np.log10(age)
    tau_conv = interpolate_conv_turn_number(mass,log10_age)

    ross = prot/tau_conv
    return ross,tau_conv

def fspot_from_ross(rossby):
    log_ross = np.log10(rossby)
    if log_ross<-0.729:
        fspot = 0.278
    else:
        fspot = 6.34e-2 * rossby**(-0.881)
    return fspot

def create_synth_spec():
    raw_data_true =[]
    normalised_data_true = []
    normalised_data = []
    normalised_data_e = []
    params_injected = []
    #raw_err = []

    P = 8575

    wl = 10**(6e-6*np.arange(8575)+4.179)
    mass_draws = sampleFromSalpeter(size,2.35,0.5,1.5)
    fe_h_draws = sampleMet(size)

    eep_list = np.linspace(200,450,2000)
    eep_draws = np.random.choice(eep_list,size)
    models  = sampleMIST(size,mass_draws,eep_draws,fe_h_draws)
    chosen = sampleFromModels(models)
    pr_xspot =np.linspace(0.8,1,10000)
    x_spot_draws = np.random.choice(pr_xspot,len(chosen))
    cos_i_pr = np.linspace(0,1,1000)


    for i in range(len(chosen)):
        curr_chose = chosen[i]
        teff,logg,meh,logl = curr_chose[1],curr_chose[2],curr_chose[3],curr_chose[4]
        mass = curr_chose[0]
        age = curr_chose[-1]
        prot  = spada_val(age/1e9,mass)

        v = (2*np.pi*curr_chose[-2]*695700)/(prot*60*60*24)
        i_rad = np.arccos(np.random.choice(cos_i_pr))
        i_deg = 180/np.pi*i_rad
        sini = np.sin(i_rad)

        vsini = v*sini

        log10_vsini = np.log10(vsini)


        rossby,tau_conv = rossby_from_prot(prot,mass,age)
        fspot = fspot_from_ross(rossby) + np.random.normal(0,0.1)

        data = generate_spec(wl,x_spot_draws[i],fspot,teff,logg,meh,log10_vsini)
        if (np.isnan(data[-1])==False):
            raw_data_true.append(data)
            params_injected.append([x_spot_draws[i],fspot,teff,logg,meh,prot,i_deg,log10_vsini,age,tau_conv,mass,logl])
        #normalising
            norm_data_true = data

            normalised_data_true.append(norm_data_true)
            P = 8575
            e = 0.01* np.random.rand(P)

            normalised_data_e.append(e)
            normalised_data.append(norm_data_true+ e*np.random.randn(P))

    #dump the spectra if you don't want to make them all the time.
    with open('norm_spectra.pkl' ,'wb') as f:
        pkl.dump(normalised_data,f)

    with open('norm_spectra_e.pkl' ,'wb') as f:
        pkl.dump(normalised_data_e,f)

    with open('injected_params.pkl.pkl' ,'wb') as f:
        pkl.dump(params_injected,f)

    print("Generated " + str(len(normalised_data)[0]) + ' spectra')
    return normalised_data,normalised_data_e ,params_injected



def fit_speccy(i,normalised_data,normalised_data_e ,params_injected):
    res = None
    while res is None:
        # try:
        teff_init = params_injected[i][2]+np.random.normal(0,0.01)
        logg_init = params_injected[i][3]+np.random.normal(0,0.001)
        feh_init = params_injected[i][4]+np.random.normal(0,0.001)
        log10_vsini_init = params_injected[i][7]+np.random.normal(0,0.001)
        f_spot_init = params_injected[i][1]+np.random.normal(0,0.001)
        x_spot_init = params_injected[i][0]+np.random.normal(0,0.001)
        p_init = [x_spot_init,f_spot_init,teff_init,logg_init,feh_init,log10_vsini_init]
        res = op.curve_fit(
                f=generate_spec,
                xdata=wl,
                ydata=normalised_data[i],
                p0=  p_init,
                bounds=(
                    [0.8, 0.,2000,  3, -2,0.01],
                    [1,   0.5,8000, 5,  2,1.3]
                ),
                sigma=normalised_data_e[i],xtol = None, full_output=True
                )
        # except:
        #     pass

    return res,p_init


def fit_speccy_non_spot(i,normalised_data,normalised_data_e ,params_injected):
    res = None
    while res is None:
        # try:
        teff_init = params_injected[i][2]+np.random.normal(0,0.01)
        logg_init = params_injected[i][3]+np.random.normal(0,0.001)
        feh_init = params_injected[i][4]+np.random.normal(0,0.001)
        log10_vsini_init = params_injected[i][7]+np.random.normal(0,0.001)
        p_init = [teff_init,logg_init,feh_init,log10_vsini_init]
        res = op.curve_fit(
                f=generate_spec_non_spot,
                xdata=wl,
                ydata=normalised_data[i],
                p0=  p_init,
                bounds=(
                    [2000,  3, -2,0.01],
                    [8000, 5,  2,1.3]
                ),
                sigma=normalised_data_e[i],xtol = None, full_output=True
                )
        # except:
        #     pass

    return res,p_init


label_names = ["xspot", "fspot", "teff", "logg", "feh","log10_vsini"]
L = len(label_names)
label_names_non_spot = ["teff", "logg", "feh","log10_vsini"]
L_n = len(label_names_non_spot)


starts = 2
E = 2

#when full_output = false
if (starts ==1):
    opt_non_spot = np.nan * np.ones((E, L_n))
    inits_non_spot = np.nan * np.ones((E, L_n))
    opt_spot = np.nan * np.ones((E, L))
    inits_spot = np.nan * np.ones((E, L))
else:
    opt_non_spot = np.nan * np.ones((starts,E, L_n))
    inits_non_spot = np.nan * np.ones((starts,E, L_n))
    opt_spot = np.nan * np.ones((starts,E, L))
    inits_spot = np.nan * np.ones((starts,E, L))


#when full_output = true
def run_fitting(normalised_data,normalised_data_e ,params_injected):
    res_non_spot = []
    res_spot = []

    for i in tqdm(range(E)):
        if (starts == 1):

            fitted_data  = fit_speccy(i,normalised_data,normalised_data_e ,params_injected)
            fitted_data_non_spot = fit_speccy_non_spot(i,normalised_data,normalised_data_e ,params_injected)
            res_spot.append(fitted_data)
            res_non_spot.append(fitted_data_non_spot)

            opt_spot[i] = fitted_data[0][0]
            opt_non_spot[i] = fitted_data_non_spot[0][0]

            inits_non_spot[i] = fitted_data_non_spot[1]
            inits_spot[i] = fitted_data[1]

        else:
            starts_non_spot = []
            starts_spot = []
            for j in range(starts):

                fitted_data  = fit_speccy(i,normalised_data,normalised_data_e ,params_injected)
                fitted_data_non_spot = fit_speccy_non_spot(i,normalised_data,normalised_data_e ,params_injected)


                starts_spot.append(fitted_data)
                starts_non_spot.append(fitted_data_non_spot)

                opt_spot[j,i] = fitted_data[0][0]
                opt_non_spot[j,i] = fitted_data_non_spot[0][0]

                inits_non_spot[j,i] = np.array(fitted_data_non_spot[1])
                inits_spot[j,i] = np.array(fitted_data[1])
            res_spot.append(starts_spot)
            res_non_spot.append(starts_non_spot)

    with open('res_spotted.pkl' ,'wb') as f:
        pkl.dump(res_spot,f)

    with open('res_non_spotted.pkl' ,'wb') as f:
        pkl.dump(res_non_spot,f)

    with open('opt_spotted.pkl' ,'wb') as f:
        pkl.dump(np.array(opt_spot),f)

    with open('opt_non_spotted.pkl' ,'wb') as f:
        pkl.dump(np.array(opt_non_spot),f)

    with open('inits_spotted.pkl' ,'wb') as f:
        pkl.dump(np.array(inits_spot),f)

    with open('inits_non_spotted.pkl' ,'wb') as f:
        pkl.dump(np.array(inits_non_spot),f)


    return res_spot,res_non_spot,inits_spot,inits_non_spot,opt_spot,opt_non_spot

if __name__ == "__main__":
    normalised_data,normalised_data_e,params_injected = create_synth_spec()
    res_spot,res_non_spot,inits_spot,inits_non_spot,opt_spot,opt_non_spot = run_fitting(normalised_data,normalised_data_e ,params_injected)
