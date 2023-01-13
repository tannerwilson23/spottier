import dill as pkl
import matplotlib.pyplot as plt
import numpy as np


with open('norm_spectra.pkl' ,'rb') as f:
    normalised_data = pkl.load(f)

with open('norm_spectra_e.pkl' ,'rb') as f:
    normalised_data_e = pkl.load(f)

with open('injected_params.pkl' ,'rb') as f:
    params_injected = pkl.load(f)

with open('res_spotted.pkl' ,'rb') as f:
    res_spot = pkl.load(f)

with open('res_non_spotted.pkl' ,'rb') as f:
    res_non_spot = pkl.load(f)

with open('opt_spotted.pkl' ,'rb') as f:
    opt_spot = pkl.load(f)

with open('opt_non_spotted.pkl' ,'rb') as f:
    opt_non_spot = pkl.load(f)

with open('inits_spotted.pkl' ,'rb') as f:
    inits_spot = pkl.load(f)

with open('inits_non_spotted.pkl' ,'rb') as f:
    inits_non_spot = pkl.load(f)


x_spots,fspots,teffs,loggs,mehs,prots,i_degs,log10_vsinis,ages,tau_convs,masss,logls = np.array(params_injected).T
def plot_hr():
    plt.figure()
    plt.scatter(teffs,loggs,c =mehs)
    cbar = plt.colorbar()
    cbar.set_label('[Fe/H]')
    plt.xlabel('Teff (K)')
    plt.ylabel(r'logg (cm/s$^2$)')
    plt.xlim(8000,3800)
    plt.title('HR of chosen stars logg vs Teff.')
    plt.show()
    return
