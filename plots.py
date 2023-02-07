import dill as pkl
import matplotlib.pyplot as plt
import numpy as np
import os


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


with open('best_fit_spotted.pkl' ,'rb') as f:
    best_fit_spotted = pkl.load(f)

# with open('e_best_fit_spotted.pkl' ,'wb') as f:
#     e_best_fit_spotted = pkl.load(f)

with open('red_chi2_spotted.pkl' ,'rb') as f:
    red_chi2_spotted = pkl.load(f)

with open('best_fit_non_spotted.pkl' ,'rb') as f:
    best_fit_non_spotted = pkl.load(f)

# with open('e_best_fit_non_spotted.pkl' ,'wb') as f:
#     e_best_fit_non_spotted = pkl.load(f)

with open('red_chi2_non_spotted.pkl' ,'rb') as f:
    red_chi2_non_spotted = pkl.load(f)

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

para = np.array(params_injected[0:1500])
opt_spot = opt_spot[~(inits_spot.T[0]==0)]
params_injected = para[~(inits_spot.T[0]==0)]
opt_non_spot = opt_non_spot[~(inits_spot.T[0]==0)]

opt_spot_rem = opt_spot[~(inits_spot.T[0]==0)]
injected_rem = para[~(inits_spot.T[0]==0)]
opt_non_spot_rem = opt_non_spot[~(inits_spot.T[0]==0)]


x_spots,fspots,teffs,loggs,mehs,prots,i_degs,log10_vsinis,ages,tau_convs,masss,logls = np.array(params_injected).T
def plot_hr():
    plt.figure()
    plt.scatter(teffs,loggs,c =mehs)
    cbar = plt.colorbar()
    set_label('[Fe/H]')
    plt.xlabel('Teff (K)')
    plt.ylabel(r'logg (cm/s$^2$)')
    plt.xlim(8000,3800)
    plt.title('HR of chosen stars logg vs Teff.')
    plt.save_fig('hr_plot.png',dpi = 600)
    return

def hist_plot():
    binss = [[0.8,1.0],[0,0.5],[3800,8000],[4.2,4.8],[-1.5,0.5],[0,0.8]]

    label_names = [r"$x_{spot}$", r"$f_{spot}$", r"$T_{eff}$", r"$\log{g}$", "[Fe/H]", r"$v_{micro}$"]


    limit_dfe = [-0.3,0.3]
    binny = np.linspace(*limit_dfe,30)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat[0:5]):

        res_plot = ax.inset_axes([0,1.05,1,0.2])
        res_plot.set_xlim(*binss[i])
        res_plot.set_ylim(0,0.1)
        res_plot.xaxis.set_major_locator(plt.NullLocator())
        res_plot.set_ylabel(r'$\sigma \ \Delta$[Fe/H]')




        ax.scatter(np.array(params_injected).T[i]*np.ones(390),
        best_fit_spotted.T[4]-best_fit_non_spotted.T[2],
        c="k",s = 0.1)

        bins = np.linspace(*binss[i],11)
        for k in range(10):
            ins = ax.inset_axes([k/10,0,0.1,1])
            ii = (np.array(params_injected).T[i]>bins[k])*(np.array(params_injected).T[i]<bins[k+1])
            ins.hist(best_fit_spotted.T[4][ii]-best_fit_non_spotted.T[2][ii],density=True,orientation='horizontal',bins= binny,fill = False)
            ins.set_ylim(-0.3,0.3)

            ins.xaxis.set_major_locator(plt.NullLocator())
            ins.yaxis.set_major_locator(plt.NullLocator())
            ins.patch.set_alpha(0)

            res_plot.scatter(((bins[k]+bins[k+1])/2),np.std(best_fit_spotted.T[4][ii]-best_fit_non_spotted.T[2][ii]),c = 'k')

        ax.set_xlabel(f"{label_names[i]} Injected")
        ax.set_ylabel(r"$\Delta$[Fe/H]")


        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])

        ax.set_xlim(binss[i])
        ax.set_ylim(limit_dfe)
        #ax.set_title(f"$\sigma_{{opt}} = {np.std(opt_labels.T[i]):.2f}$")
        ax.plot((10000,-1000),(0,0),color = 'k',linestyle = 'dashed')

    #axes.flat[0].legend()
    fig.tight_layout()
    plt.save_fig('hist_plot.png',dpi = 600)
    return


def plot_d_fe():
    label_names = [r"$x_{spot}$", r"$f_{spot}$", r"$T_{eff}$", r"$\log{g}$", "[Fe/H]", r"$v_{sini}$"]

    limitss = [[0.75,1.05],[-0.05,0.5],[3800,8000],[4.2,4.8],[-1.5,0.5],[0,0.8]]
    limit_dfe = [-0.2,0.3]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat[0:5]):
        ax.scatter(
            np.array(params_injected).T[i]*np.ones(390),
            best_fit_spotted.T[4]-best_fit_non_spotted.T[2],
            c="k",

        )

        ax.set_xlabel(f"{label_names[i]} Injected")
        ax.set_ylabel(r"$\Delta$[Fe/H]")

        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])

        ax.set_xlim(limitss[i])
        ax.set_ylim(limit_dfe)
        ax.plot((10000,-1000),(0,0),color = 'k',linestyle = 'dashed')

    fig.tight_layout()

    fig.savefig("plot_d_fe.png", dpi=300)
    return

def plot_comparison():
    label_names = [r"$x_{spot}$", r"$f_{spot}$", r"$T_{eff}$", r"$\log{g}$", "[Fe/H]", r"$v_{micro}$"]

    limitss = [[0.75,1.05],[-0.05,0.6],[3800,8000],[4.2,4.8],[-1.5,0.5],[0,0.8]]
    limit_dfe = [-1.5,0.5]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat[0:5]):
        ax.scatter(
            np.array(params_injected).T[i]*np.ones(390),
            best_fit_non_spotted.T[2],
            c="tab:blue",
            label = 'Non-spotted Model'

        )
        ax.scatter(
            np.array(params_injected).T[i]*np.ones(390),
            best_fit_spotted.T[4],
            c="tab:orange",
            label = 'Spotted Model'

        )

        ax.set_xlabel(f"{label_names[i]} Injected")
        ax.set_ylabel(r"[Fe/H]")

        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])

        ax.set_xlim(limitss[i])
        ax.set_ylim(limit_dfe)

    axes.flat[0].legend()
    fig.tight_layout()

    fig.savefig("plot_comparison.png", dpi=300)
    return


def plot_recovery():
    label_names = [r"$x_{spot}$", r"$f_{spot}$", r"$T_{eff}$", r"$\log{g}$", "[Fe/H]", r"$v_{sini}$"]

    limitss = [[0.8,1.],[0,0.6],[3800,8000],[4.2,4.8],[-1.5,0.5],[0,2.5]]
    limit_dfe = [-1.5,0.5]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for i, ax in enumerate(axes.flat[0:6]):
        if (i <5):
            ax.scatter(
                np.array(params_injected[0:20]).T[i]*np.ones(20),
                best_fit_spotted.T[i],
                c="k",
                label = 'Spotted Model'

            )
        else:
            ax.scatter(
                np.array(params_injected[0:20]).T[7]*np.ones(20),
                best_fit_spotted.T[i],
                c="k",
                label = 'Spotted Model')

        ax.plot([-1.5,10000],[-1.5,10000],color = 'k',linestyle = 'dashed')

        ax.set_xlabel(f"{label_names[i]} Injected")
        ax.set_ylabel(f"{label_names[i]} Recovered")

        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])

        ax.set_xlim(limitss[i])
        ax.set_ylim(limitss[i])
    fig.tight_layout()
    fig.savefig("recov_tests.png", dpi=300)


def plot_mad_hr():
    plt.figure(figsize=(12,7))
    res = scipy.stats.binned_statistic_2d(injected_rem.T[2],injected_rem.T[3],opt_spot_rem.T[4]-opt_non_spot_rem.T[2],statistic=scipy.stats.median_abs_deviation,bins = [teff_bin,log_bin])
    #x,y = np.meshgrid(res[1],res[2])
    plt.pcolormesh(res[1],res[2],res[0].T)
    plt.ylim(4.8,3.9)
    plt.xlim(8000,3500)

    c = plt.colorbar()
    plt.clim(0,0.1)
    c.set_label(r'MAD $\Delta [Fe/H]$')


    plt.xlabel(r'$T_{eff}$ (K)')
    plt.ylabel(r'$\log{g}$')
    plt.savefig('hr_mad.png')
    
    
def hist_plots(which_param):

    binss = [[0.8,1.0],[0,0.5],[3800,8000],[4.2,4.8],[-1.5,0.5],[0,1.3]]

    iiii = [0,1,2,3,4,7]

    label_names = [r"$x_{spot}$", r"$f_{spot}$", r"$T_{eff}$", r"$\log{g}$", "[Fe/H]", r"$vsini$"]


    limits_d = [[-200,200],[-0.5,0.5],[-0.4,0.4],[-0.5,0.5]]

    limits_mad = [[0,100],[0,0.1],[0,0.1],[0,0.1]]
    limits_mean = [[-50,50],[-0.05,0.05],[-0.05,0.05],[-0.05,0.05]]


    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for i, ax in enumerate(axes.flat):

        res_plot = ax.inset_axes([0,1.05,1,0.2])
        
        res_plot.set_xlim(*binss[i])
        
        res_plot2 = res_plot.twinx()
        
        #res_plot.set_ylim(0,0.2)
        res_plot.xaxis.set_major_locator(plt.NullLocator())
        res_plot.set_ylabel(r"MAD $\Delta$"+ f"{label_names[which_param+2]}")
        
        res_plot2.xaxis.set_major_locator(plt.NullLocator())
        res_plot2.set_ylabel(r"$\mu$ $\Delta$"+ f"{label_names[which_param+2]}")


        res_plot.set_ylim(*limits_mad[which_param])
        res_plot2.set_ylim(*limits_mean[which_param])
        

        ax.scatter(np.array(injected_rem).T[iiii[i]]*np.ones(opt_spot_rem.shape[0]),
        opt_spot_rem.T[which_param+2]-opt_non_spot_rem.T[which_param],
        c="k",s = 0.1)
        
        ax.set_ylim(*limits_d[which_param])

        bins = np.linspace(*binss[i],11)
        for k in range(10):
            binny = np.linspace(*limits_d[which_param],30)
            ins = ax.inset_axes([k/10,0,0.1,1])
            ii = (np.array(injected_rem).T[iiii[i]]>bins[k])*(np.array(injected_rem).T[iiii[i]]<bins[k+1])
            ins.hist(opt_spot_rem.T[which_param+2][ii]-opt_non_spot_rem.T[which_param][ii],density=True,orientation='horizontal',bins= binny,fill = False)
            #ins.set_ylim(-0.4,0.4)

            ins.xaxis.set_major_locator(plt.NullLocator())
            ins.yaxis.set_major_locator(plt.NullLocator())
            ins.patch.set_alpha(0)

            res_plot.scatter(((bins[k]+bins[k+1])/2),scipy.stats.median_abs_deviation(opt_spot_rem.T[which_param+2][ii]-opt_non_spot_rem.T[which_param][ii]),c = 'k')
            res_plot2.scatter(((bins[k]+bins[k+1])/2),np.median(opt_spot_rem.T[which_param+2][ii]-opt_non_spot_rem.T[which_param][ii]),marker = 'x',c = 'k')

        ax.set_xlabel(f"{label_names[i]} Injected")
        ax.set_ylabel(r"$\Delta$"+ f"{label_names[which_param+2]}")


        limits = np.hstack([ax.get_xlim(), ax.get_ylim()])

        ax.set_xlim(binss[i])
        #ax.set_ylim(limit_dfe)
        #ax.set_title(f"$\sigma_{{opt}} = {np.std(opt_labels.T[i]):.2f}$")
        ax.plot((10000,-1000),(0,0),color = 'k',linestyle = 'dashed')

    #axes.flat[0].legend()
    fig.tight_layout()
    plt.savefig('hist_plot'+str(which_param)+'.png',dpi = 300)
    return


def plot_age_mad():
    fig, ax = plt.subplots(1,1,figsize=(12, 8))

    binss = [0,5]
    bins = np.linspace(0,5,11)

    res_plot = ax.inset_axes([0,1.05,1,0.2])

    res_plot.set_xlim(*binss)


    #res_plot.set_ylim(0,0.2)
    res_plot.xaxis.set_major_locator(plt.NullLocator())
    res_plot.set_ylabel(r"MAD $\Delta$"+ f"[Fe/H]")

    ax.scatter(np.array(injected_rem).T[8]/1e9*np.ones(opt_spot_rem.shape[0]),
    opt_spot_rem.T[4]-opt_non_spot_rem.T[2],
    c="k",s = 0.1)

    ax.set_ylim(-1,1)
    ax.set_xlim(0,5)
    ax.set_ylabel(r'$\Delta$ [Fe/H]')
    ax.set_xlabel(r'Age (Gyr)')

    res_plot.set_ylim(0,0.03)

    for k in range(10):
        #binny = np.linspace(*limits_d[which_param],30)
        ins = ax.inset_axes([k/10,0,0.1,1])
        ii = (np.array(injected_rem).T[8]/1e9>bins[k])*(np.array(injected_rem).T[8]/1e9<bins[k+1])
        #ins.hist(opt_spot_rem.T[which_param+2][ii]-opt_non_spot_rem.T[which_param][ii],density=True,orientation='horizontal',bins= binny,fill = False)
        #ins.set_ylim(-0.4,0.4)

        ins.xaxis.set_major_locator(plt.NullLocator())
        ins.yaxis.set_major_locator(plt.NullLocator())
        ins.patch.set_alpha(0)


        res_plot.scatter(((bins[k]+bins[k+1])/2),scipy.stats.median_abs_deviation(opt_spot_rem.T[4][ii]-opt_non_spot_rem.T[2][ii]),c = 'k')
        #res_plot2.scatter(((bins[k]+bins[k+1])/2),np.median(opt_spot_rem.T[which_param+2][ii]-opt_non_spot_rem.T[which_param][ii]),marker = 'x',c = 'k')
