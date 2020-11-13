import matplotlib

matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import cmocean
import numpy as np
import os
import ast
import pickle
import pandas as pd
from collections import defaultdict

from oggm import workflow, cfg, tasks, utils
from oggm.core.flowline import FileModel
from oggm.graphics import plot_centerlines

from relic.postprocessing import (mae_weighted, optimize_cov, calc_coverage,
                                  get_ensemble_length, get_rcp_ensemble_length)
from relic.preprocessing import name_plus_id, GLCDICT, MERGEDICT


def paramplots(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 7])

    allvars = ['prcp_scaling_factor', 'mbbias', 'glena_factor']
    varcols = {'mbbias': np.array([-1400, -1200, -1000, -800, -600, -400, -200,
                                   -100, 0, 100, 200, 400, 600, 800, 1000]),
               'prcp_scaling_factor': np.arange(0.5, 4.1, 0.25),
               'glena_factor': np.arange(1, 4.1, 0.5)}

    for var, ax in zip(allvars, [ax1, ax2, ax3]):
        notvars = allvars.copy()
        notvars.remove(var)

        # lets use OGGM HISTALP default
        papar = {'glena_factor': 1.0, 'mbbias': 0, 'prcp_scaling_factor': 1.75}

        # store specific runs
        dfvar = pd.DataFrame([], columns=varcols[var], index=df.index)

        # OGGM standard
        for run in df.columns:
            if run == 'obs':
                continue
            para = ast.literal_eval('{' + run + '}')

            if ((np.isclose(para[notvars[0]],
                            papar[notvars[0]], atol=0.01)) and
                    (np.isclose(para[notvars[1]],
                                papar[notvars[1]], atol=0.01))):
                dfvar.loc[:, para[var]] = df.loc[:, run]

        if var == 'prcp_scaling_factor':
            lbl = 'Precip scaling factor'

            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(
                cmocean.cm.deep))
            normalize = mcolors.Normalize(vmin=0,
                                          vmax=4.5)
            bounds = np.arange(0.375, 4.2, 0.25)
            cbarticks = np.arange(1, 4.1, 1)

        elif var == 'glena_factor':
            lbl = 'Glen A factor'

            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(
                cmocean.cm.matter))
            normalize = mcolors.Normalize(vmin=0,
                                          vmax=4.5)
            bounds = np.arange(0.75, 4.3, 0.5)
            cbarticks = np.arange(1, 4.1, 1)

        elif var == 'mbbias':
            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(
                cmocean.cm.balance))
            cmaplist = [cmap(i) for i in range(cmap.N)]
            cmaplist[128] = (0.412, 0.847, 0.655, 1.0)
            cmap = mcolors.LinearSegmentedColormap.from_list('mcm', cmaplist,
                                                             cmap.N)
            cbarticks = np.array([-1400, -1000, -600, -200,
                                  0, 200, 600, 1000])
            bounds = np.array([-1500, -1300, -1100, -900, -700, -500, -300,
                               -150, -50, 50, 100, 300, 500, 700, 900, 1100])
            normalize = mcolors.Normalize(vmin=-1600,
                                          vmax=1600)
            lbl = 'MB bias [mm w.e.]'

        colors = [cmap(normalize(n)) for n in varcols[var]]
        scalarmappaple = cm.ScalarMappable(norm=normalize, cmap=cmap)
        cbaxes = inset_axes(ax, width="3%", height="40%", loc=3)
        cbar = plt.colorbar(scalarmappaple, cax=cbaxes,
                            label=lbl,
                            boundaries=bounds)
        cbar.set_ticks(cbarticks)
        cbaxes.tick_params(axis='both', which='major', labelsize=16)
        cbar.set_label(label=lbl, size=16)

        # plot observations
        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax, color='k', style='.',
                 marker='o', label='Observed length change',
                 markersize=6)

        dfvar = dfvar.sort_index(axis=1)

        # default parameter column
        dc = np.where(dfvar.columns == papar[var])[0][0]
        dfvar.loc[:, varcols[var][dc]].rolling(y_len, center=True).mean(). \
            plot(ax=ax, color=colors[dc], linewidth=5,
                 label='{}: {} (OGGM default)'.
                 format(lbl, str(varcols[var][dc])))

        # all parameters
        nolbl = ['' for i in np.arange(len(dfvar.columns))]
        dfvar.columns = nolbl
        dfvar.rolling(y_len, center=True).mean().plot(ax=ax, color=colors,
                                                      linewidth=2)

        ax.set_xlabel('Year', fontsize=26)
        ax.set_xlim([1850, 2010])
        ax.set_ylim([-4000, 2000])
        ax.tick_params(axis='both', which='major', labelsize=22)
        if not ax == ax1:
            ax.set_yticklabels([])
        ax.grid(True)
        ax.set_xticks(np.arange(1880, 2010, 40))
        ax.legend(fontsize=16, loc=2)

    ax1.set_ylabel('relative length change [m]', fontsize=26)

    name = name_plus_id(rgi_id)
    fig1.suptitle('%s' % name, fontsize=28)
    fig1.subplots_adjust(left=0.09, right=0.99, bottom=0.12, top=0.89,
                         wspace=0.05)
    fn1 = os.path.join(pout, 'calibration_%s.png' % glid)
    fig1.savefig(fn1)


def past_simulation_and_params(glcdict, pout, y_len=5):
    for glid, df in glcdict.items():

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        fig = plt.figure(figsize=[20, 7])

        gs = GridSpec(1, 4)  # 1 rows, 4 columns

        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3])

        df.loc[:, 'obs'].plot(ax=ax1, color='k', marker='o',
                              label='Observations')

        # OGGM standard
        for run in df.columns:
            if run == 'obs':
                continue
            para = ast.literal_eval('{' + run + '}')
            if ((np.abs(para['prcp_scaling_factor'] - 1.75) < 0.01) and
                    (para['mbbias'] == 0) and
                    (para['glena_factor'] == 1)):
                df.loc[:, run].rolling(y_len, center=True). \
                    mean().plot(ax=ax1, linewidth=2, color='k',
                                label='OGGM default parameter run')
                oggmdefault = run

        maes = mae_weighted(df).sort_values()

        idx2plot = optimize_cov(df.loc[:, maes.index[:150]],
                                df.loc[:, 'obs'], glid, minuse=5)

        ensmean = df.loc[:, idx2plot].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()
        ensstdmean = df.loc[:, idx2plot].std(axis=1).rolling(y_len,
                                                              center=True).mean()

        # coverage
        cov = calc_coverage(df, idx2plot, df['obs'])

        ax1.fill_between(ensmeanmean.index, ensmeanmean - ensstdmean,
                         ensmeanmean + ensstdmean, color='xkcd:teal', alpha=0.5)

        # nolbl = df.loc[:, idx2plot2].rolling(y_len, center=True).mean().copy()
        # nolbl.columns = ['' for i in range(len(nolbl.columns))]
        #df.loc[:, idx2plot2].rolling(y_len, center=True).mean().plot(
        #    ax=ax1, linewidth=0.8)

        # plot ens members
        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='xkcd:teal',
                         label='ensemble parameters runs')

        # reference run (basically min mae)
        df.loc[:, maes.index[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='xkcd:lavender',
                 label='minimum wMAE parameter run')

        name = name_plus_id(rgi_id)

        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]

        ax1.set_title('%s' % name, fontsize=28)

        ax1.text(2030, -4900, 'wMAE ensemble mean = %.2f m\n'
                              'wMAE minimum run = %.2f m' %
                 (mae_ens, mae_best), fontsize=18,
                 horizontalalignment='right')

        ax1.text(2040, -4900, '%d ensemble members\n'
                              'coverage = %.2f' %
                 (len(idx2plot), cov), fontsize=18)

        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(-0.1, -0.15), loc='upper left',
                   fontsize=18, ncol=2)

        # parameter plots
        from colorspace import sequential_hcl
        col = sequential_hcl('Blue-Yellow').colors(len(idx2plot) + 3)
        for i, run in enumerate(idx2plot):
            para = ast.literal_eval('{' + run + '}')
            psf = para['prcp_scaling_factor']
            gla = para['glena_factor']
            mbb = para['mbbias']
            mbb = (mbb - -1400) * (4-0.5) / (1000 - -1400) + 0.5

            ax2.plot([1, 2, 3], [psf, gla, mbb], color=col[i], linewidth=2)

        ax2.set_xlabel('calibration parameters', fontsize=18)
        ax2.set_ylabel('Precipitation scaling factor\nGlen A factor',
                       fontsize=18)
        ax2.set_xlim([0.8, 3.2])
        ax2.set_ylim([0.3, 4.2])
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['Psf', 'GlenA', 'MB bias'], fontsize=16)
        ax2.tick_params(axis='y', which='major', labelsize=16)
        ax2.grid(True)

        ax3 = ax2.twinx()
        # scale to same y lims
        scale = (4.2-0.3)/(4.0-0.5)
        dy = (2400*scale-2400)/2
        ax3.set_ylim([-1400-dy, 1000+dy])
        ax3.set_ylabel('mass balance bias [m w.e. ]', fontsize=18)
        ax3.set_yticks(np.arange(-1400, 1100, 400))
        ax3.set_yticklabels(['-1.4', '-1.0', '-0.6', '-0.2',
                             '0.2', '0.6', '1.0'])
        ax3.tick_params(axis='both', which='major', labelsize=16)

        fig.subplots_adjust(left=0.08, right=0.95, bottom=0.24, top=0.93,
                            wspace=0.5)

        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig.savefig(fn1)

        used = dict()
        used['oggmdefault'] = oggmdefault
        used['minmae'] = idx2plot[0]
        used['ensemble'] = idx2plot

        pickle.dump(used, open(os.path.join(pout, 'runs_%s.p' % glid), 'wb'))


def past_simulation_and_commitment(rgi, allobs, allmeta, histalp_storage,
                                   comit_storage, comit_storage_noseed,
                                   pout, y_len=5, comyears=300):

    cols = ['xkcd:teal',
            'xkcd:orange',
            'xkcd:azure',
            'xkcd:tomato',
            'xkcd:blue',
            'xkcd:chartreuse',
            'xkcd:green'
            ]

    obs = allobs.loc[rgi.split('_')[0]]
    meta = allmeta.loc[rgi.split('_')[0]]

    fn99 = 'model_diagnostics_commitment1999_{:02d}.nc'
    df99 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn99, meta)
    fn85 = 'model_diagnostics_commitment1885_{:02d}.nc'
    df85 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn85, meta)
    fn70 = 'model_diagnostics_commitment1970_{:02d}.nc'
    df70 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn70, meta)

    # plot
    fig, ax1 = plt.subplots(1, figsize=[20, 7])

    obs.plot(ax=ax1, color='k', marker='o',
             label='Observations')

    # past
    ensmean = df99.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df99.std(axis=1).rolling(y_len, center=True).mean()

    ax1.fill_between(ensmeanmean.loc[:2015].index,
                     ensmeanmean.loc[:2015] - ensstdmean.loc[:2015],
                     ensmeanmean.loc[:2015] + ensstdmean.loc[:2015],
                     color=cols[0], alpha=0.5)

    ensmeanmean.loc[:2015].plot(ax=ax1, linewidth=4.0, color=cols[0],
                                label='HISTALP climate')

    # dummy
    ax1.plot(0, 0, 'w-', label=' ')

    # 1999
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color=cols[1], alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color=cols[1],
                                label='Random climate (1984-2014)')

    # 1970
    ensmean = df70.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df70.std(axis=1).rolling(y_len, center=True).mean()
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color=cols[5], alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color=cols[5],
                                label='Random climate (1960-1980)')

    # 1885
    ensmean = df85.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df85.std(axis=1).rolling(y_len, center=True).mean()
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color=cols[2], alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color=cols[2],
                                label='Random climate (1870-1900)')

    # ---------------------------------------------------------------------
    # plot commitment ensemble length
    # 1984
    efn99 = 'model_diagnostics_commitment1999_{:02d}.nc'
    edf99 = get_ensemble_length(rgi, histalp_storage, comit_storage_noseed,
                                efn99, meta)
    ensmean = edf99.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = edf99.std(axis=1).rolling(y_len, center=True).mean()
    postlength = ensmeanmean.dropna().iloc[-30:].mean()
    poststd = ensstdmean.dropna().iloc[-30:].mean()

    ax1.fill_between([2014+comyears+10, 2014+comyears+25],
                     postlength + poststd, postlength - poststd,
                     color=cols[3], alpha=0.5)
    ax1.plot([2014+comyears+10.5, 2014+comyears+24.5], [postlength, postlength], linewidth=4.0,
             color=cols[3],
             label=('Random climate (1984-2014) '
                    'equlibrium length'))

    # 1970
    efn70 = 'model_diagnostics_commitment1970_{:02d}.nc'
    edf70 = get_ensemble_length(rgi, histalp_storage, comit_storage_noseed,
                                efn70, meta)
    ensmean = edf70.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = edf70.std(axis=1).rolling(y_len, center=True).mean()
    prelength = ensmeanmean.dropna().iloc[-30:].mean()
    prestd = ensstdmean.dropna().iloc[-30:].mean()
    ax1.fill_between([2014+comyears+10, 2014+comyears+25],
                     prelength + prestd, prelength - prestd,
                     color=cols[6], alpha=0.5)
    ax1.plot([2014+comyears+10.5, 2014+comyears+24.5], [prelength, prelength],
             linewidth=4.0,
             color=cols[6],
             label=('Random climate (1960-1980) '
                    'equlibrium length'))

    # 1885
    efn85 = 'model_diagnostics_commitment1885_{:02d}.nc'
    edf85 = get_ensemble_length(rgi, histalp_storage, comit_storage_noseed,
                                efn85, meta)
    ensmean = edf85.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = edf85.std(axis=1).rolling(y_len, center=True).mean()
    prelength = ensmeanmean.dropna().iloc[-30:].mean()
    prestd = ensstdmean.dropna().iloc[-30:].mean()
    ax1.fill_between([2014+comyears+10, 2014+comyears+25],
                     prelength + prestd, prelength - prestd,
                     color=cols[4], alpha=0.5)
    ax1.plot([2014+comyears+10.5, 2014+comyears+24.5], [prelength, prelength],
             linewidth=4.0,
             color=cols[4],
             label=('Random climate (1870-1900) '
                    'equlibrium length'))

    # ---------------------------------------------------------------------

    ylim = ax1.get_ylim()
    #ax1.plot([2015, 2015], ylim, 'k-', linewidth=2)
    ax1.set_xlim([1850, 2014+comyears+30])
    #ax1.set_ylim(ylim)

    ax2 = ax1.twinx()
    ax2.set_ylabel('approximate\n absolute glacier length [m]', fontsize=26)
    y1, y2 = get_absolute_length(ylim[0], ylim[1], rgi, df99, histalp_storage)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.set_ylim([y1, y2])

    name = name_plus_id(rgi)
    ax1.set_title('%s' % name, fontsize=28)

    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)

    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.set_xticks([1850, 1950, 2014, 2114, 2214, 2314])
    ax1.set_xticklabels(['1850', '1950', '2014/0', '100', '200', '300'])
    ax1.grid(True)

    ax1.legend(bbox_to_anchor=(-0.0, -0.17), loc='upper left', fontsize=18,
               ncol=3)

    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.3, top=0.93,
                        wspace=0.5)
    fn1 = os.path.join(pout, 'commit_%s.png' % rgi)
    fig.savefig(fn1)


def past_simulation_and_projection(rgi, allobs, allmeta, histalp_storage,
                                   proj_storage, comit_storage,
                                   pout, y_len=5,):

    cols = ['xkcd:teal',
            'xkcd:azure',
            'xkcd:lime',
            'xkcd:orange',
            'xkcd:magenta',
            'xkcd:tomato',
            'xkcd:blue',
            'xkcd:green'
            ]

    obs = allobs.loc[rgi.split('_')[0]]
    meta = allmeta.loc[rgi.split('_')[0]]

    dfall = pd.DataFrame([], index=np.arange(1850, 2101))
    dfallstd = pd.DataFrame([], index=np.arange(1850, 2101))

    for rcp in ['rcp26', 'rcp45', 'rcp60', 'rcp85']:

        dfrcp = get_rcp_ensemble_length(rgi, histalp_storage, proj_storage,
                                        rcp, meta)

        ensmean = dfrcp.mean(axis=1)
        dfall.loc[:, rcp] = ensmean.rolling(y_len, center=True).mean()
        dfallstd.loc[:, rcp] = dfrcp.std(axis=1).\
            rolling(y_len, center=True).mean()

    # plot
    fig, ax1 = plt.subplots(1, figsize=[20, 7])

    obs.plot(ax=ax1, color='k', marker='o',
             label='Observations')

    # past
    ax1.fill_between(dfall.loc[:2015, rcp].index,
                     dfall.loc[:2015, rcp] - dfallstd.loc[:2015, rcp],
                     dfall.loc[:2015, rcp] + dfallstd.loc[:2015, rcp],
                     color=cols[0], alpha=0.5)
    dfall.loc[:2015, rcp].plot(ax=ax1, linewidth=4.0, color=cols[0],
                               label='HISTALP climate')

    # dummy
    ax1.plot(0, 0, 'w-', label=' ')

    # projections
    # rcp26
    ax1.fill_between(dfall.loc[2015:, 'rcp26'].index,
                     dfall.loc[2015:, 'rcp26'] - dfallstd.loc[2015:, 'rcp26'],
                     dfall.loc[2015:, 'rcp26'] + dfallstd.loc[2015:, 'rcp26'],
                     color=cols[1], alpha=0.5)
    dfall.loc[2015:, 'rcp26'].plot(ax=ax1, linewidth=4.0, color=cols[1],
                                   label='RCP 2.6 climate')

    # rcp45
    dfall.loc[2015:, 'rcp45'].plot(ax=ax1, linewidth=4.0, color=cols[2],
                                   label='RCP 4.5 climate')
    # dummy
    ax1.plot(0, 0, 'w-', label=' ')

    # rcp60
    dfall.loc[2015:, 'rcp60'].plot(ax=ax1, linewidth=4.0, color=cols[3],
                                   label='RCP 6.0 climate')

    # rcp85
    ax1.fill_between(dfall.loc[2015:, 'rcp85'].index,
                     dfall.loc[2015:, 'rcp85'] - dfallstd.loc[2015:, 'rcp85'],
                     dfall.loc[2015:, 'rcp85'] + dfallstd.loc[2015:, 'rcp85'],
                     color=cols[4], alpha=0.5)
    dfall.loc[2015:, 'rcp85'].plot(ax=ax1, linewidth=4.0, color=cols[4],
                                   label='RCP 8.5 climate')

    # dummy
    ax1.plot(0, 0, 'w-', label=' ')

    # plot commitment length
    # 1984
    fn99 = 'model_diagnostics_commitment1999_{:02d}.nc'
    df99 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn99, meta)
    ensmean = df99.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df99.std(axis=1).rolling(y_len, center=True).mean()
    postlength = ensmeanmean.dropna().iloc[-30:].mean()
    poststd = ensstdmean.dropna().iloc[-30:].mean()

    ax1.fill_between([2105, 2111],
                     postlength + poststd, postlength - poststd,
                     color=cols[5], alpha=0.5)
    ax1.plot([2105.5, 2110.5], [postlength, postlength], linewidth=4.0,
             color=cols[5],
             label=('Random climate (1984-2014) '
                    'equilibrium length'))

    # 1970
    fn70 = 'model_diagnostics_commitment1970_{:02d}.nc'
    df70 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn70, meta)
    ensmean = df70.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df70.std(axis=1).rolling(y_len, center=True).mean()
    prelength = ensmeanmean.dropna().iloc[-30:].mean()
    prestd = ensstdmean.dropna().iloc[-30:].mean()
    ax1.fill_between([2105, 2111],
                     prelength + prestd, prelength - prestd,
                     color=cols[7], alpha=0.5)
    ax1.plot([2105.5, 2110.5], [prelength, prelength], linewidth=4.0,
             color=cols[7],
             label=('Random climate (1960-1980) '
                    'equilibrium length'))

    # 1885
    fn85 = 'model_diagnostics_commitment1885_{:02d}.nc'
    df85 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn85, meta)
    ensmean = df85.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df85.std(axis=1).rolling(y_len, center=True).mean()
    prelength = ensmeanmean.dropna().iloc[-30:].mean()
    prestd = ensstdmean.dropna().iloc[-30:].mean()
    ax1.fill_between([2105, 2111],
                     prelength + prestd, prelength - prestd,
                     color=cols[6], alpha=0.5)
    ax1.plot([2105.5, 2110.5], [prelength, prelength], linewidth=4.0,
             color=cols[6],
             label=('Random climate (1870-1900) '
                    'equilibrium length'))

    ylim = ax1.get_ylim()
    ax1.set_xlim([1850, 2112])

    ax2 = ax1.twinx()
    ax2.set_ylabel('apporixmate\n absolute glacier length [m]', fontsize=26)
    y1, y2 = get_absolute_length(ylim[0], ylim[1], rgi, df99, histalp_storage)
    ax2.tick_params(axis='both', which='major', labelsize=22)
    ax2.set_ylim([y1, y2])

    name = name_plus_id(rgi)
    ax1.set_title('%s' % name, fontsize=28)

    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)

    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True)

    ax1.legend(bbox_to_anchor=(0.0, -0.17), loc='upper left', fontsize=18,
               ncol=4)

    fig.subplots_adjust(left=0.09, right=0.9, bottom=0.3, top=0.93,
                        wspace=0.5)

    fn1 = os.path.join(pout, 'proj_%s.png' % rgi)
    fig.savefig(fn1)


def get_mean_temps_eq(rgi, histalp_storage, comit_storage, ensmembers):
    from oggm import cfg, utils, GlacierDirectory
    from oggm.core.massbalance import MultipleFlowlineMassBalance
    from oggm.core.flowline import FileModel
    import shutil


    # 1. get mean surface heights
    df85 = pd.DataFrame([])
    df99 = pd.DataFrame([])
    for i in range(ensmembers):
        fnc1 = os.path.join(comit_storage, rgi,
                            'model_run_commitment1885_{:02d}.nc'.format(i))
        fnc2 = os.path.join(comit_storage, rgi,
                            'model_run_commitment1999_{:02d}.nc'.format(i))
        tmpmod1 = FileModel(fnc1)
        tmpmod2 = FileModel(fnc2)
        for j in np.arange(270, 301):
            tmpmod1.run_until(j)
            df85.loc[:, '{}{}'.format(i, j)] = tmpmod1.fls[-1].surface_h
            tmpmod2.run_until(j)
            df99.loc[:, '{}{}'.format(i, j)] = tmpmod2.fls[-1].surface_h

    meanhgt99 = df99.mean(axis=1).values
    meanhgt85 = df85.mean(axis=1).values

    # 2. get the climate
    # Initialize OGGM
    cfg.initialize()
    wd = utils.gettempdir(reset=True)
    cfg.PATHS['working_dir'] = wd
    utils.mkdir(wd, reset=True)
    cfg.PARAMS['baseline_climate'] = 'HISTALP'
    # and set standard histalp values
    cfg.PARAMS['temp_melt'] = -1.75

    i = 0
    storage_dir = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)
    new_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier',
                           rgi[:8], rgi[:11], rgi)
    shutil.copytree(storage_dir, new_dir)
    gdir = GlacierDirectory(rgi)
    mb = MultipleFlowlineMassBalance(gdir, filename='climate_monthly',
                                     check_calib_params=False)
    # need to do the above for every ensemble member if I consider PRECIP!
    # and set cfg.PARAMS['prcp_scaling_factor'] = pdict['prcp_scaling_factor']

    df99_2 = pd.DataFrame()
    df85_2 = pd.DataFrame()
    for i in np.arange(9, 12):
        for y in np.arange(1870, 1901):
            flyear = utils.date_to_floatyear(y, i)
            tmp = mb.flowline_mb_models[-1].get_monthly_climate(meanhgt85,
                                                                flyear)[0]
            df85_2.loc[y, i] = tmp.mean()
        for y in np.arange(1984, 2015):
            tmp = mb.flowline_mb_models[-1].get_monthly_climate(meanhgt99,
                                                                flyear)[0]
            df99_2.loc[y, i] = tmp.mean()

    t99 = df99_2.mean().mean()
    t85 = df85_2.mean().mean()
    return t85, t99


def get_mean_temps_2k(rgi, return_prcp):
    from oggm import cfg, utils, workflow, tasks
    from oggm.core.massbalance import PastMassBalance

    # Initialize OGGM
    cfg.initialize()
    wd = utils.gettempdir(reset=True)
    cfg.PATHS['working_dir'] = wd
    utils.mkdir(wd, reset=True)
    cfg.PARAMS['baseline_climate'] = 'HISTALP'
    # and set standard histalp values
    cfg.PARAMS['temp_melt'] = -1.75
    cfg.PARAMS['prcp_scaling_factor'] = 1.75

    gdir = workflow.init_glacier_regions(rgidf=rgi.split('_')[0],
                                         from_prepro_level=3,
                                         prepro_border=10)[0]
    # run histalp climate on glacier!
    tasks.process_histalp_data(gdir)

    f = gdir.get_filepath('climate_historical')
    with utils.ncDataset(f) as nc:
        refhgt = nc.ref_hgt

    mb = PastMassBalance(gdir, check_calib_params=False)

    df = pd.DataFrame()
    df2 = pd.DataFrame()

    for y in np.arange(1870, 2015):
        for i in np.arange(9, 12):
            flyear = utils.date_to_floatyear(y, i)
            tmp = mb.get_monthly_climate([refhgt], flyear)[0]
            df.loc[y, i] = tmp.mean()

        if return_prcp:
            for i in np.arange(3, 6):
                flyear = utils.date_to_floatyear(y, i)
                pcp = mb.get_monthly_climate([refhgt], flyear)[3]
                df2.loc[y, i] = tmp.mean()

    t99 = df.loc[1984:2014, :].mean().mean()
    t85 = df.loc[1870:1900, :].mean().mean()
    t2k = df.loc[1900:2000, :].mean().mean()

    if return_prcp:
        p99 = df2.loc[1984:2014, :].mean().mean()
        p85 = df2.loc[1870:1900, :].mean().mean()
        p2k = df2.loc[1900:2000, :].mean().mean()
        return t85, t99, t2k, p85, p99, p2k

    return t85, t99, t2k


def get_absolute_length(y0, y1, rgi, df, storage):
    rgipath = os.path.join(storage, rgi, '{:02d}'.format(0),
                           rgi[:8], rgi[:11], rgi)
    mfile = os.path.join(rgipath, 'model_run_histalp_{:02d}.nc'.format(0))
    tmpmod = FileModel(mfile)
    absL = tmpmod.length_m
    deltaL = df.loc[int(tmpmod.yr.values), 0]

    abs_y0 = absL + (y0 - deltaL)
    abs_y1 = absL + (y1 - deltaL)

    return abs_y0, abs_y1


def elevation_profiles(rgi, meta, histalp_storage, pout):

    name = name_plus_id(rgi)

    df1850 = pd.DataFrame()
    df2003 = pd.DataFrame()
    df2003b = pd.DataFrame()
    dfbed = pd.DataFrame()

    for i in np.arange(999):

        # Local working directory (where OGGM will write its output)
        rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)
        fn = os.path.join(rgipath, 'model_run_histalp_{:02d}.nc'.format(i))
        try:
            tmpmod = FileModel(fn)
        except FileNotFoundError:
            break

        df1850.loc[:, i] = tmpmod.fls[-1].surface_h

        # get bed surface
        dfbed.loc[:, i] = tmpmod.fls[-1].bed_h

        # HISTALP surface
        tmpmod.run_until(2003)
        df2003.loc[:, i] = tmpmod.fls[-1].surface_h
        df2003b.loc[:, i] = tmpmod.fls[-1].thick

    # RGI init surface, once is enough
    fn2 = os.path.join(histalp_storage, rgi, '00', rgi[:8], rgi[:11],
                       rgi, 'model_run_spinup_00.nc')
    tmpmod2 = FileModel(fn2)
    initsfc = tmpmod2.fls[-1].surface_h

    # get distance on line
    dx_meter = tmpmod.fls[-1].dx_meter

    meanbed = dfbed.mean(axis=1).values
    maxbed = dfbed.max(axis=1).values
    minbed = dfbed.min(axis=1).values

    # 1850
    mean1850 = df1850.mean(axis=1).values
    # where is mean glacier thinner than 1m
    ix50 = np.where(mean1850-meanbed < 1)[0][0]
    mean1850[ix50:] = np.nan

    min1850 = df1850.min(axis=1).values
    min1850[ix50:] = np.nan
    min1850[min1850 <= meanbed] = meanbed[min1850 <= meanbed]

    max1850 = df1850.max(axis=1).values
    max1850[max1850 <= meanbed] = meanbed[max1850 <= meanbed]

    # 2003
    mean2003 = df2003.mean(axis=1).values
    # where is mean glacier thinner than 1m
    ix03 = np.where(mean2003-meanbed < 1)[0][0]
    mean2003[ix03:] = np.nan

    min2003 = df2003.min(axis=1).values
    min2003[ix03:] = np.nan
    min2003[min2003 <= meanbed] = meanbed[min2003 <= meanbed]

    max2003 = df2003.max(axis=1).values
    max2003[max2003 <= meanbed] = meanbed[max2003 <= meanbed]

    lastx = np.where(initsfc-meanbed < 1)[0][0]
    initsfc[lastx:] = np.nan
    initsfc[lastx] = meanbed[lastx]

    dis = np.arange(len(meanbed)) * dx_meter / 1000
    xmax = sum(np.isfinite(mean1850))
    ymax = np.nanmax(mean1850) + 50
    ymin = minbed[np.where(np.isfinite(mean1850))].min() - 50

    fig, ax = plt.subplots(1, figsize=[15, 9])

    ax.fill_between(dis[:xmax+1], dis[:xmax+1] * 0 + ymin, minbed[:xmax+1],
                    color='0.7', alpha=0.5)
    ax.fill_between(dis[:xmax+1], minbed[:xmax+1], maxbed[:xmax+1],
                    color='xkcd:tan', alpha=0.5)
    ax.plot(dis[:xmax+1], meanbed[:xmax+1], 'k-', color='xkcd:tan',
            linewidth=3, label='Glacier bed elevation [m]')

    ax.fill_between(dis, min1850, max1850, color='xkcd:azure', alpha=0.5)
    ax.plot(dis, mean1850, 'k-', color='xkcd:azure', linewidth=4,
            label=('Surface elevation [m] year {:d}\n'
                   '(initialization state after spinup)'.
                   format(meta['first'])))

    ax.fill_between(dis, min2003, max2003, color='xkcd:teal', alpha=0.5)
    ax.plot(dis, mean2003, 'k-', color='xkcd:teal', linewidth=4,
            label=('Surface elevation [m] year 2003\n'
                   '(from HISTALP ensemble simulations)'))

    ax.plot(dis, initsfc, 'k-', color='xkcd:crimson', linewidth=4,
            label=('Surface elevation [m] year 2003\n'
                   '(from RGI initialization)'))

    ax.legend(loc=1, fontsize=20)

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0, dis[xmax])
    ax.set_xlabel('Distance along major flowline [km]', fontsize=28)
    ax.set_ylabel('Elevation [m a.s.l.]', fontsize=28)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.grid(True)
    ax.set_title(name, fontsize=30)
    fig.tight_layout()
    fn = os.path.join(pout, 'profile_%s' % rgi)
    if ('3643' in rgi) or ('1450' in rgi) or ('2051' in rgi) or ('897' in rgi):
        fig.savefig('{}.svg'.format(fn))
    fig.savefig('{}.png'.format(fn))


def grey_madness(glcdict, pout, y_len=5):
    for glid, df in glcdict.items():

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        fig, ax1 = plt.subplots(figsize=[20, 7])

        # OGGM standard
        for run in df.columns:
            if run == 'obs':
                continue
            para = ast.literal_eval('{' + run + '}')
            if ((np.abs(para['prcp_scaling_factor'] - 1.75) < 0.01) and
                    (para['mbbias'] == 0) and
                    (para['glena_factor'] == 1)):
                oggmdefault = run
                break

        nolbl = df.loc[:, df.columns != 'obs'].\
            rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]
        nolbl.plot(ax=ax1, linewidth=0.8, color='0.7')

        df.loc[:, oggmdefault].rolling(y_len, center=True).mean().plot(
            ax=ax1, linewidth=0.8, color='0.7',
            label='Every possible calibration parameter combination')

        df.loc[:, oggmdefault].rolling(y_len, center=True).mean().\
            plot(ax=ax1, color='k', linewidth=2,
                 label='OGGM default parameters')

        df.loc[:, 'obs'].plot(ax=ax1, color='k', marker='o',
                              label='Observations')

        name = name_plus_id(rgi_id)

        ax1.set_title('%s' % name, fontsize=28)

        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2014])
        ax1.set_ylim([-7500, 4000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(-0.0, -0.15), loc='upper left',
                   fontsize=18, ncol=2)

        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.24, top=0.93,
                            wspace=0.5)

        fn1 = os.path.join(pout, 'all_%s.png' % glid)
        fig.savefig(fn1)


def run_and_plot_merged_montmine(pout):
    # Set-up
    cfg.initialize(logging_level='WORKFLOW')
    cfg.PATHS['working_dir'] = utils.gettempdir(dirname='OGGM-merging',
                                                reset=True)
    # Use a suitable border size for your domain
    cfg.PARAMS['border'] = 80
    cfg.PARAMS['use_intersects'] = False

    montmine = workflow.init_glacier_directories(['RGI60-11.02709'],
                                                 from_prepro_level=3)[0]

    gdirs = workflow.init_glacier_directories(['RGI60-11.02709',
                                               'RGI60-11.02715'],
                                              from_prepro_level=3)
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)
    gdirs_merged = workflow.merge_glacier_tasks(gdirs, 'RGI60-11.02709',
                                                return_all=False,
                                                filename='climate_monthly',
                                                buffer=2.5)

    # plot centerlines
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 10])
    plot_centerlines(montmine, ax=ax1, use_flowlines=True)

    xt = ax1.get_xticks()
    ax1.set_xticks(xt[::2])
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax1.set_title('entity glacier', fontsize=24)

    plot_centerlines(gdirs_merged, ax=ax2, use_model_flowlines=True)
    ax2.tick_params(axis='both', which='major', labelsize=20)
    ax2.set_title('merged with Glacier de Ferpecle', fontsize=24)

    axs = fig.get_axes()
    axs[3].remove()
    axs[2].tick_params(axis='y', labelsize=16)
    axs[2].set_ylabel('Altitude [m]', fontsize=18)

    fig.suptitle('Glacier du Mont Mine', fontsize=24)
    fig.subplots_adjust(left=0.04, right=0.99, bottom=0.08, top=0.89,
                        wspace=0.3)

    fn = os.path.join(pout, 'merged_montmine.png')
    fig.savefig(fn)

    # run glaciers with negative t bias
    # some model settings
    years = 125
    tbias = -1.5

    # model Mont Mine glacier as entity and complile the output
    tasks.run_constant_climate(montmine, nyears=years,
                               output_filesuffix='_entity',
                               temperature_bias=tbias)
    ds_entity = utils.compile_run_output([montmine], path=False,
                                         filesuffix='_entity')

    # model the merged glacier and complile the output
    tasks.run_constant_climate(gdirs_merged, nyears=years,
                               output_filesuffix='_merged',
                               temperature_bias=tbias,
                               climate_filename='climate_monthly')
    ds_merged = utils.compile_run_output([gdirs_merged], path=False,
                                         filesuffix='_merged')

    #
    # bring them to same size again
    tbias = -2.2
    years = 125

    tasks.run_constant_climate(montmine, nyears=years,
                               output_filesuffix='_entity1',
                               temperature_bias=tbias)
    ds_entity1 = utils.compile_run_output([montmine], path=False,
                                          filesuffix='_entity1')

    # and let them shrink again
    # some model settings
    tbias = -0.5
    years = 100

    # load the previous entity run
    tmp_mine = FileModel(
        montmine.get_filepath('model_run', filesuffix='_entity1'))
    tmp_mine.run_until(years)

    tasks.run_constant_climate(montmine, nyears=years,
                               output_filesuffix='_entity2',
                               init_model_fls=tmp_mine.fls,
                               temperature_bias=tbias)
    ds_entity2 = utils.compile_run_output([montmine], path=False,
                                          filesuffix='_entity2')

    # model the merged glacier and complile the output
    tmp_merged = FileModel(
        gdirs_merged.get_filepath('model_run', filesuffix='_merged'))
    tmp_merged.run_until(years)

    tasks.run_constant_climate(gdirs_merged, nyears=years,
                               output_filesuffix='_merged2',
                               init_model_fls=tmp_merged.fls,
                               temperature_bias=tbias,
                               climate_filename='climate_monthly')
    ds_merged2 = utils.compile_run_output([gdirs_merged], path=False,
                                          filesuffix='_merged2')

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[20, 7])

    dse = ds_entity.length.to_series().rolling(5, center=True).mean()
    dsm = ds_merged.length.to_series().rolling(5, center=True).mean()
    ax1.plot(dse.values, 'C1', label='Entity glacier', linewidth=3)
    ax1.plot(dsm.values, 'C2', label='Merged glacier', linewidth=3)
    ax1.set_xlabel('Simulation time [yr]', fontsize=20)
    ax1.set_ylabel('Glacier length[m]', fontsize=20)
    ax1.grid(True)
    ax1.legend(loc=2, fontsize=18)

    dse2 = ds_entity2.length.to_series().rolling(5, center=True).mean()
    dsm2 = ds_merged2.length.to_series().rolling(5, center=True).mean()
    ax2.plot(dse2.values, 'C1', label='Entity glacier', linewidth=3)
    ax2.plot(dsm2.values, 'C2', label='Merged glacier', linewidth=3)
    ax2.set_xlabel('Simulation time [yr]', fontsize=22)
    ax2.set_ylabel('Glacier length [m]', fontsize=22)
    ax2.grid(True)
    ax2.legend(loc=1, fontsize=18)

    ax1.set_xlim([0, 120])
    ax2.set_xlim([0, 100])
    ax1.set_ylim([7500, 12000])
    ax2.set_ylim([7500, 12000])
    ax1.tick_params(axis='both', which='major', labelsize=20)
    ax2.tick_params(axis='both', which='major', labelsize=20)

    fig.subplots_adjust(left=0.08, right=0.96, bottom=0.11, top=0.93,
                        wspace=0.3)

    fn = os.path.join(pout, 'merged_montmine_timeseries.png')
    fig.savefig(fn)


def climate_vs_lengthchange(dfout, pout):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=[20, 15])

    ost = dfout.loc[dfout['lon'] >= 9.5]
    west = dfout.loc[dfout['lon'] < 9.5]

    # ax1: temp, winter
    ost.plot.scatter(x='dl 1885-1970', y='dt win', color='C1',
                     ax=ax1, s=80, label='Temp. Oct-Apr (East)')
    ost.plot.scatter(x='dl 1885-1970', y='dt djf', color='C3',
                     ax=ax1, s=80, label='Temp. DJF (East)')
    west.plot.scatter(x='dl 1885-1970', y='dt win', color='C2', marker='s',
                      ax=ax1, s=80, label='Temp. Oct-Apr (West)')
    west.plot.scatter(x='dl 1885-1970', y='dt djf', color='C4', marker='s',
                      ax=ax1, s=80, label='Temp. DJF (West)')

    # ax2: temp, sommer
    ost.plot.scatter(x='dl 1885-1970', y='dt som', color='C1',
                     ax=ax2, s=80, label='Temp. Mai-Sep (East)')
    ost.plot.scatter(x='dl 1885-1970', y='dt jja', color='C3',
                     ax=ax2, s=80, label='Temp. JJA (East)')
    west.plot.scatter(x='dl 1885-1970', y='dt som', color='C2', marker='s',
                      ax=ax2, s=80, label='Temp. Mai-Sep (West)')
    west.plot.scatter(x='dl 1885-1970', y='dt jja', color='C4', marker='s',
                      ax=ax2, s=80, label='Temp. JJA (West)')

    # ax3: pcp, winter
    west.plot.scatter(x='dl 1885-1970', y='dp win', color='C2', marker='s',
                      ax=ax3, s=80, label='Prcp. Oct-Apr (West)')
    west.plot.scatter(x='dl 1885-1970', y='dp djf', color='C4', marker='s',
                      ax=ax3, s=80, label='Prcp. DJF (West)')
    ost.plot.scatter(x='dl 1885-1970', y='dp win', color='C1',
                     ax=ax3, s=80, label='Prcp. Oct-Apr (East)')
    ost.plot.scatter(x='dl 1885-1970', y='dp djf', color='C3',
                     ax=ax3, s=80, label='Prcp. DJF (East)')

    # ax4: pcp, sommer
    west.plot.scatter(x='dl 1885-1970', y='dp jja', color='C4', marker='s',
                      ax=ax4, s=80, label='Prcp. JJA (West)')
    west.plot.scatter(x='dl 1885-1970', y='dp som', color='C2', marker='s',
                      ax=ax4, s=80, label='Prcp. Mai-Sep (West)')
    ost.plot.scatter(x='dl 1885-1970', y='dp jja', color='C3',
                     ax=ax4, s=80, label='Prcp. JJA (East)')
    ost.plot.scatter(x='dl 1885-1970', y='dp som', color='C1',
                     ax=ax4, s=80, label='Prcp. Mai-Sep (East)')

    ax4.set_xlabel(('Equilibrium length difference\nbetween 1870-1900 '
                    'and 1960-1980 climate'), fontsize=20)
    ax3.set_xlabel(('Equilibrium length difference\nbetween 1870-1900 '
                    'and 1960-1980 climate'), fontsize=20)
    ax1.set_ylabel(('Temperature difference between\n 1870-1900 and '
                    '1960-1980 climate'), fontsize=20)
    ax3.set_ylabel(('Precipitation difference between\n 1870-1900 and '
                    '1960-1980 climate'), fontsize=20)
    ax2.set_ylabel('')
    ax4.set_ylabel('')
    ax1.set_xlabel('')
    ax2.set_xlabel('')

    ax1.set_ylim([-1.0, 0.2])
    ax2.set_ylim([-1.0, 0.2])

    ax3.set_ylim([-350, 50])
    ax4.set_ylim([-350, 50])

    for ax in [ax1, ax2, ax3, ax4]:

        ax.grid(True)
        ax.legend(loc=3, ncol=2, fontsize=18)

        ax.set_xlim([-4, 2])

        ax.tick_params(axis='both', which='major', labelsize=20)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.93,
                        wspace=0.2, hspace=0.2)

    fig.savefig(os.path.join(pout, 'climate_vs_length.png'))


def histogram(pin, pout):
    glena = defaultdict(int)
    mbbias = defaultdict(int)
    prcpsf = defaultdict(int)

    for glc in GLCDICT.keys():
        glid = str(glc)
        if MERGEDICT.get(glc):
            glid += '_merged'
        rundictpath = os.path.join(pin, 'runs_%s.p' % glid)
        rundict = pickle.load(open(rundictpath, 'rb'))
        ens = rundict['ensemble']
        for run in ens:
            para = ast.literal_eval('{' + run + '}')
            prcpsf[para['prcp_scaling_factor']] += 1
            glena[para['glena_factor']] += 1
            mbbias[para['mbbias']] += 1

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[20, 7])
    ax1.bar(list(glena.keys()), glena.values(), width=0.4)
    ax1.set_xlabel('Glen A factor', fontsize=22)
    ax1.set_ylabel('# used in ensemble', fontsize=22)

    ax2.bar(list(prcpsf.keys()), prcpsf.values(), width=0.2)
    ax2.set_xlabel('Prcp SF factor', fontsize=22)
    ax2.set_ylabel('# used in ensemble', fontsize=22)

    ax3.bar(list(mbbias.keys()), mbbias.values(), width=150)
    ax3.set_xlabel('MB bias', fontsize=22)
    ax3.set_ylabel('# used in ensemble', fontsize=22)

    for ax in [ax1, ax2, ax3]:
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.grid(True)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.11, top=0.93,
                        wspace=0.2, hspace=0.2)

    fig.savefig(os.path.join(pout, 'histo.png'))
