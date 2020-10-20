import matplotlib

matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import ast
import pickle
import pandas as pd
import xarray as xr

from oggm.core.flowline import FileModel

from relic.postprocessing import (mae_weighted, optimize_cov, calc_coverage,
                                  relative_length_change)
from relic.preprocessing import GLCDICT


def paramplots(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[25, 5])

    allvars = ['prcp_scaling_factor', 'mbbias', 'glena_factor']

    for var, ax in zip(allvars, [ax1, ax2, ax3]):
        notvars = allvars.copy()
        notvars.remove(var)

        # lets use OGGM HISTALP default
        papar = {'glena_factor': 1.0, 'mbbias': 0, 'prcp_scaling_factor': 1.75}

        # store specific runs
        varcols = {'mbbias': [-1400, -1200, -1000, -800, -600, -400, -200,
                              -100, 0, 100, 200, 400, 600, 800, 1000],
                   'prcp_scaling_factor': np.arange(0.5, 4.1, 0.25),
                   'glena_factor': np.arange(1, 4.1, 0.5)}

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

        import cmocean
        from matplotlib.colors import LinearSegmentedColormap

        if var == 'prcp_scaling_factor':

            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(cmocean.cm.deep))
            colors = [cmap(x) for x in np.linspace(0,1,21)][2:-4]

            lbl = 'Precip SF: '
        elif var == 'glena_factor':
            lbl = 'Glen A: '

            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(
                cmocean.cm.matter))
            colors = [cmap(x) for x in np.linspace(0, 1, 7)]

        elif var == 'mbbias':
            cmap = LinearSegmentedColormap('lala', cmocean.tools.get_dict(
                cmocean.cm.balance))
            colors = [cmap(x) for x in np.linspace(0, 0.5, 11)][2:10]
            colors.append((0.5, 0.5, 0.5, 1.0))
            colors = colors + [cmap(x) for x in np.linspace(0.5, 1, 11)][1:8]

            lbl = 'MB bias [mm w.e.]: '

        # plot observations
        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax, color='k', style='.',
                 marker='o', label='Observed length change',
                 markersize=6)

        dfvar = dfvar.sort_index(axis=1)

        # first parameter
        if var != 'glena_factor':
            dfvar.loc[:, varcols[var][0]].rolling(y_len, center=True).mean().\
                plot(ax=ax, color=colors[0], linewidth=2,
                     label='{}{}'.format(lbl, str(varcols[var][0])))

            ax.plot(0, 0, '|k', label=' ')

        # default parameter column
        dc = np.where(dfvar.columns == papar[var])[0][0]
        dfvar.loc[:, varcols[var][dc]].rolling(y_len, center=True).mean(). \
            plot(ax=ax, color=colors[dc], linewidth=5,
                 label='{}{} (OGGM default)'.format(lbl,
                                                    str(varcols[var][dc])))

        # dummy
        ax.plot(0, 0, '|k', label=' ')

        # last parameter
        dfvar.loc[:, varcols[var][-1]].rolling(y_len, center=True).mean(). \
            plot(ax=ax, color=colors[-1], linewidth=2,
                 label='{}{}'.format(lbl, str(varcols[var][-1])))

        # all pparameters
        nolbl = ['' for i in np.arange(len(dfvar.columns))]
        dfvar.columns = nolbl
        dfvar.rolling(y_len, center=True).mean().plot(ax=ax, color=colors,
                                                      linewidth=2)

        ax.set_xlabel('Year', fontsize=18)
        ax.set_xlim([1850, 2020])
        ax.set_ylim([-4000, 1000])
        ax.tick_params(axis='both', which='major', labelsize=16)
        if not ax == ax1:
            ax.set_yticklabels([])
        ax.grid(True)

        ax.legend(fontsize=12, loc=3)

    # ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_ylabel('relative length change [m]', fontsize=18)

    name = GLCDICT.get(rgi_id)[2]
    fig1.suptitle('%s' % name, fontsize=30)
    fig1.tight_layout(rect=[0, 0.0, 0.99, 0.94])
    fn1 = os.path.join(pout, '%s.pdf' % glid)
    fn1 = os.path.join(pout, '%s.png' % glid)
    fig1.savefig(fn1)


def past_simulation_and_params(glcdict, tribdict, pout, y_len=5):
    for glid, df in glcdict.items():

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        # if rgi_id != 'RGI60-11.03643':
        #    continue

        # if (rgi_id != 'RGI60-11.01450') and (rgi_id != 'RGI60-11.02051') and (rgi_id != 'RGI60-11.01270') and (rgi_id != 'RGI60-11.03643') and (rgi_id != 'RGI60-11.00897'):
        #    #if (rgi_id != 'RGI60-11.02755') and (rgi_id != 'RGI60-11.03646'):
        #    continue

        fig = plt.figure(figsize=[23, 8])

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
                                label='HISTALP climate (OGGM default parameters run)')
                oggmdefault = run

        # remove runs with to long tributaries:
        if ('_merged' in glid) and ('897' in glid):
            trib = tribdict[glid]
            from relic.preprocessing import merge_pair_dict
            y0 = merge_pair_dict(rgi_id)[2]
            toolong = (trib.loc[y0+20:] == 0).any()
            df.drop(toolong.index[toolong], axis=1, inplace=True)
            trib.drop(toolong.index[toolong], axis=1, inplace=True)
            tooshort = (trib.loc[:y0-20] < 0).any()
            df.drop(tooshort.index[tooshort], axis=1, inplace=True)
            print('{}: removed {} runs'.format(glid,
                                               (len(toolong.index[toolong]) +
                                                len(tooshort.index[tooshort])))
                  )
        else:
            continue

        maes = mae_weighted(df).sort_values()

        idx2plot2 = optimize_cov(df.loc[:, maes.index[:150]],
                                 df.loc[:, 'obs'], glid, minuse=5)
        # idx2plot2 = optimize_cov2(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5)

        ensmean = df.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()
        ensstdmean = df.loc[:, idx2plot2].std(axis=1).rolling(y_len,
                                                              center=True).mean()

        # coverage
        cov = calc_coverage(df, idx2plot2, df['obs'])

        #ax1.fill_between(ensmeanmean.index, ensmeanmean - ensstdmean,
        #                 ensmeanmean + ensstdmean, color='xkcd:teal', alpha=0.5)

        # nolbl = df.loc[:, idx2plot2].rolling(y_len, center=True).mean().copy()
        # nolbl.columns = ['' for i in range(len(nolbl.columns))]
        df.loc[:, idx2plot2].rolling(y_len, center=True).mean().plot(
            ax=ax1, linewidth=0.8)

        #ax1.plot(0, 0, color='C0', linewidth=10,
        #         label='ensemble mean +/- 1 std', alpha=0.5)

        # plot ens members
        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='xkcd:teal',
                         label='HISTALP climate (ensemble parameters runs)')

        # reference run (basically min mae)
        df.loc[:, maes.index[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='xkcd:lavender',
                 label='HISTALP climate (minimum wMAE parameter run)')

        name = GLCDICT.get(rgi_id)[2]

        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]

        ax1.set_title('%s' % name, fontsize=28)
        ax1.text(1990, -4700, '%d ensemble members  '
                              '    coverage = %.2f\n'
                              'wMAE enselbe = %.2f  '
                              '  wMAE best = %.2f' %
                 (len(idx2plot2), cov, mae_ens,
                  mae_best), fontsize=18)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(0.0, -0.175), loc='upper left', fontsize=14,
                   ncol=2)

        # parameter plots
        from colorspace import sequential_hcl
        #col = sequential_hcl('Blues 3').colors(len(idx2plot2) + 3)
        col = sequential_hcl('Blue-Yellow').colors(len(idx2plot2) + 3)
        for i, run in enumerate(idx2plot2):
            para = ast.literal_eval('{' + run + '}')
            psf = para['prcp_scaling_factor']
            #psf = (psf - 0.5) / (4 - 0.5)
            gla = para['glena_factor']
            #gla = (gla - 1) / (4 - 1.0)
            mbb = para['mbbias']
            # mbb = (mbb - -1400) / (1000 - -1400)
            # scall mbb to range 0.5-4
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

        fig.subplots_adjust(left=0.07, right=0.96, bottom=0.24, top=0.93,
                            wspace=0.5)
        #fig.tight_layout(h_pad=0.0)

        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig.savefig(fn1)

        used = dict()
        # used['oggmdefault'] = oggmdefault
        used['minmae'] = idx2plot2[0]
        used['ensemble'] = idx2plot2

        pickle.dump(used, open(os.path.join(pout, 'runs_%s.p' % glid), 'wb'))


def past_simulation_and_commitment(rgi, allobs, allmeta, histalp_storage,
                                   comit_storage, comit_storage_noseed,
                                   pout, y_len=5, comyears=300):

    cols = ['xkcd:teal',
            'xkcd:orange',
            'xkcd:azure',
            'xkcd:tomato',
            'xkcd:blue'
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
    fig, ax1 = plt.subplots(1, figsize=[23, 10])

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

    #ax1.plot(0, 0, color='C0', linewidth=10,
    #         label='ensemble mean +/- 1 std', alpha=0.5)
    ensmeanmean.loc[:2015].plot(ax=ax1, linewidth=4.0, color=cols[0],
                                label='HISTALP climate')

    # 1999
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color=cols[1], alpha=0.5)

    #ax1.plot(0, 0, color='C1', linewidth=10,
    #         label='ensemble mean +/- 1 std (1999)', alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color=cols[1],
                                label='Random climate (1984-2014)')

    # 1885
    ensmean = df85.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df85.std(axis=1).rolling(y_len, center=True).mean()
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color=cols[2], alpha=0.5)

    #ax1.plot(0, 0, color='C9', linewidth=10,
    #         label='ensemble mean +/- 1 std (1885)', alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color=cols[2],
                                label='Random climate (1870-1900)')

    # 1970
    ensmean = df70.mean(axis=1)
    ensmeanmean = ensmean.rolling(y_len, center=True).mean()
    ensstdmean = df70.std(axis=1).rolling(y_len, center=True).mean()
    ax1.fill_between(ensmeanmean.loc[2015:].index,
                     ensmeanmean.loc[2015:] - ensstdmean.loc[2015:],
                     ensmeanmean.loc[2015:] + ensstdmean.loc[2015:],
                     color='xkcd:chartreuse', alpha=0.5)

    # ax1.plot(0, 0, color='C9', linewidth=10,
    #         label='ensemble mean +/- 1 std (1885)', alpha=0.5)
    ensmeanmean.loc[2015:].plot(ax=ax1, linewidth=4.0, color='xkcd:chartreuse',
                                label='Random climate (1960-1980)')

    """
    # climate temperatures
    ensmembers = i
    
    t85, t99 = get_mean_temps_eq(rgi, histalp_storage, comit_storage, ensmembers)
    ax1.text(2330, postlength.mean(),
             'tmean = {:.2f}'.format(t99))
    ax1.text(2330, (prelength.mean() + postlength.mean()) / 2,
             'dT = {:.2f}'.format(t99-t85))
    ax1.text(2330, prelength.mean(),
             'tmean = {:.2f})'.format(t85))
    
    t85a, t99a, t2k = get_mean_temps_2k(rgi)
    ax1.text(2330, postlength.mean(),
             'dT = {:.2f} ({:.2f})'.format(t99a-t2k, t99a))
    ax1.text(2330, prelength.mean(),
             'dT = {:.2f} ({:.2f})'.format(t85a-t2k, t85a))

    ax1.text(2330, (prelength.mean()+postlength.mean())/2,
                   't2k = {:.2f}'.format(t2k))
    """
    # ---------------------------------------------------------------------
    # plot commitment ensemble length
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
    #ax1.plot(0, 0, color='C1', linewidth=10,
    #         label='random climate 1984-2014 mean +/- 1 std', alpha=0.5)
    ax1.plot([2014+comyears+10.5, 2014+comyears+24.5], [postlength, postlength], linewidth=4.0,
             color=cols[3],
             label='Random climate (1984-2014) equlibrium length from multiple seeds')

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
    #ax1.plot(0, 0, color='C9', linewidth=10,
    #         label='random climate 1870-1900 +/- 1 std', alpha=0.5)
    ax1.plot([2014+comyears+10.5, 2014+comyears+24.5], [prelength, prelength],
             linewidth=4.0,
             color=cols[4],
             label='Random climate (1870-1900) equlibrium length from multiple seeds')
    # ---------------------------------------------------------------------

    ylim = ax1.get_ylim()
    #ax1.plot([2015, 2015], ylim, 'k-', linewidth=2)
    ax1.set_xlim([1850, 2014+comyears+30])
    #ax1.set_ylim(ylim)

    ax2 = ax1.twinx()
    ax2.set_ylabel('approximate absolute glacier length [m]', fontsize=22)
    y1, y2 = get_absolute_length(ylim[0], ylim[1], rgi, df99, histalp_storage)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_ylim([y1, y2])

    name = GLCDICT.get(rgi.split('_')[0])[2]
    ax1.set_title('%s' % name, fontsize=28)

    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)

    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.set_xticks([1900, 2000, 2114, 2214, 2314])
    ax1.set_xticklabels(['1900', '2000', '100', '200', '300'])
    ax1.grid(True)

    ax1.legend(bbox_to_anchor=(-0.0, -0.1), loc='upper left', fontsize=14,
               ncol=3)

    fig.tight_layout()
    fn1 = os.path.join(pout, 'commit_%s.png' % rgi)
    fig.savefig(fn1)


def get_ensemble_length(rgi, histalp_storage, future_storage,
                        ensemble_filename, meta):

    df = pd.DataFrame([], index=np.arange(1850, 3000))

    for i in np.arange(999):

        rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)

        try:
            sp = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_spinup_{:02d}.nc'.format(i)))
            hi = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_histalp_{:02d}.nc'.format(i)))
        except FileNotFoundError:
            break

        sp = sp.length_m.to_dataframe()['length_m']
        hi = hi.length_m.to_dataframe()['length_m']
        df.loc[:, i] = relative_length_change(meta, sp, hi)

    ensemble_count = i
    # future
    for i in np.arange(ensemble_count):

        fut = xr.open_dataset(os.path.join(future_storage, rgi,
                                           ensemble_filename.format(i)))

        fut = fut.length_m.to_dataframe()['length_m']
        fut.index = fut.index + 2014
        df.loc[2015:, i] = (fut - fut.iloc[0] + df.loc[2014, i]).loc[2015:]

    return df


def get_rcp_ensemble_length(rgi, histalp_storage, future_storage,
                            rcp, meta):
    cmip = ['CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'CanESM2',
            'GFDL-CM3', 'GFDL-ESM2G', 'GISS-E2-R', 'IPSL-CM5A-LR',
            'MPI-ESM-LR', 'NorESM1-M']

    dfrcp = pd.DataFrame([], index=np.arange(1850, 2101))

    for i in np.arange(999):

        rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)

        try:
            sp = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_spinup_{:02d}.nc'.format(i)))
            hi = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_histalp_{:02d}.nc'.format(i)))
        except FileNotFoundError:
            break

        sp = sp.length_m.to_dataframe()['length_m']
        hi = hi.length_m.to_dataframe()['length_m']
        dfrcp.loc[:, i] = relative_length_change(meta, sp, hi)

    nr_ensemblemembers = i

    # projection
    for i in np.arange(nr_ensemblemembers):

        df_cm = pd.DataFrame()
        for cmi in cmip:
            try:
                cm = xr.open_dataset(
                    os.path.join(future_storage, rgi,
                                 'model_diagnostics_{}_{}_{:02d}.nc'.
                                 format(cmi, rcp, i)))
            except FileNotFoundError:
                continue

            df_cm.loc[:, cmi] = cm.length_m.to_dataframe()['length_m']

        cm = df_cm.mean(axis=1)
        dfrcp.loc[2015:, i] = \
            (cm - cm.iloc[0] + dfrcp.loc[2014, i]).loc[2015:]

    return dfrcp


def past_simulation_and_projection(rgi, allobs, allmeta, histalp_storage,
                                   proj_storage, comit_storage,
                                   pout, y_len=5,):

    cols = ['xkcd:teal',
            'xkcd:azure',
            'xkcd:green',
            'xkcd:orange',
            'xkcd:magenta',
            'xkcd:tomato',
            'xkcd:blue'
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
    fig, ax1 = plt.subplots(1, figsize=[23, 10])

    obs.plot(ax=ax1, color='k', marker='o',
             label='Observations')

    # past
    ax1.fill_between(dfall.loc[:2015, rcp].index,
                     dfall.loc[:2015, rcp] - dfallstd.loc[:2015, rcp],
                     dfall.loc[:2015, rcp] + dfallstd.loc[:2015, rcp],
                     color=cols[0], alpha=0.5)
    #ax1.plot(0, 0, color='C0', linewidth=10,
    #         label='ensemble mean +/- 1 std', alpha=0.5)
    dfall.loc[:2015, rcp].plot(ax=ax1, linewidth=4.0, color=cols[0],
                               label='HISTALP climate')

    # projections
    # rcp26
    ax1.fill_between(dfall.loc[2015:, 'rcp26'].index,
                     dfall.loc[2015:, 'rcp26'] - dfallstd.loc[2015:, 'rcp26'],
                     dfall.loc[2015:, 'rcp26'] + dfallstd.loc[2015:, 'rcp26'],
                     color=cols[1], alpha=0.5)
    #ax1.plot(0, 0, color='C2', linewidth=10,
    #         label='rcp26 ensemble mean +/- 1 std', alpha=0.5)
    dfall.loc[2015:, 'rcp26'].plot(ax=ax1, linewidth=4.0, color=cols[1],
                                   label='RCP 2.6 climate')

    # rcp45
    dfall.loc[2015:, 'rcp45'].plot(ax=ax1, linewidth=4.0, color=cols[2],
                                   label='RCP 4.5 climate')
    # rcp60
    dfall.loc[2015:, 'rcp60'].plot(ax=ax1, linewidth=4.0, color=cols[3],
                                   label='RCP 6.0 climate')

    # rcp85
    ax1.fill_between(dfall.loc[2015:, 'rcp85'].index,
                     dfall.loc[2015:, 'rcp85'] - dfallstd.loc[2015:, 'rcp85'],
                     dfall.loc[2015:, 'rcp85'] + dfallstd.loc[2015:, 'rcp85'],
                     color=cols[4], alpha=0.5)
    #ax1.plot(0, 0, color='C3', linewidth=10,
    #         label='rcp85 ensemble mean +/- 1 std', alpha=0.5)
    dfall.loc[2015:, 'rcp85'].plot(ax=ax1, linewidth=4.0, color=cols[4],
                                   label='RCP 8.5 climate')

    # plot commitment length
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
    #ax1.plot(0, 0, color='C1', linewidth=10,
    #         label='random climate 1984-2014 mean +/- 1 std', alpha=0.5)
    ax1.plot([2105.5, 2110.5], [postlength, postlength], linewidth=4.0,
             color=cols[5],
             label='Random climate (1984-2014) equilibrium length from multiple seeds')

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
    #ax1.plot(0, 0, color='C9', linewidth=10,
    #         label='random climate 1870-1900 +/- 1 std', alpha=0.5)
    ax1.plot([2105.5, 2110.5], [prelength, prelength], linewidth=4.0,
             color=cols[6],
             label='Random climate (1870-1900) equilibrium length from multiple seeds')

    ylim = ax1.get_ylim()
    #ax1.plot([2015, 2015], ylim, 'k-', linewidth=2)
    ax1.set_xlim([1850, 2112])
    #ax1.set_ylim(ylim)

    ax2 = ax1.twinx()
    ax2.set_ylabel('apporixmate absolute glacier length [m]', fontsize=22)
    y1, y2 = get_absolute_length(ylim[0], ylim[1], rgi, df99, histalp_storage)
    ax2.tick_params(axis='both', which='major', labelsize=18)
    ax2.set_ylim([y1, y2])

    name = GLCDICT.get(rgi.split('_')[0])[2]
    ax1.set_title('%s' % name, fontsize=28)

    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)

    ax1.tick_params(axis='both', which='major', labelsize=22)
    #ax1.set_xticks([1900, 2000, 2114, 2214, 2314])
    #ax1.set_xticklabels(['1900', '2000', '100', '200', '300'])
    ax1.grid(True)

    ax1.legend(bbox_to_anchor=(-0.0, -0.1), loc='upper left', fontsize=16,
               ncol=4)

    fig.tight_layout()
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