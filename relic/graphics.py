import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import ast

from relic.postprocessing import (calc_acdc, pareto, merged_ids, glcnames,
                                  mae_all, mae_diff_mean, mae_diff_yearly)
from relic.preprocessing import get_leclercq_observations
from relic.process_length_observations import add_custom_length


def visual_check_spinup(df, meta, tbias, pout, colname=None, cols=None):

    assert len(meta) == 1

    if cols is None:
        cols = df.columns.levels[0]

    # spinup goal is observed difference
    obs = np.zeros_like(df.index) - meta['dL2003'].iloc[0]

    fig, ax = plt.subplots(figsize=[15, 8])
    for col in cols:

        if isinstance(col, tuple) and (len(col) == 1):
            col = col[0]
        # relative spinup length
        lsp = df.loc[:, col] - df.loc[0, col]
        # lsp = df.loc[:, col]
        if lsp.dropna().size > 0:
            toplot = lsp.dropna().copy()
            tb = '%.2f' % tbias[col]
        else:
            toplot = obs.copy()
            tb = 'RunTimeError'
        ax.plot(toplot,
                label='OGGM %s = %.2e, tbias = %s' % (colname, col, tb))
    ax.plot(obs, 'k', label='Observed dL (1850-2003)')
    ax.set_title('%s %s' % (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('delte length [m]')
    ax.set_xlabel('spinup years')
    ax.legend()
    fig.tight_layout()
    fn = os.path.join(pout, 'spinup_%s.png' % meta['name'].iloc[0])
    fig.savefig(fn)


def plt_histalp_runs(spinup, df, meta, data, pout, colname=None, cols=None,
                     y_roll=1):
    if len(df) == 0:
        return
    assert len(meta) == 1
    assert meta['first'].iloc[0] == df.index[0]
    if cols is None:
        cols = df.columns.levels[0]
    fig, ax = plt.subplots(figsize=[15, 8])
    data.plot(ax=ax, color='k', marker='o', label='Observed length change')

    for col in cols:
        if isinstance(col, tuple) and (len(col) == 1):
            col = col[0]
        try:
            spin = (spinup.loc[:, col] - spinup.loc[0, col]).dropna().iloc[-1] # [0]
        except IndexError:
            # pass
            continue

        dl = spin + meta['dL2003'].iloc[0]
        # relative length change
        hist = df.loc[:, col] - df.loc[:, col].iloc[0] + dl

        ax.plot(hist.rolling(y_roll).mean(),
                label='%s = %.2e' % (colname, col), linewidth=3)
    ax.set_title('%s %s' % (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('delta length [m]')
    ax.set_xlabel('year')
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fn = os.path.join(pout, 'histalp_%s.png' % meta['name'].iloc[0])
    fig.savefig(fn)


def accum_error(spinup, df, meta, data, pout, colname=None, cols=None):
    if cols is None:
        cols = df.columns.levels[0]

    assert meta['first'].iloc[0] == df.index[0]

    fig, ax = plt.subplots(figsize=[15, 8])
    for col in cols:
        if isinstance(col, tuple) and (len(col) == 1):
            col = col[0]
        try:
            spin = (spinup.loc[:, col] - spinup.loc[0, col]).dropna().iloc[-1][0]
        except IndexError:
            pass

        acdc = calc_acdc(data, spinup, df, meta, col)

        ax.plot(acdc, label='OGGM %s = %.2e' % (colname, col))

    ax.set_title('Accumulated difference change\n%s %s' %
                 (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('acdc [m]')
    ax.set_xlabel('year')
    ax.legend()
    fig.tight_layout()
    fn = os.path.join(pout, 'acdc_%s.png' % meta['name'].iloc[0])
    fig.savefig(fn)


def plt_multiple_runs(runs, pout, y_roll=1, reference=None):

    meta, data = get_leclercq_observations()

    # get all glaciers
    glcs = [gl['rgi_id'] for gl in list(runs[0].values())[0]]

    for glid in glcs:
        _meta = meta.loc[meta['RGI_ID'] == glid].copy()
        _data = data.loc[_meta.index[0]].copy()

        fig, ax = plt.subplots(figsize=[15, 8])
        _data.rolling(y_roll, min_periods=1).mean().\
            plot(ax=ax, color='k', marker='o', label='Observed length change')

        df = pd.DataFrame([], index=np.arange(1850, 2011))
        mae = pd.Series()
        r2 = pd.Series()

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == glid][0]
            except IndexError:
                continue

            rkey = list(run.keys())[0]
            lbl = rkey + ', MAE=%.2f, r2=%.2f' % (rdic['mae'], rdic['r2'])

            df.loc[rdic['rel_dl'].index, lbl] = rdic['rel_dl']

            mae.loc[lbl] = rdic['mae']
            r2.loc[lbl] = rdic['r2']

            if rkey == reference:
                refix = lbl

        maemin = mae.idxmin()
        r2max = r2.idxmax()
        refix = None

        df.loc[:, ~df.columns.isin([maemin, r2max, refix])].\
            rolling(y_roll, center=True).mean().\
            plot(ax=ax, linewidth=0.7)#, color='0.5')

        df.loc[:, maemin].rolling(y_roll, center=True).\
            mean().plot(ax=ax, linewidth=3, color='C3')
        df.loc[:, r2max].rolling(y_roll, center=True).\
            mean().plot(ax=ax, linewidth=3, color='C2')

        if reference is not None:
            df.loc[:, refix].rolling(y_roll).mean().plot(ax=ax,
                                                         linewidth=2,
                                                         color='C0')

        """
        if 'Argentiere' in _meta['name'].iloc[0]:
            fig2, ax2 = plt.subplots(figsize=[10, 10])
            ax2.plot(_data.dropna(), df.loc[_data.dropna().index, r2max], 'o',
                     color='C2')
            ax2.plot(_data.dropna(), df.loc[_data.dropna().index, maemin], '.',
                     color='C3')
            plt.show()
        """

        ax.set_title('%s %s' % (_meta['name'].iloc[0],
                                _meta['RGI_ID'].iloc[0]))
        ax.set_ylabel('delta length [m]')
        ax.set_xlabel('year')
        ax.set_xlim([1850, 2015])
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fn = os.path.join(pout, 'histalp_%s.png' % _meta['name'].iloc[0])
        fig.savefig(fn)


def plt_correlation(runs, pout, y_len=1, y_corr=10, reference=None):

    meta, data = get_leclercq_observations()
    meta, data = add_custom_length(meta, data,
                                   ['RGI60-11.02051', 'RGI60-11.02709'])

    # get all glaciers
    glcs = [gl['rgi_id'] for gl in list(runs[0].values())[0]]

    for glid in glcs:
        # take care of merged glaciers
        rgi_id = glid.split('_')[0]
        _meta = meta.loc[meta['RGI_ID'] == rgi_id].copy()
        _data = data.loc[_meta.index[0]].copy()

        fig1, ax1 = plt.subplots(figsize=[17, 8])
        fig2, ax2 = plt.subplots(figsize=[17, 8])

        _data.rolling(y_len, min_periods=1).mean(). \
            plot(ax=ax1, color='k', marker='o', label='Observed length change')

        if '_merged' in glid:
            mids = {'RGI60-11.02715': 'RGI60-11.02709',
                    'RGI60-11.02119': 'RGI60-11.02051'
                    }
            mid = mids[rgi_id]

            _mmeta = meta.loc[meta['RGI_ID'] == mid].copy()
            _mdata = data.loc[_mmeta.index[0]].copy()
            _mdata.rolling(y_len, min_periods=1).mean(). \
                plot(ax=ax1, color='k', marker='^',
                     label='Observed length change at tributary')
            dfmerge = pd.DataFrame([], index=np.arange(1850, 2011))

        df = pd.DataFrame([], index=np.arange(1850, 2011))
        mae = pd.Series()

        mbplus = []
        mbminus = []

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == glid][0]
            except IndexError:
                continue

            if np.isnan(rdic['tbias']):
                continue

            rkey = list(run.keys())[0]
            lbl = rkey + ', MAE=%.2f, r2=%.2f' % (rdic['mae'], rdic['r2'])

            """
            # get massbalance bias
            try:
                mbbias = float([mb for mb in lbl.split(',') if 'mbbias' in mb][0].split(':')[-1])
                if mbbias == 25:
                    mbplus.append(lbl)
                elif mbbias == -25:
                    mbminus.append(lbl)
            except IndexError:
                pass
            """

            if (np.abs(_data-rdic['rel_dl']).dropna() < 20000).all():
                df.loc[rdic['rel_dl'].index, lbl] = rdic['rel_dl']

            mae.loc[lbl] = rdic['mae']

            if '_merged' in glid:
                dfmerge.loc[rdic['trib_dl'].index, lbl] = rdic['trib_dl']

        if df.empty:

            fn1 = os.path.join(pout, 'histalp_%s.png' % _meta['name'].iloc[0])
            fig1.savefig(fn1)
            fn2 = os.path.join(pout, 'correlation_%s.png' % _meta['name'].iloc[0])
            fig2.savefig(fn2)
            continue

        maemin = mae.idxmin()

        dfcorr = df.rolling(y_corr, min_periods=int(y_corr/2), center=True).\
            corr(_data)

        maxcorr = dfcorr.mean().idxmax()
        medcorr = dfcorr.median().idxmax()

        df_dt = df.copy()
        df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
        rundif = df_dt.resample("10Y").mean().diff()

        obs = _data.copy()
        obs.index = pd.to_datetime(obs.index, format="%Y")
        obs = obs.resample("10Y").mean().diff()

        xcorr = rundif.sub(obs, axis=0).abs().mean().idxmin()

        #dfxcorr = df.copy()
        #dfxcorr['obs'] = _data
        #dfxcorr = dfxcorr.corr()['obs']
        #xcorr = dfxcorr.loc[~dfxcorr.index.isin(['obs'])].idxmax()

        # length plot
        others = df.loc[:, ~df.columns.isin([maemin, maxcorr, medcorr])].\
            rolling(y_len, center=True).mean()
        # df.loc[:, ~df.columns.isin([maemin, maxcorr, medcorr])]. \
        #    rolling(y_len, center=True).mean(). \
        #    plot(ax=ax1, linewidth=0.4, color='0.8', label='_')

        others.columns = ['' for i in range(len(others.columns))]
        try:
            others.plot(ax=ax1, linewidth=0.4, color='0.8')
        except TypeError:
            pass

        df.loc[:, maemin].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=3, color='C0')
        df.loc[:, maxcorr].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=3, color='C2')
        df.loc[:, medcorr].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=3, color='C4')
        df.loc[:, xcorr].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=3, color='C6')

        if '_merged' in glid:
            dfmerge.loc[:, maemin].rolling(y_len, center=True). \
                mean().plot(ax=ax1, linewidth=2, color='C0',
                            marker='^')
            dfmerge.loc[:, xcorr].rolling(y_len, center=True). \
                mean().plot(ax=ax1, linewidth=2, color='C6',
                            marker='^')

        """
        try:
            df.loc[:, mbplus].rolling(y_len, center=True). \
                mean().plot(ax=ax1, linewidth=0.4, color='C1')
            df.loc[:, mbminus].rolling(y_len, center=True). \
                mean().plot(ax=ax1, linewidth=0.4, color='C3')
        except TypeError:
            pass
        """

        ax1.set_title('%s %s' % (_meta['name'].iloc[0],
                                 _meta['RGI_ID'].iloc[0]))
        ax1.set_ylabel('delta length [m]')
        ax1.set_xlabel('year')
        ax1.set_xlim([1850, 2015])
        ax1.set_ylim([-5000, 1000])
        ax1.grid(True)
        ax1.legend()
        # fig1.tight_layout()
        fn1 = os.path.join(pout, 'histalp_%s.png' % _meta['name'].iloc[0])
        fig1.savefig(fn1)

        # corrplots
        # dfcorr.loc[:, ~dfcorr.columns.isin([maemin, maxcorr, medcorr])].\
        #    plot(ax=ax2, linewidth=0.4, color='0.8', legend=False)

        othercorr = dfcorr.loc[:, ~dfcorr.columns.isin(
            [maemin, maxcorr, medcorr])].\
            rolling(y_len, center=True).mean()

        othercorr.columns = ['' for i in range(len(othercorr.columns))]
        try:
            othercorr.plot(ax=ax2, linewidth=0.4, color='0.8')
        except TypeError:
            pass

        dfcorr.loc[:, maemin].plot(ax=ax2, linewidth=3, color='C0')
        dfcorr.loc[:, maxcorr].plot(ax=ax2, linewidth=3, color='C2')
        dfcorr.loc[:, medcorr].plot(ax=ax2, linewidth=3, color='C4')
        dfcorr.loc[:, xcorr].plot(ax=ax2, linewidth=3, color='C6')

        ax2.set_title('rolling correlation (%d years) for %s %s' %
                      (y_corr, _meta['name'].iloc[0], _meta['RGI_ID'].iloc[0]))
        ax2.set_ylabel('correlation')
        ax2.set_xlabel('year')
        ax2.set_xlim([1850, 2015])
        ax2.set_ylim([-1, 1])
        ax2.grid(True)
        ax2.legend()
        fig2.tight_layout()
        fn2 = os.path.join(pout, 'correlation_%s.png' % _meta['name'].iloc[0])
        fig2.savefig(fn2)


def poster_plot(glcdict, pout, y_len=1):

    maedyr = 5

    paretodict = pareto(glcdict, maedyr)
    # xkcdplot(glcdict, paretodict)

    for glid, df in glcdict.items():

        if df.shape[1] <= 2:
            print('foo')
            continue

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        fig1, ax1 = plt.subplots(figsize=[17, 7])

        # grey lines
        nolbl = df.loc[:, ~df.columns.isin(['obs', paretodict[glid]])]. \
            rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]

        nolbl.plot(ax=ax1, linewidth=0.5, color='0.75')

        # plot observations
        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax1, color='k', marker='o', label='Observed length change')
        if glid == 'RGI60-11.01346':
            df.loc[1984:2003, 'obs'].rolling(1, min_periods=1).mean(). \
                plot(ax=ax1, color='0.4', marker='o',
                     label='Linear interpolation to 2013 length')

        # objective 1
        maes = mae_all(df, normalised=True).idxmin()
        df.loc[:, maes].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=2, color='C0',
                        label='Best result for Objective 1')

        # objective 2
        maediff = mae_diff_yearly(df, maedyr, normalised=True).idxmin()
        df.loc[:, maediff].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=2, color='C1',
                        label='Best result for Objective 2')

        # OGGM standard
        for run in df.columns:
            if run == 'obs':
                continue
            para = ast.literal_eval('{' + run + '}')
            if ((np.abs(para['prcp_scaling_factor']-1.8) < 0.01) and
                    (para['mbbias'] == 0) and
                    (para['glena_factor'] == 1.5)):
                df.loc[:, run].rolling(y_len, center=True). \
                    mean().plot(ax=ax1, linewidth=1, color='k',
                                label='OGGM standard parameters')

        # best run
        # get parameters
        params = ast.literal_eval('{' + paretodict[glid] + '}')

        legend = ('\nBest simulated length change:\n'
                  '%.1f precipitation scaling factor\n'
                  '%.1f Glen A factor\n'
                  '%.1f [m w.e. a' r'$^{-1}$' '] mass balance bias'
                  % (params['prcp_scaling_factor'],
                     params['glena_factor'],
                     params['mbbias']/1000))

        df.loc[:, paretodict[glid]].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=4, color='C2',
                        label='Best simulated length change:')

        name = glcnames(glid)

        # add merged tributary
        if 'XXX_merged' in glid:
            mid = merged_ids(glid)

            if 'Tschierva' in name:
                trib = 'Tschierva'
            elif 'Mine' in name:
                trib = 'Mont Mine'

            glcdict[mid].loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
                plot(ax=ax1, color='k', marker='^',
                     label='Observed length change at %s' % trib)

            trib_main = glcdict[mid].loc[:, paretodict[glid]]. \
                rolling(y_len, center=True).mean()
            trib_main[trib_main != 0].\
                plot(ax=ax1, linewidth=3, color='C2',
                     label='Simulated length change at %s' % trib)

            """
            trib_trib = glcdict[mid].loc[:, paretodict[mid]]. \
                rolling(y_len, center=True).mean()
            trib_trib[trib_trib != 0].plot(ax=ax1, linewidth=3, color='C2',
                                           label='_')
            """

        ax1.set_title('%s' % name, fontsize=30)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2010])
        ax1.set_ylim([-3500, 500])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        l1 = ('%.1f precipitation scaling factor' %
              params['prcp_scaling_factor'])
        l2 = '%.1f Glen A factor' % params['glena_factor']
        l3 = ('%.1f [m w.e. a' r'$^{-1}$' '] mass balance bias' %
              (params['mbbias']/1000))

        plt.plot(0, 0, color='w', alpha=0, label=l1)
        plt.plot(0, 0, color='w', alpha=0, label=l2)
        plt.plot(0, 0, color='w', alpha=0, label=l3)

        hix = 4
        if glid == 'RGI60-11.01346':
            plt.plot(0, 0, color='w', alpha=0, label=' ')
            hix = 5

        hdl, lbl = ax1.get_legend_handles_labels()
        hdl2 = np.concatenate(([hdl[-hix]], hdl[-hix+1:],
                               hdl[0:-hix]), axis=0)
        lbl2 = np.concatenate(([lbl[-hix]], lbl[-hix+1:],
                               lbl[0:-hix]), axis=0)
        ax1.legend(hdl2, lbl2,
                   fontsize=16, loc=3, ncol=2)

        fig1.tight_layout()
        # fn1 = os.path.join(pout, 'histalp_%s.png' % name.split()[0])
        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        # fn1b = os.path.join(pout, 'histalp_%s.pdf' % name.split()[0])
        fig1.savefig(fn1)
        # fig1.savefig(fn1b)


def xkcdplot(glcs, paretodict):

    glid = 'RGI60-11.02119_merged'

    df = glcs[glid]
    run = paretodict[glid]

    now = df.loc[:, run].copy()
    now.loc[1855] = -5
    now.loc[1857] = -10
    now.loc[1860] = -15

    #now.loc[1870:1885] *= np.cos(np.linspace(0, np.pi/3, 16))
    #now.loc[1885:1910] *= np.cos(np.linspace(np.pi/3, 0, 26))
    now.loc[1870:1910] += np.append(np.linspace(0, 500, 16),
                                    np.linspace(500, 0, 25))
    now.loc[1887:1903] += np.append(np.linspace(0, 200, 7),
                                    np.linspace(200, 0, 10))

    now.loc[1915:1961] -= np.append(np.linspace(0, 400, 12),
                                    np.linspace(400, 0, 35))
    now.loc[1937:1955] -= np.linspace(0, 400, 19)
    now.loc[1956:1961] -= np.linspace(400, 0, 6)

    now.loc[1960] -= 70

    now.loc[1973:1993] += np.append(np.linspace(0, 300, 11),
                                    np.linspace(300, 0, 10))

    now = now.rolling(10, center=True).mean()

    ertime = np.arange(1860, 2010, 10)

    err = df.loc[:, 'obs'].interpolate().loc[ertime] - now.loc[ertime]

    err += np.random.randint(50, 100, len(ertime))

    x=np.array([1700, 1750, 1800, 1860])
    y=np.array([-3000, -4000, -2000, -70])
    pf = np.polyfit(x,y,5)
    p=np.poly1d(pf)
    past = pd.DataFrame(p(np.arange(1710, 1861)), index=np.arange(1710, 1861))
    past[0] += np.random.randint(-30, 30, len(past))

    x=np.array([2006, 2020, 2030, 2060, 2100])
    y=np.array([-2600, -3500, -4000, -4200, -4300])
    pf = np.polyfit(x,y,3)
    p=np.poly1d(pf)
    fut = pd.DataFrame(p(np.arange(2006, 2100)), index=np.arange(2006, 2100))
    fut[0] += np.random.randint(-30, 30, len(fut))

    with plt.xkcd():
        fig1, ax1 = plt.subplots(figsize=[17, 6])

        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax1, marker='o', label='Observed length change',
                 markeredgecolor='k', color='k')

        past.plot(ax=ax1, color='C0', linewidth=4, label=' ', legend=False)
        fut.plot(ax=ax1, color='C0', linewidth=4, label=' ', legend=False)

        now.plot(ax=ax1, linewidth=4, color='C2',
                 label='Best simulated length change:')

        ax1.errorbar(ertime, now.loc[ertime], yerr=err, fmt='none', color='C2',
                     linewidth=4)

        ax1.errorbar(1740, -4000, yerr=500, xerr=12, fmt='none', color='C3',
                     linewidth=5)
        ax1.errorbar(1795, -2000, yerr=400, xerr=5, fmt='none', color='C3',
                     linewidth=5)
        ax1.errorbar(1851, -10, yerr=100, xerr=3, fmt='none', color='C3',
                     linewidth=5)

        ax1.annotate('error?', (1810, -2000), (1870, -3500),
                     arrowprops={'width': 4, 'facecolor': 'black'},
                     fontsize=36, fontweight='bold')

        ax1.annotate('.', (2020, -4000), (1920, -3500),
                     arrowprops={'width': 4, 'facecolor': 'black'},
                     fontsize=0.1, color='white')

        ax1.text(1878, -3000, 'model', fontweight='bold', fontsize=36,
                 color='C0')

        ax1.annotate('proxy', (1740, -3500), (1730, -500),
                     arrowprops={'width': 4, 'facecolor': 'C3'},
                     fontsize=28, fontweight='bold', color='C3')
        ax1.annotate('.', (1790, -1950), (1760, -500),
                     arrowprops={'width': 4, 'facecolor': 'C3'},
                     fontsize=0.1, fontweight='bold', color='white')
        ax1.annotate('.', (1845, -50), (1760, -400),
                     arrowprops={'width': 4, 'facecolor': 'C3'},
                     fontsize=0.1, fontweight='bold', color='white')

        ax1.set_ylabel('glacier length change', fontsize=26)
        ax1.set_xlabel('time', fontsize=26)
        ax1.set_xlim([1700, 2100])
        ax1.set_ylim([-5000, 500])
        ax1.set_xticklabels([''])
        ax1.set_yticklabels([''])
        ax1.tick_params(axis='both', which='major', labelsize=22)

        pout = '/home/matthias/length_change_1850/multi/array'
        fig1.tight_layout()
        fn1 = os.path.join(pout, 'xkcd.png')
        fn1b = os.path.join(pout, 'xkcd.pdf')
        fig1.savefig(fn1)
        fig1.savefig(fn1b)
        plt.show()
