import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from relic.postprocessing import calc_acdc
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
