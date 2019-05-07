import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from relic.postprocessing import calc_acdc
from relic.preprocessing import get_leclercq_observations


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


def plt_histalp_runs(spinup, df, meta, data, pout, colname=None, cols=None):
    if len(df) == 0:
        return
    assert len(meta) == 1
    assert meta['first'].iloc[0] == df.index[0]
    if cols is None:
        cols = df.columns.levels[0]
    fig, ax = plt.subplots(figsize=[15, 8])
    data.plot(ax=ax, color='k', marker='.', label='Observed length change')

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

        ax.plot(hist, label='OGGM %s = %.2e' % (colname, col))
    ax.set_title('%s %s' % (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('delta length [m]')
    ax.set_xlabel('year')
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


def plt_multiple_runs(runs, pout):

    meta, data = get_leclercq_observations()

    # get all glaciers
    glcs = [gl['rgi_id'] for gl in list(runs[0].values())[0]]

    for glid in glcs:
        _meta = meta.loc[meta['RGI_ID'] == glid].copy()
        _data = data.loc[_meta.index[0]].copy()

        fig, ax = plt.subplots(figsize=[15, 8])
        _data.plot(ax=ax, color='k', marker='.',
                   label='Observed length change')

        for run in runs:
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == glid][0]
            except IndexError:
                continue
            rkey = list(run.keys())[0]

            # tbias = rdic['tbias']

            spin = (rdic['spinup'].loc[:] -
                    rdic['spinup'].loc[0]).dropna().iloc[-1]

            dl = spin + _meta['dL2003'].iloc[0]

            # relative length change
            hist = rdic['histalp'].loc[:] - rdic['histalp'].iloc[0] + dl

            #ax.plot(hist, label='%s, %s = %.2f' % (rkey, 'tbias', tbias))
            ax.plot(hist, label='%s' % rkey)

        ax.set_title('%s %s' % (_meta['name'].iloc[0],
                                _meta['RGI_ID'].iloc[0]))
        ax.set_ylabel('delta length [m]')
        ax.set_xlabel('year')
        ax.legend()
        fig.tight_layout()
        fn = os.path.join(pout, 'histalp_%s.png' % _meta['name'].iloc[0])
        fig.savefig(fn)

