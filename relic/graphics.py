import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def visual_check_spinup(df, meta, tbias, colname=None, cols=None):

    assert len(meta) == 1

    if cols is None:
        cols = df.columns.levels[0]

    # spinup goal is observed difference
    obs = np.zeros_like(df.index) - meta['dL2003'].iloc[0]

    fig, ax = plt.subplots(figsize=[15, 8])
    for col in cols:

        # relative spinup length
        #lsp = df.loc[:, col] - df.loc[0, col]
        lsp = df.loc[:, col]

        ax.plot(lsp, label='OGGM %s = %.2e, tbias = %.2f' % (colname, col, tbias))
    ax.plot(obs, 'k', label='Observed dL (1850-2003)')
    ax.set_title('%s %s' % (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('delte length [m]')
    ax.set_xlabel('spinup years')
    ax.legend()
    fig.tight_layout()
    """
    fn = os.path.join(cfg.PATHS['working_dir'],
                      'spinup_%s.png' % glc['name'].iloc[0])
    fig.savefig(fn)
    """


def plt_histalp_runs(spinup, df, meta, data, colname=None, cols=None):

    assert len(meta) == 1

    if cols is None:
        cols = df.columns.levels[0]

    fig, ax = plt.subplots(figsize=[15, 8])

    data.plot(ax=ax, color='k', marker='.', label='Observed length change')

    assert meta['first'].iloc[0] == df.index[0]

    for col in cols:

        try:
            spin = (spinup.loc[:, col] - spinup.loc[0, col]).dropna().iloc[-1][0]
        except IndexError:
            pass
        dl = spin + meta['dL2003'].iloc[0]

        # relative length change
        hist = df.loc[:, col] - df.loc[:, col].iloc[0] + dl

        ax.plot(hist, label='OGGM %s = %.2e' % (colname, col))
    ax.set_title('%s %s' % (meta['name'].iloc[0], meta['RGI_ID'].iloc[0]))
    ax.set_ylabel('delta length [m]')
    ax.set_xlabel('year')
    ax.legend()
    fig.tight_layout()
