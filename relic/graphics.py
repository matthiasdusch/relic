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

    for col in cols:

        # relative spinup length
        lsp = df.loc[:, col] - df.loc[0, col]

        fig, ax = plt.subplots(figsize=[15, 8])
        ax.plot(lsp, 'C0', label='OGGM Spinup')
        ax.plot(obs, 'k', label='Observed dL (1850-2003)')
        ax.set_title('%s %s\n(Bias = %.2f [deg C], %s = %.2e)' %
                     (meta['name'].iloc[0], meta['RGI_ID'].iloc[0], tbias,
                      colname, col))
        ax.set_ylabel('delte length [m]')
        ax.set_xlabel('spinup years')
        ax.legend()
        fig.tight_layout()
        """
        fn = os.path.join(cfg.PATHS['working_dir'],
                          'spinup_%s.png' % glc['name'].iloc[0])
        fig.savefig(fn)
        """
