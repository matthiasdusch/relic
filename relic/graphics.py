import matplotlib

matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import os
import ast
import pickle
import pandas as pd

from relic.postprocessing import mae_weighted, optimize_cov2, calc_coverage_2
from relic.preprocessing import GLCDICT


def paramplots(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[25, 5])

    # get MAEs
    maes = mae_weighted(df, normalised=True).sort_values().iloc[:100]

    allvars = ['prcp_scaling_factor', 'mbbias', 'glena_factor']

    for var, ax in zip(allvars, [ax1, ax2, ax3]):
        notvars = allvars.copy()
        notvars.remove(var)

        # paretodict = pareto3({glid: df})
        # papar = ast.literal_eval('{' + paretodict[glid][0] + '}')

        # papar = {'glena_factor': 1.0, 'mbbias': -200, 'prcp_scaling_factor': 3.5}

        # lets use min MAE here
        # papar = ast.literal_eval('{' + maes.index[0] + '}')
        # lets use OGGM HISTALP default
        papar = {'glena_factor': 1.0, 'mbbias': 0, 'prcp_scaling_factor': 1.75}

        """
        # uncomment for 100 grey lines
        nolbl = df.loc[:, maes.index].rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]
        nolbl.plot(ax=ax, linewidth=0.5, color='0.8')

        # plot 1 for label
        df.loc[:, maes.index[0]].rolling(y_len, center=True). \
            mean().plot(ax=ax, linewidth=0.5, color='0.8',
                        label='100 smallest MAE runs')
        """

        # store specific runs
        dfvar = pd.DataFrame([], index=df.index)

        # OGGM standard
        for run in df.columns:
            if run == 'obs':
                continue
            para = ast.literal_eval('{' + run + '}')
            if ((np.abs(para['prcp_scaling_factor'] - 1.75) < 0.01) and
                    (para['mbbias'] == 0) and
                    (para['glena_factor'] == 1)):
                df.loc[:, run].rolling(y_len, center=True). \
                    mean().plot(ax=ax, linewidth=6, color='k',
                                label='OGGM default parameters')

            if ((np.isclose(para[notvars[0]],
                            papar[notvars[0]], atol=0.01)) and
                    (np.isclose(para[notvars[1]],
                                papar[notvars[1]], atol=0.01))):
                dfvar.loc[:, para[var]] = df.loc[:, run]

        df.loc[:, maes.index[0]].rolling(y_len, center=True). \
            mean().plot(ax=ax, linewidth=6, color='C2',
                        label='minimal MAE run')

        if var == 'prcp_scaling_factor':
            colors = ["#4B0055", "#471D67", "#3C3777", "#1E4D85", "#006290",
                      "#007796", "#008A98", "#009B95", "#00AC8E", "#00BA82",
                      "#25C771", "#73D25B", "#A6DA42", "#D4E02D", "#FDE333"]
            colors.reverse()
            lbl = 'Precip SF: '
        elif var == 'glena_factor':
            lbl = 'Glen A: '
            colors = ["#001889", "#67008E", "#9C008E", "#C32D80", "#DD5E61",
                      "#EC8F21", "#F1C500"]
            colors.reverse()
        elif var == 'mbbias':
            # colors = ["#00308D","#064D9B","#436CB7","#698CD6","#8DADF8","#B1CFFF","#FFD8D9","#FFB5B5","#F59393","#D17171","#AE5050","#8C2F2F","#6C0203"]
            colors = ["#023FA5", "#1B44A4", "#2B4AA4", "#3852A5", "#465BA7",
                      "#5767AC", "#737EB5", "#B66C7B", "#A84E63", "#A03D57",
                      "#9A304E", "#962346", "#921740"]  # ,"#8E063B"
            lbl = 'MB bias [mm w.e.]: '

        dfvar = dfvar.sort_index(axis=1)
        col = dfvar.columns.astype('str').to_list()
        col = [lbl + c for c in col]
        col[1:-1] = ['' for i in np.arange(1, len(dfvar.columns) - 1)]
        dfvar.columns = col
        dfvar.rolling(y_len, center=True).mean().plot(ax=ax, color=colors,
                                                      linewidth=2)

        # plot observations
        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax, color='k', style='.',
                 marker='o', label='Observed length change',
                 markersize=6)

        # ax.set_title('%s' % name, fontsize=30)
        # ax.set_ylabel('relative length change [m]', fontsize=26)
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


def past_simulation_and_params(glcdict, pout, y_len=5):
    for glid, df in glcdict.items():

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        # if rgi_id != 'RGI60-11.03643':
        #    continue

        # if (rgi_id != 'RGI60-11.01450') and (rgi_id != 'RGI60-11.02051') and (rgi_id != 'RGI60-11.01270') and (rgi_id != 'RGI60-11.03643') and (rgi_id != 'RGI60-11.00897'):
        #    #if (rgi_id != 'RGI60-11.02755') and (rgi_id != 'RGI60-11.03646'):
        #    continue

        fig = plt.figure(figsize=[23, 8])

        gs = GridSpec(1, 4)  # 3 rows, 3 columns

        ax1 = fig.add_subplot(gs[0, 0:3])
        ax2 = fig.add_subplot(gs[0, 3])

        df.loc[:, 'obs'].plot(ax=ax1, color='k', marker='o',
                              label='Observed length change')

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
                                label='OGGM default parameters')
                oggmdefault = run

        maes = mae_weighted(df, normalised=False).sort_values()

        idx2plot2 = optimize_cov2(df.loc[:, maes.index[:150]],
                                  df.loc[:, 'obs'], glid, minuse=5)
        # idx2plot2 = optimize_cov2(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5)

        ensmean = df.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()
        ensstdmean = df.loc[:, idx2plot2].std(axis=1).rolling(y_len,
                                                              center=True).mean()

        # coverage
        cov = calc_coverage_2(df, idx2plot2, df['obs'])

        ax1.fill_between(ensmeanmean.index, ensmeanmean - ensstdmean,
                         ensmeanmean + ensstdmean, color='C0', alpha=0.6)

        # nolbl = df.loc[:, idx2plot2].rolling(y_len, center=True).mean().copy()
        # nolbl.columns = ['' for i in range(len(nolbl.columns))]
        # nolbl.plot(ax=ax1, linewidth=0.8, color='C0')

        ax1.plot(0, 0, color='C0', linewidth=10,
                 label='ensemble mean +/- 1 std')

        # plot ens members
        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='C1',
                         label='ensemble mean')

        # reference run (basically min mae)
        df.loc[:, maes.index[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='C4',
                 label='minimum wMAE run')

        name = GLCDICT.get(rgi_id)[2]

        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]

        ax1.set_title('%s' % name, fontsize=28)
        ax1.text(1970, -4700, '%d ensemble members  '
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
                   ncol=3)

        # parameter plots
        from colorspace import sequential_hcl
        col = sequential_hcl('Blues 3').colors(len(idx2plot2) + 3)
        for i, run in enumerate(idx2plot2):
            para = ast.literal_eval('{' + run + '}')
            psf = para['prcp_scaling_factor']
            psf = (psf - 0.5) / (4 - 0.5)
            gla = para['glena_factor']
            gla = (gla - 1) / (4 - 1.0)
            mbb = para['mbbias']
            mbb = (mbb - -1400) / (1000 - -1400)

            ax2.plot([1, 2, 3], [psf, gla, mbb], color=col[i], linewidth=2)

        ax2.set_xlabel('calibration parameters', fontsize=20)
        ax2.set_ylabel('normalized values', fontsize=20)
        ax2.set_xlim([0.8, 3.2])
        ax2.set_ylim([-0.1, 1.1])
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['Psf', 'GlenA', 'MB bias'], fontsize=16)
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels([''])

        # fig1.subplots_adjust(right=0.7)
        fig.tight_layout()
        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig.savefig(fn1)

        used = dict()
        used['oggmdefault'] = oggmdefault
        used['minmae'] = idx2plot2[0]
        used['ensemble'] = idx2plot2

        pickle.dump(used, open(os.path.join(pout, 'runs_%s.p' % glid), 'wb'))