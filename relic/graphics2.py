import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import pickle
import pandas as pd

from relic.postprocessing import (pareto, std_quotient,
                                  mae_weighted, pareto3)
from relic.preprocessing import GLCDICT


def allruns(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    fig1, ax1 = plt.subplots(figsize=[17, 9])

    nolbl = df.loc[:, df.columns != 'obs'].rolling(y_len, center=True).mean().copy()
    nolbl.columns = ['' for i in range(len(nolbl.columns))]
    nolbl.plot(ax=ax1, linewidth=0.5, color='0.6')

    # plot observations
    df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
        plot(ax=ax1, color='k', marker='o', label='Observed length change',
             markersize=10)

    # OGGM standard
    for run in df.columns:
        if run == 'obs':
            continue
        para = ast.literal_eval('{' + run + '}')
        if ((np.abs(para['prcp_scaling_factor'] - 1.75) < 0.01) and
                (para['mbbias'] == 0) and
                (para['glena_factor'] == 1)):
            df.loc[:, run].rolling(y_len, center=True). \
                mean().plot(ax=ax1, linewidth=3, color='k',
                            label='OGGM default parameters')

    name = GLCDICT.get(rgi_id)[2]

    ax1.set_title('%s' % name, fontsize=30)
    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)
    ax1.set_xlim([1850, 2020])
    ax1.set_ylim([-12000, 5000])
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True)

    ax1.legend(fontsize=20, loc=3)

    fig1.tight_layout()
    fn1 = os.path.join(pout, 'allruns.pdf')
    fig1.savefig(fn1)


def spezruns(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    fig1, ax1 = plt.subplots(figsize=[17, 9])

    # get MAEs
    maes = mae_weighted(df, normalised=True).sort_values().iloc[:130]

    # get stdquot
    stdmae = std_quotient(df.loc[:, np.append(maes.index.values, 'obs')],
                          normalised=True).sort_values()

    nolbl = df.loc[:, maes.index].rolling(y_len, center=True).mean().copy()
    nolbl.columns = ['' for i in range(len(nolbl.columns))]
    nolbl.plot(ax=ax1, linewidth=0.5, color='0.8')

    # plot observations
    df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
        plot(ax=ax1, color='k', marker='o', label='Observed length change',
             markersize=10)

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

    # objective 1
    #df.loc[:, maes.index[0]].rolling(y_len, center=True). \
    #    mean().plot(ax=ax1, linewidth=2, color='C0',
    #                label='min MAE weighted')

    #df.loc[:, stdmae.index[0]].rolling(y_len, center=True). \
    #    mean().plot(ax=ax1, linewidth=2, color='C1',
    #                label='min |1 - std quotient|')

    paretodict = pareto({glid: df}, None)
    df.loc[:, paretodict[glid]].rolling(y_len, center=True). \
        mean().plot(ax=ax1, linewidth=6, color='C2',
                    label='finally chosen run')

    print(paretodict[glid])

    name = GLCDICT.get(rgi_id)[2]

    ax1.set_title('%s' % name, fontsize=30)
    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)
    ax1.set_xlim([1850, 2020])
    ax1.set_ylim([-3000, 1000])
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True)

    ax1.legend(fontsize=20, loc=3)

    fig1.tight_layout()
    fn1 = os.path.join(pout, 'dummy.pdf')
    fig1.savefig(fn1)


def paramplots(df, glid, pout, y_len=None):
    # take care of merged glaciers
    rgi_id = glid.split('_')[0]

    allvars = ['prcp_scaling_factor', 'mbbias', 'glena_factor']
    for var in allvars:
        notvars = allvars.copy()
        notvars.remove(var)

        fig1, ax1 = plt.subplots(figsize=[17, 9])

        # get MAEs
        maes = mae_weighted(df, normalised=True).sort_values().iloc[:130]


        # plot observations
        df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
            plot(ax=ax1, color='k', marker='o', label='Observed length change',
                 markersize=10)

        paretodict = pareto3({glid: df})
        papar = ast.literal_eval('{' + paretodict[glid][0] + '}')

        nolbl = df.loc[:, paretodict[glid]].rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]
        nolbl.plot(ax=ax1, linewidth=0.5, color='0.8')

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
                    mean().plot(ax=ax1, linewidth=2, color='k',
                                label='OGGM default parameters')

            if ((np.isclose(para[notvars[0]],
                            papar[notvars[0]], atol=0.01)) and
                (np.isclose(para[notvars[1]],
                            papar[notvars[1]], atol=0.01))):

                dfvar.loc[:, para[var]] = df.loc[:, run]

        if var == 'prcp_scaling_factor':
            colors = ["#4B0055","#471D67","#3C3777","#1E4D85","#006290","#007796","#008A98","#009B95","#00AC8E","#00BA82","#25C771","#73D25B","#A6DA42","#D4E02D","#FDE333"]
            colors.reverse()
            lbl = 'Precip SF: '
        elif var == 'glena_factor':
            lbl = 'Glen A: '
            colors = ["#001889","#67008E","#9C008E","#C32D80","#DD5E61","#EC8F21","#F1C500"]
            colors.reverse()
        elif var == 'mbbias':
            colors = ["#00308D","#064D9B","#436CB7","#698CD6","#8DADF8","#B1CFFF","#FFD8D9","#FFB5B5","#F59393","#D17171","#AE5050","#8C2F2F","#6C0203"]
            lbl = 'MB bias: '

        dfvar = dfvar.sort_index(axis=1)
        col = dfvar.columns.astype('str').to_list()
        col = [lbl + c for c in col]
        col[1:-1] = ['' for i in np.arange(1, len(dfvar.columns)-1)]
        dfvar.columns = col
        dfvar.rolling(y_len, center=True).mean().plot(ax=ax1, color=colors,
                                                      linewidth=4)

        df.loc[:, paretodict[glid][0]].rolling(y_len, center=True). \
            mean().plot(ax=ax1, linewidth=6, color='C2',
                        label='finally chosen run')

        name = GLCDICT.get(rgi_id)[2]

        ax1.set_title('%s' % name, fontsize=30)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-2500, 500])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(fontsize=20, loc=3)

        fig1.tight_layout()
        fn1 = os.path.join(pout, 'dummy_%s_%s.png' % (glid, var))
        fig1.savefig(fn1)

