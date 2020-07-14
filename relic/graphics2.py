import matplotlib
matplotlib.use('TkAgg')  # noqa

import matplotlib.pyplot as plt
import numpy as np
import os
import ast
import pickle
import pandas as pd
import itertools


from relic.postprocessing import (pareto, std_quotient,
                                  mae_weighted, pareto3,
                                  mean_error_weighted)
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

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[25, 5])

    # get MAEs
    maes = mae_weighted(df, normalised=True).sort_values().iloc[:100]

    allvars = ['prcp_scaling_factor', 'mbbias', 'glena_factor']

    for var, ax in zip(allvars, [ax1, ax2, ax3]):
        notvars = allvars.copy()
        notvars.remove(var)

        #paretodict = pareto3({glid: df})
        #papar = ast.literal_eval('{' + paretodict[glid][0] + '}')

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
            colors = ["#4B0055","#471D67","#3C3777","#1E4D85","#006290","#007796","#008A98","#009B95","#00AC8E","#00BA82","#25C771","#73D25B","#A6DA42","#D4E02D","#FDE333"]
            colors.reverse()
            lbl = 'Precip SF: '
        elif var == 'glena_factor':
            lbl = 'Glen A: '
            colors = ["#001889","#67008E","#9C008E","#C32D80","#DD5E61","#EC8F21","#F1C500"]
            colors.reverse()
        elif var == 'mbbias':
            #colors = ["#00308D","#064D9B","#436CB7","#698CD6","#8DADF8","#B1CFFF","#FFD8D9","#FFB5B5","#F59393","#D17171","#AE5050","#8C2F2F","#6C0203"]
            colors = ["#023FA5","#1B44A4","#2B4AA4","#3852A5","#465BA7","#5767AC","#737EB5","#B66C7B","#A84E63","#A03D57","#9A304E","#962346","#921740"]#,"#8E063B"
            lbl = 'MB bias [mm w.e.]: '

        dfvar = dfvar.sort_index(axis=1)
        col = dfvar.columns.astype('str').to_list()
        col = [lbl + c for c in col]
        col[1:-1] = ['' for i in np.arange(1, len(dfvar.columns)-1)]
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

    #ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_ylabel('relative length change [m]', fontsize=18)

    name = GLCDICT.get(rgi_id)[2]
    fig1.suptitle('%s' % name, fontsize=30)
    fig1.tight_layout(rect=[0, 0.0, 0.99, 0.94])
    fn1 = os.path.join(pout, '%s.pdf' % glid)
    fn1 = os.path.join(pout, '%s.png' % glid)
    fig1.savefig(fn1)


def gpr(df, glid, pout, y_len=5):

    rgi_id = glid.split('_')[0]

    fig1, ax1 = plt.subplots(figsize=[17, 9])

    # get MAEs
    maes = mae_weighted(df, normalised=True).sort_values().iloc[:13]

    nolbl = df.loc[:, maes.index].rolling(y_len, center=True).mean().copy()
    nolbl.columns = ['' for i in range(len(nolbl.columns))]
    nolbl.plot(ax=ax1, linewidth=0.5, color='0.8')

    # plot observations
    df.loc[:, 'obs'].rolling(1, min_periods=1).mean(). \
        plot(ax=ax1, color='k', marker='o', label='Observed length change',
             markersize=10)



    x = np.array([])
    y = np.array([])

    for idx, _ in maes.iteritems():
        x = np.append(x, df.loc[:, idx].dropna().index.values)
        y = np.append(y, df.loc[:, idx].dropna().values)

    x = x.reshape(len(x), 1)
    y = y.reshape(len(y), 1)




    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (ExpSineSquared, WhiteKernel,
                                                  ConstantKernel, Matern, RBF,
                                                  RationalQuadratic)

    # gp_kernel = ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) \
    #     + WhiteKernel(1e-1)
    # kernel = ConstantKernel() + Matern(length_scale=40, nu=3/2) + WhiteKernel(noise_level=1)

    kernel = (RBF(length_scale=100) +
              RationalQuadratic(alpha=50, length_scale=10) + WhiteKernel(noise_level=100))

    gpr = GaussianProcessRegressor(kernel=kernel)

    x = df.loc[:, 'obs'].dropna().index.values.reshape(-1, 1)
    y = df.loc[:, 'obs'].dropna().values.reshape(-1, 1)

    gpr.fit(x, y)

    x_pred = np.unique(x).reshape(-1, 1)
    y_pred, sigma = gpr.predict(x_pred, return_std=True)

    print(gpr)
    ax1.plot(x_pred, y_pred, 'r')
    ax1.fill_between(x_pred[:, 0], y_pred[:, 0] - sigma, y_pred[:, 0] + sigma,
                     alpha=0.5, color='k')

    name = GLCDICT.get(rgi_id)[2]

    ax1.set_title('%s' % name, fontsize=30)
    ax1.set_ylabel('relative length change [m]', fontsize=26)
    ax1.set_xlabel('Year', fontsize=26)
    ax1.set_xlim([1850, 2020])
    ax1.set_ylim([-1500, 500])
    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax1.grid(True)

    ax1.legend(fontsize=20, loc=3)

    fig1.tight_layout()
    #fn1 = os.path.join(pout, 'allruns.pdf')
    #fig1.savefig(fn1)
    plt.show()


def paramscatter(glcdict, pout):

    paretodict = pareto3(glcdict)

    # get glacier parameters:
    glcs = [glid.strip('_merged') for glid in paretodict.keys()]
    from oggm.utils import get_rgi_glacier_entities
    rgidf = get_rgi_glacier_entities(glcs)

    print(rgidf.columns)

    fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=[17, 6])

    for glid, df in glcdict.items():
        # take care of merged glaciers
        rgi_id = glid.split('_')[0]
        print(rgi_id)

        for run in paretodict[glid]:
            # get calibration parameters
            par = ast.literal_eval('{' + run + '}')
            print(run)

            var = rgidf.loc[rgidf.RGIId == glid.strip('_merged'), 'Slope']

            ax1.plot(par['glena_factor'], var, '.k')
            ax2.plot(par['prcp_scaling_factor'], var, '.k')
            ax3.plot(par['mbbias'], var, '.k')

    ax1.set_xlabel('Glen A factor', fontsize=26)
    ax2.set_xlabel('Prcp scaling factor', fontsize=26)
    ax3.set_xlabel('MB bias', fontsize=26)

    ax1.set_ylabel('mean elevation', fontsize=26)

    ax1.tick_params(axis='both', which='major', labelsize=22)
    ax2.tick_params(axis='xaxis', which='major', labelsize=22)
    ax3.tick_params(axis='xaxis', which='major', labelsize=22)

    ax1.grid(True)
    ax2.grid(True)
    ax3.grid(True)

    fig1.tight_layout()
    # fn1 = os.path.join(pout, 'dummy.pdf')
    # fig1.savefig(fn1)
    plt.show()


def quick_plot(glcdict, pout, y_len=5):

    from relic.postprocessing import pareto_5, pareto_5dta, pareto_5dtb, pareto_simple
    # pdic = pareto_simple(glcdict)
    # pdic5b = pareto_5dtb(glcdict)

    for glid, df in glcdict.items():

        # recalc for 1980
        ix1980 = df.dropna().index[0]
        df.loc[:, df.columns != 'obs'] += df.loc[ix1980, 'obs']

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        """
        
        if (rgi_id != 'RGI60-11.03646') and (rgi_id != 'RGI60-11.00106') and (
                rgi_id != 'RGI60-11.00897') and (
                rgi_id != 'RGI60-11.01328') and (
                rgi_id != 'RGI60-11.03638') and (
                rgi_id != 'RGI60-11.03643') and (
                rgi_id != 'RGI60-11.00116') and (
                rgi_id != 'RGI60-11.00687') and (
                rgi_id != 'RGI60-11.00746') and (
                rgi_id != 'RGI60-11.01450') and (rgi_id != 'RGI60-11.01946'):
            #if (rgi_id != 'RGI60-11.03638') and (rgi_id != 'RGI60-11.03643') and (rgi_id != 'RGI60-11.01346') and (rgi_id != 'RGI60-11.02740') and (rgi_id != 'RGI60-11.02822') and (rgi_id != 'RGI60-11.02051'):
            continue
        """
        #if (rgi_id != 'RGI60-11.03646') and (rgi_id != 'RGI60-11.02755'):
        #    continue



        fig1, ax1 = plt.subplots(figsize=[21, 7])

        # plot observations
        #dfint = df.loc[:, 'obs'].astype('float').interpolate(
        #    method='linear').dropna()

        #uc = np.abs(np.array((df.loc[:, 'obs'].index - 2003) + 1))
        # uc = (np.abs(np.array(dfint.index - 2003)) + 1) * 1

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

        #from relic.postprocessing import detrend_series
        #obs_std = detrend_series(df.loc[:, 'obs']).std()
        #obs_std2 = df.loc[:, 'obs'].std()

        #ax1.fill_between(dfint.index, dfint - obs_std2, dfint + obs_std2, color='0.85')
        #ax1.fill_between(dfint.index, dfint - obs_std, dfint + obs_std, color='0.65')

        # get MAEs
        # get stdquot
        #stdquot = std_quotient(df, normalised=True).sort_values().iloc[:130]

        #stdmae = std_quotient(df.loc[:, np.append(maes.index.values, 'obs')],
        #                      normalised=True).sort_values().iloc[:13]

        """
        from relic.postprocessing import dl_last_smaller_detrendstd
        dllast = dl_last_smaller_detrendstd(df)

        from relic.postprocessing import rearfit5
        rf5 = rearfit5(df).sort_values()
        from relic.postprocessing import detrend_series
        obstd = detrend_series(df.obs).std()


        idx2plot = maes.index[:20]
        idx2plot2 = maes.index[maes < obstd]

        abser = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
            dropna().abs()

        a1k = abser.columns[(abser < 1000).all()]
        idx2plot = a1k

        maxabs = abser.max().sort_values()[:20].index

        a5 = abser.columns[(abser < 500).all()]
        idx2plot2 = maxabs
        """

        q10 = int(df.shape[1]/10)
        q3 = int(df.shape[1]/3)
        q1 = int(df.shape[1]/100)

        #pd5dt = pdic5dt[glid].sort_values()

        maes = mae_weighted(df, normalised=False).sort_values()
        idx2plot = maes.index[:q10]

        #from relic.postprocessing import maxerror
        #maxes = maxerror(df, normalised=False).sort_values()
        #idx2plot = maxes.index

        idx3 = maes.index[:q3]
        idx250 = maes.index[:250]
        #idx2plot2 = pdic5a[glid].sort_values().index[:10]
        #idx2plot3 = pdic5b[glid].sort_values().index[:10]

        from relic.postprocessing import fit_one_std, best_powerset, fit_one_std_2f, maxerror_smaller_than, maxerror, fit_one_std_1, fit_one_std_2g, fit_one_std_2h, pareto_nach_rye, pareto_nach_rye2

        # maer = maxerror(df).sort_values()
        # idx2plot3 = fit_one_std(df, pdic[glid].sort_values().index[:100], glid)
        # idx2plot2 = fit_one_std(df, maer.index[:100], glid)

        # ixuse = maes.index[:100][maes.index[:100].isin(maer.index[:100])]
        # idx2plot2, cov2 = fit_one_std_2(df.loc[:, ixuse], df.loc[:, 'obs'], glid)

        # idx2plot2, cov2 = fit_one_std_2b(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, cov=False, detrend=False, minuse=5)
        #idx2plot4, cov4 = fit_one_std_2c(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, cov=False, detrend=False,
        #                                 minuse=1, maxuse=1)

        # idx2plot2, cov2 = fit_one_std_2f(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)
        # idx2plot2, cov2 = fit_one_std_2f(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5, maxuse=8)
        # idx2plot2, cov2 = fit_one_std_2g(df.loc[:, idx2plot[:200]], df.loc[:, 'obs'], glid, minuse=5, maxuse=10)
        # idx2plot2, cov2 = fit_one_std_2g(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)

        # if glid not in ['RGI60-11.00746', 'RGI60-11.00887', 'RGI60-11.01270', 'RGI60-11.01946']:
        #     continue

        #if glid not in ['RGI60-11.02755']:
        #    continue
        # idx2plot2, cov = fit_one_std_2h(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)

        # idx2plot2, cov = fit_one_std_2g(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)


        # idx2plot2 = pareto_nach_rye2(df, glid)
        # idx2plot2 = pareto_nach_rye(df, glid)

        #from relic.postprocessing import montec
        #idx2plot2 = montec(df, glid)
        #useall.append(idx2plot2)

        from relic.postprocessing import fit_cov_and_skill
        # idx2plot2 = fit_cov_and_skill(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)
        idx2plot2 = fit_cov_and_skill(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5, maxuse=30)


        #me = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        #    dropna().mean()
        #mepl = me[me>0].sort_values().index[:25]
        #memi = me[me<0].sort_values(ascending=False).index[:25]
        #idx2plot2 = mepl.append(memi)

        from relic.postprocessing import r2
        #idx2plot2 = r2(df, normalised=True, detrend=True).sort_values().index[:50]

        # idx2plot2, cov = fit_one_std_2g(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5, maxuse=30)

        # idx2plot3, cov3 = fit_one_std_2b(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, cov=False, detrend=True)
        #idx2plot2, cov2 = fit_one_std_2b(df.loc[:, maes.index[:200]], df.loc[:, 'obs'], glid, cov=False)
        #idx2plot3, cov3 = fit_one_std_2b(df.loc[:, maes.index[:200]], df.loc[:, 'obs'], glid, cov=True)

        # idx2plot2 = best_powerset(df, mae_weighted(df).sort_values().index[:100])


        # best x runs as grey lines
        #nolbl = df.loc[:, idx2plot].rolling(y_len, center=True).mean().copy()
        #nolbl.columns = ['' for i in range(len(nolbl.columns))]
        #nolbl.plot(ax=ax1, linewidth=0.5, color='0.6')

        #df.loc[:, idx2plot2].rolling(y_len, center=True).mean().\
        #    plot(ax=ax1, linewidth=0.5)

        ensmean = df.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()
        ensmed = df.loc[:, idx2plot2].median(axis=1)

        ensstdmean = df.loc[:, idx2plot2].std(axis=1).rolling(y_len, center=True).mean()

        # coverage
        from relic.postprocessing import calc_coverage_2
        cov = calc_coverage_2(df, idx2plot2, df['obs'])
        # ensstd = ensmean.std()
        # ensstddt = detrend_series(ensmean).std()

        #ax1.fill_between(ensmeanmean.index, ensmeanmean - ensstdmean,
        #                 ensmeanmean + ensstdmean, color='C0', alpha=0.6)
        nolbl = df.loc[:, idx2plot2].rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]
        nolbl.plot(ax=ax1, linewidth=0.8, color='C0')

        ax1.plot(0, 0, color='C0', linewidth=10, label='ensemble mean +/- 1 std')


        #ax1.fill_between(ensmean.index, ensmean.values - ensstddt,
        #                 ensmean.values + ensstddt, color='C0', alpha=0.8)

        # plot ens members

        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='C1', label='ensemble mean')
        #ensmed.plot(ax=ax1, linewidth=1.5, color='C3', label='ensemble median')

        #nolbl = df.loc[:, idx2plot3].rolling(y_len, center=True).mean().copy()
        #nolbl.columns = ['' for i in range(len(nolbl.columns))]
        #nolbl.plot(ax=ax1, linewidth=0.8, color='m')

        # reference run (basically min mae)
        df.loc[:, idx2plot[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='C4')

        name = GLCDICT.get(rgi_id)[2]

        from relic.postprocessing import rmse_weighted
        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        rmse_ens = rmse_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]
        rmse_best = rmse_weighted(df, normalised=False).sort_values()[0]

        sprd = np.sqrt(df.loc[:, idx2plot2].var(axis=1).mean())

        rmspread = rmse_ens/sprd

        ax1.set_title('%s' % name, fontsize=28)
        ax1.text(2030, -2000, '%d ensemble members\n'
                              'coverage = %.2f\n'
                              'skill (RMSE / SPREAD) = %.2f\n'
                              'RMSE ensemble = %.2f\n'
                              'RMSE best = %.2f\n'
                              'MAE enselbe = %.2f\n'
                              'MAE best = %.2f' %
                 (len(idx2plot2), cov, rmspread, rmse_ens, rmse_best, mae_ens,
                  mae_best), fontsize=18)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=14)
        # fig1.subplots_adjust(right=0.7)
        fig1.tight_layout()
        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig1.savefig(fn1)

        used = {}
        used['oggmdefault'] = oggmdefault
        used['minmae'] = idx2plot[0]
        used['ensemble'] = idx2plot2

        pickle.dump(used, open(os.path.join(pout, 'runs_%s.p' % glid), 'wb'))


def quick_plot_and_hist(glcdict, pout, y_len=5):

    for glid, df in glcdict.items():

        # recalc for 1980
        ix1980 = df.dropna().index[0]
        df.loc[:, df.columns != 'obs'] += df.loc[ix1980, 'obs']

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        #if rgi_id != 'RGI60-11.03643':
        #    continue

        #if (rgi_id != 'RGI60-11.01450') and (rgi_id != 'RGI60-11.02051') and (rgi_id != 'RGI60-11.01270') and (rgi_id != 'RGI60-11.03643') and (rgi_id != 'RGI60-11.00897'):
        #    #if (rgi_id != 'RGI60-11.02755') and (rgi_id != 'RGI60-11.03646'):
        #    continue

        fig = plt.figure(figsize=[23, 9])

        from matplotlib.gridspec import GridSpec
        gs = GridSpec(3, 3)  # 2 rows, 3 columns

        ax1 = fig.add_subplot(gs[0:2, :])  # Second row, span all columns
        ax2 = fig.add_subplot(gs[2, 0])  # First row, first column
        ax3 = fig.add_subplot(gs[2, 1])  # First row, second column
        ax4 = fig.add_subplot(gs[2, 2])  # First row, third column


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

        q10 = int(df.shape[1] / 10)

        maes = mae_weighted(df, normalised=False).sort_values()
        idx2plot = maes.index[:q10]

        from relic.postprocessing import optimize_skill, optimize_skill2, optimize_cov, optimize_all, optimize_all2, optimize_cov2
        #idx2plot2 = fit_cov_and_skill2(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5, maxuse=25)
        idx2plot2 = optimize_cov2(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5)
        #idx2plot2 = optimize_cov2(df.loc[:, df.columns != 'obs'], df.loc[:, 'obs'], glid, minuse=5)
        #
        # idx2plot2 = maes.index[:20]

        """
        cov = skll = 0
        idx2plot2 = [maes.index[0]]
        n = 1
        while cov < 0.9:
            idx2plot2.append(maes.index[n])
            cov = calc_coverage_2(df, idx2plot2, df['obs'])

            ensmean = df.loc[:, idx2plot2].mean(axis=1)
            rmse_ens = rmse_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
            sprd = np.sqrt(df.loc[:, idx2plot2].var(axis=1).mean())
            skll = rmse_ens / sprd
            # normalise to 1
            skll = 1 / skll if skll > 1 else skll

            n += 1
            if n == len(maes):
                break
        """

        #mer = mean_error_weighted(df, normalised=False)
        #merp = mer[mer > 0].sort_values()
        #merm = mer[mer < 0].sort_values(ascending=False)
        #idx2plot2 = merm[:15].index.to_list() + merp[:15].index.to_list()

        from relic.postprocessing import calc_coverage_2, rmse_weighted

        ensmean = df.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()

        # coverage
        cov = calc_coverage_2(df, idx2plot2, df['obs'])
        nolbl = df.loc[:, idx2plot2].rolling(y_len, center=True).mean().copy()
        nolbl.columns = ['' for i in range(len(nolbl.columns))]
        nolbl.plot(ax=ax1, linewidth=0.8, color='C0')

        ax1.plot(0, 0, color='C0', linewidth=10,
                 label='ensemble mean +/- 1 std')

        # plot ens members
        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='C1',
                         label='ensemble mean')

        # reference run (basically min mae)
        df.loc[:, idx2plot[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='C4')

        name = GLCDICT.get(rgi_id)[2]

        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        #rmse_ens = rmse_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]
        #rmse_best = rmse_weighted(df, normalised=False).sort_values()[0]

        sprd = np.sqrt(df.loc[:, idx2plot2].var(axis=1).mean())

        rmspread = mae_ens / sprd

        ax1.set_title('%s' % name, fontsize=28)
        ax1.text(2030, -2500, '%d ensemble members\n'
                              'coverage = %.2f\n'
                              'skill (MAE / SPREAD) = %.2f\n'
                              'MAE enselbe = %.2f\n'
                              'MAE best = %.2f' %
                 (len(idx2plot2), cov, rmspread, mae_ens,
                  mae_best), fontsize=18)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=14)

        # histograms
        from collections import defaultdict
        glena = defaultdict(int)
        mbbias = defaultdict(int)
        prcpsf = defaultdict(int)

        for run in idx2plot2:
            para = ast.literal_eval('{' + run + '}')
            prcpsf[para['prcp_scaling_factor']] += 1
            glena[para['glena_factor']] += 1
            mbbias[para['mbbias']] += 1

        ax2.bar(list(glena.keys()), glena.values(), width=0.4)
        ax2.set_xlabel('Glen A factor')
        ax2.set_ylabel('# used in ensemble')
        ax2.set_xlim([0.5, 4.5])

        ax3.bar(list(prcpsf.keys()), prcpsf.values(), width=0.2)
        ax3.set_xlabel('Prcp SF factor')
        ax3.set_ylabel('# used in ensemble')
        ax3.set_xlim([0, 4.5])

        ax4.bar(list(mbbias.keys()), mbbias.values(), width=150)
        ax4.set_xlabel('MB bias')
        ax4.set_ylabel('# used in ensemble')
        ax4.set_xlim([-1600, 1200])


        # fig1.subplots_adjust(right=0.7)
        fig.tight_layout()
        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig.savefig(fn1)



def quick_crossval(glcdict, glcdict_1980, pout, y_len=5):

    for glid, df in glcdict.items():

        if glid == 'RGI60-11.02709_merged':
            continue
        if glid == 'RGI60-11.02051_merged':
            continue

        df80 = glcdict_1980[glid]
        # recalc for 1980
        ix1980 = df80.dropna().index[0]
        df80.loc[:, df80.columns != 'obs'] += df80.loc[ix1980, 'obs']

        fullobs = df.loc[:, 'obs']
        df = df.loc[:ix1980-1, df80.columns]

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        fig1, ax1 = plt.subplots(figsize=[21, 7])

        fullobs.plot(ax=ax1, color='k', marker='o',
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

        # get MAEs
        maes = mae_weighted(df, normalised=False).sort_values()

        q10 = int(df.shape[1]/10)

        idx2plot = maes.index

        from relic.postprocessing import fit_one_std_2g

        idx2plot2, cov = fit_one_std_2g(df.loc[:, idx2plot], df.loc[:, 'obs'], glid, minuse=5, maxuse=20)

        # -------- 1850 ----------
        ensmean = df.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean = ensmean.rolling(y_len, center=True).mean()

        ensstdmean = df.loc[:, idx2plot2].std(axis=1).rolling(y_len, center=True).mean()

        ax1.fill_between(ensmeanmean.index, ensmeanmean - ensstdmean,
                         ensmeanmean + ensstdmean, color='C0', alpha=0.6)
        ax1.plot(0, 0, color='C0', linewidth=10, label='ensemble mean +/- 1 std')

        ensmeanmean.plot(ax=ax1, linewidth=4.0, color='C1', label='ensemble mean')

        # reference run (basically min mae)
        df.loc[:, idx2plot[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='C4')

        # -------- 1980 ----------
        ensmean80 = df80.loc[:, idx2plot2].mean(axis=1)
        ensmeanmean80 = ensmean80.rolling(y_len, center=True).mean()

        ensstdmean80 = df80.loc[:, idx2plot2].std(axis=1).rolling(y_len, center=True).mean()

        ax1.fill_between(ensmeanmean80.index, ensmeanmean80 - ensstdmean80,
                         ensmeanmean80 + ensstdmean80, color='C2', alpha=0.6)
        ax1.plot(0, 0, color='C0', linewidth=10, label='ensemble mean +/- 1 std')

        ensmeanmean80.plot(ax=ax1, linewidth=4.0, color='C3', label='ensemble mean')

        # reference run (basically min mae)
        df80.loc[:, idx2plot[0]].rolling(y_len, center=True).mean(). \
            plot(ax=ax1, linewidth=3, color='C5')

        name = GLCDICT.get(rgi_id)[2]

        from relic.postprocessing import rmse_weighted
        mae_ens = mae_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        rmse_ens = rmse_weighted(pd.concat([ensmean, df['obs']], axis=1))[0]
        mae_best = maes[0]
        rmse_best = rmse_weighted(df, normalised=False).sort_values()[0]

        sprd = np.sqrt(df.loc[:, idx2plot2].var(axis=1).mean())

        rmspread = rmse_ens/sprd

        ax1.set_title('%s' % name, fontsize=28)
        ax1.text(2030, -2000, '%d ensemble members\n'
                              'coverage = %.2f\n'
                              'skill (RMSE / SPREAD) = %.2f\n'
                              'RMSE ensemble = %.2f\n'
                              'RMSE best = %.2f\n'
                              'MAE enselbe = %.2f\n'
                              'MAE best = %.2f' %
                 (len(idx2plot2), cov, rmspread, rmse_ens, rmse_best, mae_ens,
                  mae_best), fontsize=18)
        ax1.set_ylabel('relative length change [m]', fontsize=26)
        ax1.set_xlabel('Year', fontsize=26)
        ax1.set_xlim([1850, 2020])
        ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left', fontsize=14)
        # fig1.subplots_adjust(right=0.7)
        fig1.tight_layout()
        fn1 = os.path.join(pout, 'histalp_%s.png' % glid)
        fig1.savefig(fn1)


def quick_params(glcdict, pout):
    from relic.postprocessing import pareto4
    pdic4 = pareto4(glcdict)

    for glid, df in glcdict.items():

        # take care of merged glaciers
        rgi_id = glid.split('_')[0]

        fig1, ax1 = plt.subplots(figsize=[10, 10])

        # get MAEs
        # maes = mae_weighted(df, normalised=False).sort_values()
        maes = pdic4[glid]

        of = pd.DataFrame([], columns=['mbbias', 'prcp', 'glena', 'err'])

        for run in maes.index:
            # get calibration parameters
            par = ast.literal_eval('{' + run + '}')

            of = of.append({'glena': par['glena_factor'],
                            'prcp': par['prcp_scaling_factor'],
                            'mbbias': par['mbbias'],
                            'err': maes.loc[run]},
                           ignore_index=True)

        mbs = [-800, -200, 200]
        gas = [1, 2, 3]

        for mb, ga in itertools.product(mbs, gas):
            of.loc[(of.mbbias == mb) & (of.glena == ga), :].\
                plot(x='prcp', y='err', ax=ax1,
                     label='mb=%d, A=%.1f' % (mb, ga))

        name = GLCDICT.get(rgi_id)[2]

        ax1.set_title('%s' % name, fontsize=30)
        ax1.set_ylabel('error', fontsize=26)
        ax1.set_xlabel('prcp', fontsize=26)
        #ax1.set_xlim([1850, 2020])
        #ax1.set_ylim([-3500, 1000])
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.grid(True)

        ax1.legend(bbox_to_anchor=(1.04, 1), loc='upper left')
        # fig1.subplots_adjust(right=0.7)
        fig1.tight_layout()
        fn1 = os.path.join(pout, 'params_%s.png' % glid)
        fig1.savefig(fn1)
