import numpy as np
import pandas as pd
import ast
import os

import pickle

from relic.length_observations import get_length_observations
from relic.preprocessing import GLCDICT


def calc_acdc(_obs, spinup, model, meta, col):

    obs = _obs.dropna().copy()

    try:
        spin = (spinup.loc[:, col] - spinup.loc[0, col]).dropna().iloc[-1][0]
    except IndexError:
        return pd.Series([np.nan])

    dl = spin + meta['dL2003'].iloc[0]
    # relative length change
    hist = model.loc[:, col] - model.loc[:, col].iloc[0] + dl

    # only look at points with measurements
    hist = hist.loc[obs.index].squeeze()

    # accumulated difference change
    acdc = (obs-hist).diff().abs().cumsum()
    acdc.iloc[0] = 0

    return acdc


def relative_length_change(meta, spinup, histrun):
    spin = (spinup.loc[:] - spinup.loc[0]).dropna().iloc[-1]
    dl = spin + meta['dL2003']
    # relative length change
    rel_dl = histrun.loc[:] - histrun.iloc[0] + dl

    return rel_dl


def mae(obs, model):
    return np.mean(np.abs(obs-model).dropna())


def max_error(df, amax=200, detrend=False, normalised=False):
    maxe = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
        dropna().abs().max().sort_values()

    # maxedf = df.loc[:, maxe.loc[maxe < amax].index]
    maxeidx = maxe.loc[maxe < amax].index
    if len(maxeidx) < 2:
        maxeidx = maxe.index[:50]
    return maxeidx


def mae_all(df, normalised=False):
    maeall = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
        dropna().abs().mean()

    if normalised:
        # return maeall/maeall.max()
        return (maeall - maeall.min())/(maeall.max()-maeall.min())
    else:
        return maeall


def std_quotient(df, normalised=False):
    # make sure only to calculate where both obs and run are available
    # take any index
    runidx = df.loc[:, df.columns[df.columns != 'obs'][0]].dropna().index
    obsidx = df.loc[:, 'obs'].dropna().index
    idx = np.intersect1d(runidx, obsidx)

    stdall = df.loc[idx].std()

    stdquot = (stdall.loc[stdall.index != 'obs'] / stdall.loc['obs'])

    if normalised:
        # bring it relative to 1
        stqn = np.abs(1-stdquot)
        return (stqn - stqn.min())/(stqn.max()-stqn.min())
    else:
        return stdquot


def r2(df, normalised=True):
    df['obs'] = df['obs'].astype(float)
    corr2obs = df.corr()['obs']

    corr = corr2obs[corr2obs.index != 'obs']

    if normalised:
        corn = 1 - corr
        assert np.all(corn > 0)
        assert np.all(corn < 2)
        return (corn - corn.min())/(corn.max()-corn.min())
    else:
        return corr


def mean_error_weighted(df, normalised=False):

    # calculate ME, without mean first
    me = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).dropna()

    # index of observations
    oix = me.index

    # observation gaps
    doix = np.diff(oix)

    # weight as size of gap in both directions
    wgh = np.zeros_like(oix, dtype=float)
    wgh[0:-1] = doix/2
    wgh[1:] = doix/2
    # and a minimum weight of 1
    wgh = np.maximum(wgh, 1)

    # now apply weight
    meall = me.mul(wgh, axis=0)
    # and take mean
    meall = meall.mean()

    if normalised:
        # return maeall/maeall.max()
        return (meall - meall.min())/(meall.max()-meall.min())
    else:
        return meall


def mae_weighted(df, normalised=False):

    # calculate MAE, without mean first
    maeall = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        dropna().abs()

    # index of observations
    oix = maeall.index

    # observation gaps
    doix = np.diff(oix)

    # weight as size of gap in both directions
    wgh = np.zeros_like(oix, dtype=float)
    wgh[0:-1] = doix/2
    wgh[1:] = doix/2
    # and a minimum weight of 1
    wgh = np.maximum(wgh, 1)

    # now apply weight
    maeall = maeall.mul(wgh, axis=0)
    # and take mean
    maeall = maeall.mean()

    if normalised:
        # return maeall/maeall.max()
        return (maeall - maeall.min())/(maeall.max()-maeall.min())
    else:
        return maeall


def mae_diff_mean(df, yr, normalised=False):
    # MAE of XX year difference
    df_dt = df.copy()
    df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
    rundif = df_dt.resample("%dY" % yr).mean().diff()

    maediff = rundif.loc[:, rundif.columns != 'obs']. \
        sub(rundif.loc[:, 'obs'], axis=0).dropna().abs().mean()

    if normalised:
        return (maediff - maediff.min())/(maediff.max()-maediff.min())
    else:
        return maediff


def mae_diff_yearly(df, yr, normalised=False):
    # MAE of XX year difference
    df_dt = df.copy()
    df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
    # rundif = df_dt.resample("%dY" % yr).mean().diff()
    # rundif = df.diff(yr).copy()
    rundif = df.copy() * np.nan

    obsyrs = pd.Series(df.obs.dropna().index)
    for y1 in obsyrs.iloc[:-yr]:
        y2 = obsyrs[((obsyrs-y1-yr) >= 0)].iloc[0]
        rundif.loc[y1, :] = df.loc[y2, :] - df.loc[y1, :]

    maediff = rundif.loc[:, rundif.columns != 'obs'].\
        sub(rundif.loc[:, 'obs'], axis=0).dropna().abs().mean()

    if normalised:
        return (maediff - maediff.min())/(maediff.max()-maediff.min())
    else:
        return maediff


def diff_corr(df, yr=10, normalised=False):
    # MAE of XX year difference
    rundif = df.copy() * np.nan

    obsyrs = pd.Series(df.obs.dropna().index)
    for y1 in obsyrs.iloc[:-yr]:
        y2 = obsyrs[((obsyrs-y1-yr) >= 0)].iloc[0]
        rundif.loc[y1, :] = df.loc[y2, :] - df.loc[y1, :]

    corre = rundif.dropna().corr()['obs'].loc[~rundif.columns.isin(['obs'])]

    if normalised:
        return 1 - (corre - corre.min())/(corre.max() - corre.min())
    else:
        return corre


def dummy_dismantel_multirun():
    # just to remember
    import ast
    # rvaldict keys are strings, transform to dict with
    # combinationdict=ast.literal_eval(rvaldictkey)


def runs2df(runs, glenamin=0):

    # get all glaciers
    glcs = []
    for run in runs:
        glcs += [gl['rgi_id'] for gl in list(run.values())[0]]
    glcs = np.unique(glcs).tolist()

    # subset
    glcs = ['RGI60-11.00746',
            'RGI60-11.00897_merged',
            'RGI60-11.01270',
            'RGI60-11.01450_merged',
            'RGI60-11.01946',
            'RGI60-11.02051_merged',
            'RGI60-11.02740',
            'RGI60-11.03638',
            'RGI60-11.03643_merged',
            'RGI60-11.03646']

# take care of merged ones
    rgi_ids = [gl.split('_')[0] for gl in glcs]

    meta, data = get_length_observations(rgi_ids)

    # store results per glacier in a dict
    glcdict = {}

    for rgi, mrgi in zip(rgi_ids, glcs):
        _meta = meta.loc[rgi].copy()
        _data = data.loc[rgi].copy()

        df = pd.DataFrame([], index=np.arange(1850, 2020))
        df.loc[_data.index, 'obs'] = _data

        """
        if 'XXX_merged' in mrgi:
            # mid = merged_ids(mrgi)

            _mmeta = meta.loc[meta['RGI_ID'] == mid].copy()
            _mdata = data.loc[_mmeta.index[0]].copy()
            dfmerge = pd.DataFrame([], index=np.arange(1850, 2011))
            dfmerge.loc[_mdata.index, 'obs'] = _mdata
        """

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == mrgi][0]
            except IndexError:
                continue

            if np.isnan(rdic['tbias']):
                continue

            rkey = list(run.keys())[0]
            para = ast.literal_eval('{' + rkey + '}')

            if para['glena_factor'] < glenamin:
                continue

#            if para['glena_factor'] > 3:
#                continue

            """
            if para['mbbias'] < -800:
                continue
            if para['mbbias'] > 800:
                continue

            if para['prcp_scaling_factor'] < 1:
                continue
            if para['prcp_scaling_factor'] > 3:
                continue
            """

            #if not np.isclose(para['prcp_scaling_factor'], 1.75, atol=0.01):
            #    continue

            df.loc[rdic['rel_dl'].index, rkey] = rdic['rel_dl']

            """
            if 'XXX_merged' in glid:
                dfmerge.loc[rdic['trib_dl'].index, rkey] = rdic['trib_dl']
            """

        glcdict[mrgi] = df
        """
        if 'XXX_merged' in glid:
            glcdict[mid] = dfmerge
        """

    return glcdict


def rearfit(df, normalised=False):
    # this is basically the MAE of the last x years:
    # get the modelruns last indices
    moix = df.loc[:, df.columns != 'obs'].dropna().index[-10:]

    mae10 = mae_weighted(df.loc[moix], normalised=normalised)

    return mae10


def mean_error_pm(glcdict, n=15):
    medict = {}

    for glc, df in glcdict.items():
        # get the smallest MAEs
        mes = mean_error_weighted(glcdict[glc], normalised=False)

        meplus = mes[mes >= 0].sort_values()[:n]
        meminus = mes[mes < 0].sort_values(ascending=False)[:n]

        medict[glc] = meplus.append(meminus).index

    return medict


def pareto3(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():


        """
        # get them in a normaliesed way
        maes = mae_weighted(glcdict[glc].loc[:,
                            np.append(df.index, 'obs')],
                            # np.append(_maes.index.values, 'obs')],
                            normalised=True).sort_values()
        stdq = std_quotient(glcdict[glc].loc[:,
                            np.append(df.index, 'obs')],
                            normalised=True).sort_values()
        """


        """
        # get the smallest MAEs
        _maes = mae_weighted(glcdict[glc], normalised=False).sort_values().iloc[:1300]
        
        maes = mae_weighted(df.loc[:, np.append(_maes.index, 'obs')],
                            normalised=True)

        stdq = std_quotient(df.loc[:, np.append(_maes.index, 'obs')],
                            normalised=True)

        rf = rearfit(df.loc[:, np.append(_maes.index, 'obs')],
                     normalised=True)

        cor = r2(df.loc[:, np.append(_maes.index, 'obs')],
                 normalised=True)
        """

        maes = mae_weighted(df, normalised=True)

        stdq = std_quotient(df, normalised=True)

        rf = rearfit(df, normalised=True)

        cor = r2(df, normalised=True)


        # same for the stdq
        # stdq = std_quotient(glcdict[glc].loc[:,
        #                    np.append(_maes.index.values, 'obs')],
        #                    normalised=True)

        # utopian, is 0/0/0 anyway...
        up = [maes.min(), stdq.min(), cor.min(), rf.min()]

        # euclidian dist
        edisx = np.sqrt(5*((maes - up[0]) ** 2) +
                        (stdq - up[1]) ** 2 +
                        (cor - up[2]) ** 2 +
                        (rf - up[3]) ** 2).sort_values()

        # paretodict[glc] = edisx.index
        paretodict[glc] = edisx.loc[edisx < edisx.quantile(0.05)].index

    return paretodict


def paretoX(glcdict, x=20):
    paretodict = {}

    for glc in glcdict.keys():

        # get the smallest MAEs
        _maes = mae_weighted(glcdict[glc], normalised=False).sort_values()

        dfstat = pd.DataFrame([], columns=['prcp', 'glena', 'mbbias',
                                           'mae', 'stdq'])

        for run in _maes.index:
            para = ast.literal_eval('{' + run + '}')

            dfstat.loc[run, 'prcp'] = para['prcp_scaling_factor']
            dfstat.loc[run, 'glena'] = para['glena_factor']
            dfstat.loc[run, 'mbbias'] = para['mbbias']
            dfstat.loc[run, 'mae'] = _maes.loc[run]


        dfstat = dfstat.astype(float)
        #mb0 = dfstat.groupby('mbbias').median()['mae'].sort_values().index[0]
        #dfstat = dfstat.loc[(dfstat['mbbias'] == mb0)] # |
                            #(dfstat['mbbias'] == mb0 + 200) |
                            #(dfstat['mbbias'] == mb0 - 200)]

        mbmin = dfstat.groupby('mbbias').median()['mae'].min()
        dfstat = dfstat.loc[dfstat['mae'] < mbmin]

        # get them in a normaliesed way
        maes = mae_weighted(glcdict[glc].loc[:,
                            np.append(dfstat.index, 'obs')],
                            # np.append(_maes.index.values, 'obs')],
                            normalised=True).sort_values()

        stdq = std_quotient(glcdict[glc].loc[:,
                            np.append(dfstat.index, 'obs')],
                            normalised=True).sort_values()


        # same for the stdq
        #stdq = std_quotient(glcdict[glc].loc[:,
        #                    np.append(_maes.index.values, 'obs')],
        #                    normalised=True)

        # utopian
        up = [maes.min(), stdq.min()]

        # euclidian dist
        edisx = np.sqrt((maes-up[0])**2 + (stdq-up[1])**2).sort_values()

        paretodict[glc] = edisx.index

        #plot_pareto(glc, edisx.index[:x], maes, stdq)

    return paretodict



def pareto(glcdict, pout):
    paretodict = {}

    for glc in glcdict.keys():

        # get my measures
        # maes = mae_all(glcdict[glc], normalised=True)
        # maediff = mae_diff_yearly(glcdict[glc], maedyr, normalised=True)
        # corre = diff_corr(glcdict[glc], yr=maedyr, normalised=True)

        # stdq = std_quotient(glcdict[glc], normalised=True)
        # maes = mae_weighted(glcdict[glc], normalised=True)

        # get the 15 smallest MAEs
        _maes = mae_weighted(glcdict[glc], normalised=False).sort_values().iloc[:130]
        # get them in a normaliesed way
        maes = mae_weighted(glcdict[glc].loc[:,
                            np.append(_maes.index.values, 'obs')],
                            normalised=True)
        # same for the stdq
        stdq = std_quotient(glcdict[glc].loc[:,
                            np.append(_maes.index.values, 'obs')],
                            normalised=True)

        # utopian
        up = [maes.min(), stdq.min()]
        # up = [maes.min(), maediff.min(), corre.min()]

        # euclidian dist
        # TODO pareto weight
        pwgh = 1
        # pwgh = 1
        edisx = np.sqrt(pwgh*(maes-up[0])**2 +
                        (stdq-up[1])**2).idxmin()
        #                (corre-up[2])**2).idxmin()

        paretodict[glc] = edisx

        # plot_pareto(glc, edisx, maes, stdq)

    if pout is not None:
        paretopth = os.path.join(pout, 'pareto.p')
        pickle.dump(paretodict, open(paretopth, 'wb'))

    return paretodict


def pareto2(stats, glcdict):

    paretodict = {}

    for glc, df in glcdict.items():
        dfs = stats[glc].astype(float)

        mb0 = dfs.groupby('mbbias').median()['mae'].sort_values().index[0]
        dfs = dfs.loc[(dfs['mbbias'] == mb0) |
                      (dfs['mbbias'] == mb0 + 200) |
                      (dfs['mbbias'] == mb0 - 200)]

        # get the 15 smallest MAEs
        #_maes = mae_weighted(df.loc[:, np.append(dfs.index, 'obs')],
        #                     normalised=False).sort_values().iloc[:78]

        # get them in a normaliesed way
        maes = mae_weighted(df.loc[:, np.append(dfs.index, 'obs')],
                            normalised=True)

        # same for the stdq
        stdq = std_quotient(df.loc[:, np.append(dfs.index, 'obs')],
                            normalised=True)

        # utopian
        up = [maes.min(), stdq.min()]

        edisx = np.sqrt((maes-up[0])**2 +
                        (stdq-up[1])**2).idxmin()

        paretodict[glc] = edisx

        plot_pareto(glc, edisx, maes, stdq)

    return paretodict


def plot_pareto(glc, edisx, maes, maediff):
    import matplotlib.pyplot as plt
    import ast
    import os
    from colorspace import diverging_hcl, sequential_hcl
    from matplotlib.colors import ListedColormap

    fig1, ax1 = plt.subplots(figsize=[15, 8])
    rgi_id = glc.split('_')[0]
    name = GLCDICT.get(rgi_id)[2]

    ax1.plot(0, 0, '*k', markersize=20, label='utopian solution')

    if isinstance(edisx, list):
        print(name)
        for run in edisx:
            ax1.plot(maes.loc[run], maediff.loc[run], '*r', color='C2',
                     markersize=10, markeredgecolor='C3',
                     label='')
            print(run)
        edisx = edisx[0]

    ax1.plot(maes.loc[edisx], maediff.loc[edisx], '*r', color='C2',
             markersize=25, markeredgecolor='C3',
             label='choosen as best solution')

    ax1.plot(maes.loc[maes.idxmin()], maediff.loc[maes.idxmin()],
             '*r', color='C0', markersize=20,
             label='best result for Objective 1')
    ax1.plot(maes.loc[maediff.idxmin()], maediff.loc[maediff.idxmin()],
             '*r', color='C1', markersize=20,
             label='best result for Objective 2')

    df = pd.DataFrame([], columns=['mae', 'maedif', 'mb', 'prcp'])

    # prcp
    cmap = diverging_hcl(h=[260, 0], c=80, l=[30, 90], power=2).cmap(n=18)
    cmap = cmap(np.arange(cmap.N))
    cmap[:, -1] = np.append(np.linspace(1, 0.5, 9)**2, np.linspace(0.5, 1, 9)**2)
    prcpcmp = ListedColormap(cmap)

    # mbbias
    cmap = diverging_hcl(h=[130, 43], c=100, l=[70, 90], power=1.5).cmap(n=8)
    cmap = cmap(np.arange(cmap.N))
    cmap[:, -1] = np.append(np.linspace(1, 0.5, 4)**2, np.linspace(0.5, 1, 4)**2)
    mbbcmp = ListedColormap(cmap)

    # glena
    cmap = sequential_hcl('Blue-Yellow').cmap(8)(np.arange(8))
    glacmp = ListedColormap(cmap)

    # prcp2
    cmap = sequential_hcl('Heat').cmap(16)(np.arange(16))
    prcpcmp2 = ListedColormap(cmap)

    for run in maes.index:
        prm = ast.literal_eval('{' + run + '}')

        df = df.append({'mae': maes.loc[run],
                        'maedif': maediff.loc[run],
                        'mb': prm['mbbias'],
                        'prcp': prm['prcp_scaling_factor'],
                        'glena': prm['glena_factor']},
                       ignore_index=True)

        """
        plt.plot(maes.loc[run], maediff.loc[run], 'ok',
                 color=mbdicc[prm['mbbias']],
                 alpha=mbdica[prm['mbbias']])

        plt.plot(maes.loc[run], maediff.loc[run], 'ok',
                 color=pcdicc[prm['prcp_scaling_factor']],
                 alpha=pcdica[prm['prcp_scaling_factor']])
        """

    #sc2 = ax1.scatter(df.mae, df.maedif, c=df.mb, cmap=mbbcmp, label='')
    sc2 = ax1.scatter(df.mae, df.maedif, c=df.glena, cmap=glacmp.reversed(), label='',
                      s=100)
    sc1 = ax1.scatter(df.mae, df.maedif, c=df.prcp, cmap=prcpcmp2.reversed(),
                      label='', s=25)

    cx1 = fig1.add_axes([0.71, 0.38, 0.05, 0.55])
    cb1 = plt.colorbar(sc1, cax=cx1)
    cb1.set_label('Precipitation scaling factor', fontsize=16)

    cx2 = fig1.add_axes([0.85, 0.38, 0.05, 0.55])
    cb2 = plt.colorbar(sc2, cax=cx2)
    cb2.set_label('Glen A Factor', fontsize=16)

    ax1.set_title(name, fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=22)

    ax1.set_ylabel('STD quotient (normalised)',
                   fontsize=26)
    ax1.set_xlabel('MAE of relative length change (normalised)', fontsize=26)

    ax1.legend(bbox_to_anchor=(1.04, 0),
               fontsize=18, loc="lower left", ncol=1)

    ax1.grid(True)
    fig1.tight_layout()
    pout = '/home/matthias/length_change_1850/neu/'
    fn1 = os.path.join(pout, 'pareto_%s.png' % glc)
    # fn2 = os.path.join(pout, 'pareto_%s.pdf' % name.split()[0])
    fig1.savefig(fn1)
    # fig1.savefig(fn2)
