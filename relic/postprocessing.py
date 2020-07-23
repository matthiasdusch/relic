import numpy as np
import pandas as pd
import ast
import os
from copy import deepcopy

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


def _mae(obs, model):
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


def std_quotient(_df, normalised=False, detrend=False):

    df = deepcopy(_df)

    if detrend:
        for x in df.columns:
            df.loc[:, x] = detrend_series(df.loc[:, x])

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


def r2(_df, normalised=False, detrend=False):

    df = deepcopy(_df)

    if detrend:
        for x in df.columns:
            df.loc[:, x] = detrend_series(df.loc[:, x])

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

    # now apply weight and mean
    meall = me.mul(wgh, axis=0).sum() / sum(wgh)

    if normalised:
        # return maeall/maeall.max()
        return (meall - meall.min())/(meall.max()-meall.min())
    else:
        return meall


def rmse_weighted(_df, normalised=False, weighted=True):

    df = deepcopy(_df)

    # calculate squared error, without mean first
    rmseall = ((df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0))**2).dropna()

    # index of observations
    oix = rmseall.index

    # observation gaps
    doix = np.diff(oix)

    # weight as size of gap in both directions
    wgh = np.zeros_like(oix, dtype=float)
    wgh[0:-1] = doix/2
    wgh[1:] += doix/2
    # and a minimum weight of 1
    wgh = np.maximum(wgh, 1)

    # now apply weight and mean
    if weighted:
        rmseall = rmseall.mul(wgh, axis=0).sum()/sum(wgh)
    else:
        rmseall = rmseall.mean()

    # and take mean and root
    rmseall = np.sqrt(rmseall)

    if normalised:
        # return maeall/maeall.max()
        return (rmseall - rmseall.min())/(rmseall.max()-rmseall.min())
    else:
        return rmseall


def mae_weighted(_df, normalised=False, detrend=False, weighted=True):

    df = deepcopy(_df)

    if detrend:
        for x in df.columns:
            df.loc[:, x] = detrend_series(df.loc[:, x])

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
    wgh[1:] += doix/2
    # and a minimum weight of 1
    wgh = np.maximum(wgh, 1)

    # now apply weight and mean
    if weighted:
        maeall = maeall.mul(wgh, axis=0).sum()/sum(wgh)
    else:
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

    """
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
    """

# take care of merged ones
    rgi_ids = [gl.split('_')[0] for gl in glcs]

    meta, data = get_length_observations(rgi_ids)

    # store results per glacier in a dict
    glcdict = {}

    tbiasdict = {}

    for rgi, mrgi in zip(rgi_ids, glcs):
        _meta = meta.loc[rgi].copy()
        _data = data.loc[rgi].copy()

        df = pd.DataFrame([], index=np.arange(1850, 2020))
        df.loc[_data.index, 'obs'] = _data

        tbias_series = pd.Series([])

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

            tbias_series[rkey] = rdic['tbias']

            """
            if 'XXX_merged' in glid:
                dfmerge.loc[rdic['trib_dl'].index, rkey] = rdic['trib_dl']
            """

        glcdict[mrgi] = df
        tbiasdict[mrgi] = tbias_series
        """
        if 'XXX_merged' in glid:
            glcdict[mid] = dfmerge
        """

    return glcdict, tbiasdict


def rearfit(df, normalised=False):
    # this is basically the MAE of the last x years:
    # get the modelruns last indices
    moix = df.loc[:, df.columns != 'obs'].dropna().index[-10:]

    mae10 = mae_weighted(df.loc[moix], normalised=normalised)

    return mae10


def rearfit5(df, normalised=False):
    # get the modelruns last indices
    dllast = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        dropna().iloc[-5:].abs().max()

    if normalised:
        # return maeall/maeall.max()
        return (dllast - dllast.min())/(dllast.max()-dllast.min())
    else:
        return dllast


def maxerror(df, normalised=False):
    # get the modelruns last indices
    maer = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
        dropna().abs().max()

    if normalised:
        # return maeall/maeall.max()
        return (maer - maer.min())/(maer.max()-maer.min())
    else:
        return maer


def cumerror(df, normalised=False):
    # accumulated difference change
    car = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        dropna().diff().abs().cumsum(axis=0).iloc[-1]

    if normalised:
        # return maeall/maeall.max()
        raise NotImplementedError
    else:
        return car


def maxerror_smaller_than(df, maxer=None):
    if maxer is None:
        # use std of obs
        maxer = df.loc[:, 'obs'].std()

    # get the max error
    maer = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        dropna().abs().max()

    return maer[maer < maxer]


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
        paretodict[glc] = edisx.loc[edisx < edisx.quantile(0.01)].index

    return paretodict


def polyfit_coef(df, normalised=False, detrend=True):

    dfx = deepcopy(df)
    dfx['obs'] = dfx['obs'].astype(float)

    if detrend:
        for x in dfx.columns:
            dfx.loc[:, x] = detrend_series(dfx.loc[:, x])

    pf = pd.DataFrame([], columns=['poly'], index=dfx.columns[dfx.columns != 'obs'])

    oix = df.dropna().index

    for x, _ in pf.iterrows():
        pf.loc[x, 'poly'] = np.polynomial.polynomial.polyfit(
            dfx.loc[oix, x].values, dfx.loc[oix, 'obs'].values, 1)[1]

    pf = pf['poly'].astype(float)

    if normalised:
        npf = (1 - pf).abs()
        return (npf - npf.min())/(npf.max()-npf.min())
    else:
        return pf


def pareto_simple(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():
        maes = mae_weighted(df, normalised=True, detrend=False)
        stdq = std_quotient(df, normalised=True, detrend=True)
        # euclidian dist
        edisx = np.sqrt(1 * maes**2 +
                        1 * stdq**2).sort_values()
        paretodict[glc] = edisx
    return paretodict


def pareto_5dta(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():

        maes = mae_weighted(df, normalised=True, detrend=False)
        # stdq = std_quotient(df, normalised=True)
        pf = polyfit_coef(df, normalised=True, detrend=False)
        # rf = rearfit5(df, normalised=True)
        cor = r2(df, normalised=True, detrend=True)
        maxer = maxerror(df, normalised=True)

        # euclidian dist
        edisx = np.sqrt(1 * maes**2 +
                        # 1 * stdq**2 +
                        1 * cor**2 +
                        1 * pf**2 +
                        # 1 * rf**2 +
                        1 * maxer**2
                        ).sort_values()

        paretodict[glc] = edisx

    return paretodict


def pareto_5dtb(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():

        maes = mae_weighted(df, normalised=True, detrend=False)
        # stdq = std_quotient(df, normalised=True)
        pf = polyfit_coef(df, normalised=True, detrend=False)
        # rf = rearfit5(df, normalised=True)
        cor = r2(df, normalised=True, detrend=True)
        # maxer = maxerror(df, normalised=True)

        # euclidian dist
        edisx = np.sqrt(2 * maes**2 +
                        # 1 * stdq**2 +
                        1 * cor**2 +
                        1 * pf**2
                        # 1 * rf**2 +
                        # 1 * maxer**2
                        ).sort_values()

        paretodict[glc] = edisx

    return paretodict


def pareto_5(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():

        maes = mae_weighted(df, normalised=True, detrend=False)
        # stdq = std_quotient(df, normalised=True)
        pf = polyfit_coef(df, normalised=True, detrend=False)
        # rf = rearfit5(df, normalised=True)
        cor = r2(df, normalised=True, detrend=False)
        # maxer = maxerror(df, normalised=True)

        # euclidian dist
        edisx = np.sqrt(1 * maes**2 +
                        # 1 * stdq**2 +
                        1 * cor**2 +
                        1 * pf**2
                        # 1 * rf**2 +
                        # 1 * maxer**2
                        ).sort_values()

        paretodict[glc] = edisx

    return paretodict


def pareto4(glcdict):
    paretodict = {}

    for glc, df in glcdict.items():

        maes = mae_weighted(df, normalised=True)

        stdq = std_quotient(df, normalised=True)

        rf = rearfit5(df, normalised=True)

        cor = r2(df, normalised=True)

        maxer = maxerror(df, normalised=True)

        # same for the stdq
        # stdq = std_quotient(glcdict[glc].loc[:,
        #                    np.append(_maes.index.values, 'obs')],
        #                    normalised=True)

        # utopian, is 0/0/0 anyway...

        # euclidian dist
        edisx = np.sqrt(1 * maes**2 +
                        #1 * stdq**2 +
                        1 * cor**2 +
                        #1 * rf**2 +
                        1 * maxer**2).sort_values()

        paretodict[glc] = edisx
        # paretodict[glc] = edisx.loc[edisx < edisx.quantile(0.01)].index

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


def detrend_series(series):
    from sklearn.linear_model import LinearRegression

    x = series.dropna().index.values.reshape(-1, 1)
    y = series.dropna().values.reshape(-1, 1)

    model = LinearRegression()
    model.fit(x, y)

    x_pred = series.index.values.reshape(-1, 1)
    pred = model.predict(x_pred)

    return series - pred.reshape(-1)


def dl_last_smaller_detrendstd(df):
    # get std of detrend observations
    dtstd = detrend_series(df.obs).std()

    # delta of last observation
    dllast = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
        dropna().iloc[-1].abs()

    return dllast.index[dllast < dtstd]


def old_rmse(df):

    # calculate MAE, without mean first
    return (df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).
            dropna()**2).mean().sqrt()


def coverage_loop2(df, use, obs, normalised=False):
    ' USE THE MEDIAN'

    coverage = pd.Series()

    for col, val in df.iteritems():

        # don't use anything twice
        if col in use:
            continue

        coverage[col] = calc_coverage_3(df, use + [col], obs)

    if normalised:
        covn = 1 - coverage
        if (covn == 0).all():
            # if all are 0 (= max coverage) return her and dont divde 0
            return covn

        try:
            assert np.all(covn >= 0)
            assert np.all(covn <= 1)
        except:
            print('whatsgoingon')
        return (covn - covn.min())/(covn.max()-covn.min())
    else:
        return coverage


def coverage_loop(df, use, obs, normalised=False):

    coverage = pd.Series()

    for col, val in df.iteritems():

        # don't use anything twice
        if col in use:
            continue

        nucov = calc_coverage_2(df, use + [col], obs)
        coverage[col] = nucov

    if normalised:
        covn = 1 - coverage
        if (covn == 0).all():
            # if all are 0 (= max coverage) return her and dont divde 0
            return covn

        try:
            assert np.all(covn >= 0)
            assert np.all(covn <= 1)
        except:
            pass
            # print('whatsgoingon')
        # return (covn - covn.min())/(covn.max()-covn.min())
        return covn / covn.max()
    else:
        return coverage


def calc_coverage(df, use, obs, detrend=False):

    # ensemble mean
    ens = df.loc[:, use].mean(axis=1)

    # use observation std
    """
    obs_std = detrend_series(obs).std()
    obs_p1 = obs + obs_std
    obs_m1 = obs - obs_std
    cover = ((obs_p1 > ens.loc[obs.index]) &
             (obs_m1 < ens.loc[obs.index])).sum()/len(obs.index)
    """

    # use ensemble std
    if detrend:
        dt_std = detrend_series(ens).std()
    else:
        dt_std = ens.std()

    ens_p1 = ens + dt_std
    ens_m1 = ens - dt_std

    # good index
    oix = ens.loc[obs.dropna().index].dropna().index

    cover = ((obs.loc[oix] > ens_m1.loc[oix]) &
             (obs.loc[oix] < ens_p1.loc[oix])).sum() / len(oix)

    return cover


def calc_coverage_3(df, use, obs):

    # ensemble QUANTILES
    enq25 = df.loc[:, use].quantile(0.25, axis=1)
    enq75 = df.loc[:, use].quantile(0.75, axis=1)

    # good index
    oix = enq25.loc[obs.dropna().index].dropna().index

    cover = ((obs.loc[oix] > enq25.loc[oix]) &
             (obs.loc[oix] < enq75.loc[oix])).sum()/len(oix)

    return cover


def calc_coverage_2(df, use, obs):

    # ensemble mean
    ensmea = df.loc[:, use].mean(axis=1)
    # ensemble std
    ensstd = df.loc[:, use].std(axis=1)

    ens_p1 = ensmea + ensstd
    ens_m1 = ensmea - ensstd

    # good index
    oix = ensmea.loc[obs.dropna().index].dropna().index

    cover = ((obs.loc[oix] > ens_m1.loc[oix]) &
             (obs.loc[oix] < ens_p1.loc[oix])).sum()/len(oix)

    return cover


def mae_coverage(df, use, obs, normalised=False):
    maecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)
        # calculate MAE, without mean first
        maec = ens.sub(obs, axis=0).dropna().abs().mean()
        maecover[col] = maec
    if normalised:
        return (maecover - maecover.min())/(maecover.max()-maecover.min())
    else:
        return maecover


def mae_coverage_3(df, use, obs, normalised=False, weighted=True):
    # MEDIAN
    maecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].median(axis=1)

        maecover[col] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=weighted)[0]

    if normalised:
        return (maecover - maecover.min())/(maecover.max()-maecover.min())
    else:
        return maecover


def mae_coverage_2(df, use, obs, normalised=False, weighted=True):
    maecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        maecover[col] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=weighted)[0]

    if normalised:
        # return (maecover - maecover.min())/(maecover.max()-maecover.min())
        return maecover / maecover.max()
    else:
        return maecover


def maxerror_coverage(df, use, obs, normalised=False):
    maxe = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        maxe[col] = maxerror(pd.concat([obs, ens], axis=1), normalised=False)[0]

    if normalised:
        return maxe / maxe.max()
    else:
        return maxe


def rmse_coverage_3(df, use, obs, normalised=False, weighted=False):
    # MEDIAN
    rmsecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        rmsecover[col] = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=weighted)[0]

    if normalised:
        return (rmsecover - rmsecover.min())/(rmsecover.max()-rmsecover.min())
    else:
        return rmsecover


def rmse_coverage_2(df, use, obs, normalised=False, weighted=False):
    rmsecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        rmsecover[col] = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=weighted)[0]

    if normalised:
        # return (rmsecover - rmsecover.min())/(rmsecover.max()-rmsecover.min())
        return rmsecover / rmsecover.max()
    else:
        return rmsecover


def std_coverage(df, use, normalised=False):
    stdcover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue

        # following Fortin 2014
        spread = np.sqrt(df.loc[:, use + [col]].var(axis=1).mean())

        stdcover[col] = spread

    if normalised:
        # return (stdcover - stdcover.min())/(stdcover.max()-stdcover.min())
        return stdcover / stdcover.max()
    else:
        return stdcover


def pareto_coverage(runs, obs, use, cov=False):

    mac = mae_coverage(runs, use, obs, normalised=True)
    if len(use) == 0:
        stc = 0
    else:
        stc = std_coverage(runs, use, normalised=True)

    if cov:
        cover = coverage_loop(runs, use, obs, normalised=True)
    else:
        cover = 0

    # euclidian dist
    edisx = np.sqrt(1 * stc**2 +
                    1 * cover**2 +
                    1 * mac**2
                    ).sort_values()

    return edisx


def pareto_coverage_2(runs, obs, use, cov=False, mae=True, spread=True):

    if mae:
        mac = mae_coverage_2(runs, use, obs, normalised=True)
    else:
        mac = 0

    if spread and (len(use) > 0):
        stc = std_coverage(runs, use, normalised=True)

        # ma = mae_coverage_2(runs, use, obs, normalised=False)
        # st = std_coverage(runs, use, normalised=False)
        # stc = (1 - (ma/st)).abs()
        # stc = (stc - stc.min())/(stc.max()-stc.min())
    else:
        stc = 0

    if cov:
        cover = coverage_loop(runs, use, obs, normalised=True)
    else:
        cover = 0

    # euclidian dist
    edisx = np.sqrt(1 * stc**2 +
                    1 * cover**2 +
                    1 * mac**2
                    ).sort_values()

    return edisx


def pareto_coverage_4(runs, obs, use, cov=False, rmse=False, mae=True, spread=True, maespread=False, rmsespread=False, weighted=True):
    # MEDIAN

    if mae:
        mac = mae_coverage_3(runs, use, obs, normalised=True, weighted=False)
    else:
        mac = 0

    if rmse:
        rms = rmse_coverage_3(runs, use, obs, normalised=True, weighted=weighted)
    else:
        rms = 0

    if spread and (len(use) > 0):
        stc = std_coverage(runs, use, normalised=True)
    else:
        stc = 0

    if cov and (len(use) > 0):
        cover = coverage_loop2(runs, use, obs, normalised=True)
    else:
        cover = 0

    if maespread and (len(use) > 0):
        # SKILL IS MEAN
        ma = mae_coverage_2(runs, use, obs, normalised=False, weighted=weighted)
        st = std_coverage(runs, use, normalised=False)
        mast = (1 - (ma/st)).abs()
        mast = (mast - mast.min())/(mast.max()-mast.min())
    else:
        mast = 0

    if rmsespread and (len(use) > 0):
        # SKILL IS MEAN
        rm = rmse_coverage_2(runs, use, obs, normalised=False, weighted=weighted)
        st2 = std_coverage(runs, use, normalised=False)
        rmsp = (1 - (rm/st2)).abs()
        rmsp = (rmsp - rmsp.min())/(rmsp.max()-rmsp.min())
    else:
        rmsp = 0

    try:
        edisx = np.sqrt(1 * stc**2 +
                        1 * cover**2 +
                        1 * mac**2 +
                        1 * rmsp**2 +
                        1 * rms**2 +
                        1 * mast**2
                        ).sort_values()
    except AttributeError:
        # if all zero, return MAE
        return mae_coverage_3(runs, use, obs, normalised=True, weighted=weighted).sort_values()

    return edisx


def pareto_coverage_3(runs, obs, use, cov=False, rmse=False, mae=True, spread=True, skill=False, only_improve_skill=False, weighted=True, only_improve_coverage=False, maxerr=False,
                      return_minimum=False, return_rank0=False):

    if mae:
        mac = mae_coverage_2(runs, use, obs, normalised=True, weighted=False)
    else:
        mac = 0

    if maxerr:
        maxe = maxerror_coverage(runs, use, obs, normalised=True)
    else:
        maxe = 0

    if rmse:
        rms = rmse_coverage_2(runs, use, obs, normalised=True, weighted=weighted)
    else:
        rms = 0

    if spread and (len(use) > 0):
        stc = std_coverage(runs, use, normalised=True)
    else:
        stc = 0

    if cov and (len(use) > 0):

        if only_improve_coverage:
            cover = coverage_loop(runs, use, obs, normalised=False)
            oldcov = calc_coverage_2(runs, use, obs)

            # threshold: everything smaller we neglect
            thresh = min(oldcov, 0.7)
            cover[cover < thresh] = np.nan
            # everything else will be keept but not used in the pareto
            cover[cover.notna()] = 0

        else:
            cover = coverage_loop(runs, use, obs, normalised=True)

    else:
        cover = 0

    if skill and (len(use) > 0):
        rm = rmse_coverage_2(runs, use, obs, normalised=False, weighted=weighted)
        st2 = std_coverage(runs, use, normalised=False)
        rmsp = rm/st2

        if only_improve_skill:
            ens = runs.loc[:, use].mean(axis=1)
            oldrms = rmse_weighted(pd.concat([obs, ens], axis=1),
                                   weighted=weighted)[0]
            oldspr = np.sqrt(runs.loc[:, use].var(axis=1).mean())
            oldskill = oldrms/oldspr

            oldskill = 0 if np.isnan(oldskill) else oldskill

            # threshold: everything smaller we neglect
            thresh1 = min(oldskill, 0.7)
            rmsp[rmsp < thresh1] = np.nan
            thresh2 = max(oldskill, 2.0)
            rmsp[rmsp > thresh2] = np.nan
            # everything else will be keept but not used in the pareto
            rmsp[rmsp.notna()] = 0

        else:
            rmsp = (1 - (rmsp)).abs()
            # rmsp = (rmsp - rmsp.min())/(rmsp.max()-rmsp.min())
            rmsp = rmsp / rmsp.max()

    else:
        rmsp = 0

    try:
        edisx = np.sqrt(1 * stc**2 +
                        1 * cover**2 +
                        1 * mac**2 +
                        1 * maxe**2 +
                        2 * rmsp**2 +
                        1 * rms**2
                        ).sort_values()
    except AttributeError:
        # if all zero, return MAE
        return mae_coverage_2(runs, use, obs, normalised=True, weighted=weighted).sort_values()

    if return_minimum:
        return edisx.idxmin()
    elif return_rank0:

        rank = edisx.copy()
        for ix, _ in rank.iteritems():
            rank[ix] = ((mac < mac[ix]) &
                        (rmsp < rmsp[ix])).sum()

        r0 = rank[rank == 0].index
        return r0
    else:
        return edisx


def fit_one_std_2(runs, obs, glid):

    # list which indices to use
    use = []

    coverage_old = 0

    ite = 0

    skip = None

    while len(use) <= 10:

        # pareto
        pix = pareto_coverage(runs.loc[:, runs.columns != skip], obs, use).idxmin()

        # calculate new coverage
        coverage = calc_coverage(runs, use + [pix], obs)

        if coverage > coverage_old:
            # keep if improvement
            use.append(pix)
            coverage_old = coverage
            print('%s: %3d, %.3f' % (glid, len(use), coverage))
        else:
            # do not use in the next run
            print('decreasing, skip: %s' % pix)
            del runs[pix]
            #skip = pix

        if (len(use) >= 5) and (coverage > 0.95):
            break
        # if (len(use) > 5) and (coverage_old > coverage):
        #    break

        ite += 1
        if ite == 100:
            print('iter 100')
            break

    return use, coverage_old


def fit_one_std_2b(runs, obs, glid, minuse=5, maxuse=10, cov=False, detrend=False):

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []
    bestuse = []
    bestcoverage = 0

    for _ in range(maxuse):

        # pareto
        pix = pareto_coverage(runs, obs, use, cov=cov).idxmin()
        use.append(pix)

        # calculate new coverage
        coverage = calc_coverage(runs, use, obs, detrend=detrend)
        print('%s: %3d, %.3f' % (glid, len(use), coverage))

        if (len(use) >= minuse) and (coverage > bestcoverage):
            bestcoverage = coverage
            bestuse = use.copy()

            # search no further
            if bestcoverage > 0.8:
                break

    return bestuse, bestcoverage


def fit_one_std_2c(runs, obs, glid, minuse=5, maxuse=10, detrend=False):
    # ganz gut soweit
    # 2d: probieren nur dazu addieren was coverage erhöht

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []
    bestuse = []
    bestcoverage = 0

    for _ in range(maxuse):

        # pareto
        pix = pareto_coverage_2(runs, obs, use, spread=False).idxmin()
        use.append(pix)

        # calculate new coverage
        coverage = calc_coverage_2(runs, use, obs, detrend=detrend)
        print('%s: %3d, %.3f' % (glid, len(use), coverage))

        if (len(use) >= minuse) and (coverage > bestcoverage):
            bestcoverage = coverage
            bestuse = use.copy()

            # search no further
            if bestcoverage > 0.8:
                break

    return bestuse, bestcoverage


def fit_one_std_2d(_runs, obs, glid, minuse=5, maxuse=10, detrend=False):
    # naja, fast 2c besser
    # 2e: MAE minimieren

    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []
    coverage = -1
    bestcoverage = -1

    for _ in range(maxuse):

        # pareto
        while coverage <= bestcoverage:
            pix = pareto_coverage_2(runs, obs, use, cov=True).idxmin()
            # calculate new coverage
            coverage = calc_coverage_2(runs, use + [pix], obs, detrend=detrend)
            print('%s: %3d, %.3f, %s' % (glid, len(use)+1, coverage, pix))

            # once used, remove it. If good, added again later
            del runs[pix]

        # ok, out of loop, we improved performance with pix
        use.append(pix)
        runs[pix] = _runs[pix]
        bestcoverage = coverage

        if (len(use) >= minuse) and (bestcoverage > 0.8):
            break

    return use, bestcoverage


def fit_one_std_2h(_runs, obs, glid, minuse=5, maxuse=10):
    # goal of 2h
    # use MEDIAN...

    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []

    usedf = pd.DataFrame([], columns=['mae', 'rmse', 'spread', 'coverage', 'mae-spread', 'rms-spread', 'rmswg-spread', 'use'])

    for i in range(1, maxuse+1):

        pix = pareto_coverage_4(runs, obs, use, cov=True, mae=False, rmse=True, spread=False, maespread=False, rmsespread=False, weighted=True).idxmin()
        use.append(pix)

        ens = runs.loc[:, use].median(axis=1)

        usedf.loc[i, 'mae'] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'rmse'] = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'spread'] = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        maer = ens.sub(obs, axis=0).dropna().abs().mean()
        rms = np.sqrt((ens.sub(obs, axis=0).dropna()**2).mean())
        rmsewg = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        usedf.loc[i, 'rms-spread'] = rms/sprd
        usedf.loc[i, 'rmswg-spread'] = rmsewg/sprd
        usedf.loc[i, 'mae-spread'] = maer/sprd

        usedf.loc[i, 'coverage'] = calc_coverage_2(runs, use, obs)
        usedf.loc[i, 'use'] = use.copy()

        print('%s: %2d, %.2f, %.2f, %.2f | %.2f, %.2f' % (glid, len(use), usedf.loc[i, 'coverage'], usedf.loc[i, 'mae'], usedf.loc[i, 'rmse'], usedf.loc[i, 'rms-spread'], usedf.loc[i, 'rmswg-spread']))

        #if (i >= minuse) and (usedf.loc[i, 'coverage'] >= 0.9):
        #    break
    sk = (1-usedf.loc[minuse:, 'rmswg-spread'].astype(float)).abs()
    sk = (sk - sk.min()) / (sk.max() - sk.min())
    cv = (1-usedf.loc[minuse:, 'coverage'].astype(float)).abs()
    cv = (cv - cv.min()) / (cv.max() - cv.min())
    #ma = usedf.loc[minuse:, 'mae'].astype(float)
    #ma = (ma - ma.min()) / (ma.max() - ma.min())
    # edis = np.sqrt(0*cv**2 + sk**2)
    # idx = edis.idxmin()
    idx = usedf.loc[minuse:, 'mae'].astype(float).idxmin()
    idx = sk.idxmin()

    return usedf.loc[idx, 'use'], usedf.loc[idx, 'coverage']

    # if i == maxuse:
    #     i = usedf.loc[minuse:, 'coverage'].astype(float).idxmax()

    # return usedf.loc[i, 'use'], usedf.loc[i, 'coverage']


def fit_one_std_2g(_runs, obs, glid, minuse=5, maxuse=10):
    # goal of 2g:
    # select runs based on MAE (only? or + spread or something)
    # do until coverage > x is reached

    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []

    usedf = pd.DataFrame([], columns=['mae', 'maxerror', 'rmse', 'spread', 'coverage', 'mae-spread', 'rms-spread', 'rmswg-spread', 'use'])

    for i in range(1, maxuse+1):

        pix = pareto_coverage_3(runs, obs, use,
                                cov=False, mae=True, rmse=False, spread=False,
                                skill=True, only_improve_skill=True,
                                weighted=True, only_improve_coverage=False,
                                maxerr=True).idxmin()
        use.append(pix)

        try:
            ens = runs.loc[:, use].mean(axis=1)
        except:
            print(glid)

        usedf.loc[i, 'mae'] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'maxerror'] = maxerror(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'rmse'] = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'spread'] = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        maer = ens.sub(obs, axis=0).dropna().abs().mean()
        rms = np.sqrt((ens.sub(obs, axis=0).dropna()**2).mean())
        rmsewg = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        usedf.loc[i, 'rms-spread'] = rms/sprd
        usedf.loc[i, 'rmswg-spread'] = rmsewg/sprd
        usedf.loc[i, 'mae-spread'] = maer/sprd

        usedf.loc[i, 'coverage'] = calc_coverage_2(runs, use, obs)
        usedf.loc[i, 'use'] = use.copy()

        print('%s: %2d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f' % (glid, len(use), usedf.loc[i, 'coverage'], usedf.loc[i, 'spread'], usedf.loc[i, 'mae'], usedf.loc[i, 'rmse'], usedf.loc[i, 'maxerror'], usedf.loc[i, 'rmswg-spread']))

        #if (i >= minuse) and (usedf.loc[i, 'coverage'] >= 0.9):
        #    break
    sk = (1-usedf.loc[minuse:, 'rmswg-spread'].astype(float)).abs()
    # sk = (sk - sk.min()) / (sk.max() - sk.min())
    sk = sk / sk.max()
    cv = (1-usedf.loc[minuse:, 'coverage'].astype(float)).abs()
    # cv = (cv - cv.min()) / (cv.max() - cv.min())
    cv = cv / cv.max()
    ma = usedf.loc[minuse:, 'mae'].astype(float)
    ma = ma / ma.max()
    edis = np.sqrt(cv**2 + 0*sk**2 + ma**2)
    idx = edis.idxmin()
    # idx = usedf.loc[minuse:, 'mae'].astype(float).idxmin()

    for i in usedf.loc[idx, 'use']:
        print(i)
    return usedf.loc[idx, 'use'], usedf.loc[idx, 'coverage']

    # if i == maxuse:
    #     i = usedf.loc[minuse:, 'coverage'].astype(float).idxmax()

    # return usedf.loc[i, 'use'], usedf.loc[i, 'coverage']


def fit_one_std_2f(_runs, obs, glid, minuse=5, maxuse=10):
    # goal of 2f:
    # paretofront of mae, spread, cov, mae/spread (try importance)
    # do 10 times
    # select best ensemble

    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []

    usedf = pd.DataFrame([], columns=['mae', 'spread', 'coverage', 'mae-spread', 'use'])

    for i in range(1, maxuse+1):

        pix = pareto_coverage_3(runs, obs, use, cov=True, mae=True, spread=False, maespread=False).idxmin()
        use.append(pix)

        ens = runs.loc[:, use].mean(axis=1)

        usedf.loc[i, 'mae'] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'spread'] = np.sqrt(runs.loc[:, use].var(axis=1).mean())
        usedf.loc[i, 'mae-spread'] = usedf.loc[i, 'mae']/usedf.loc[i, 'spread']
        usedf.loc[i, 'coverage'] = calc_coverage_2(runs, use, obs)
        usedf.loc[i, 'use'] = use.copy()

        print('%s: %2d, %.2f, %.2f, %.2f. %.2f' % (glid, len(use), usedf.loc[i, 'coverage'], usedf.loc[i, 'mae'], usedf.loc[i, 'spread'], usedf.loc[i, 'mae-spread']))

    usedf = usedf.loc[minuse:maxuse]

    ma = (usedf['mae'] - usedf['mae'].min()) / (usedf['mae'].max() - usedf['mae'].min())
    sp = (usedf['spread'] - usedf['spread'].min()) / (usedf['spread'].max() - usedf['spread'].min())

    covn = 1 - usedf['coverage']
    if covn.min() == covn.max():
        cov = covn
    else:
        cov = (covn - covn.min()) / (covn.max() - covn.min())

    maspn = (1 - usedf['mae-spread']).abs()
    masp = (maspn - maspn.min()) / (maspn.max() - maspn.min())

    # euclidian dist
    edisx = np.sqrt((1 * ma**2 +
                     0 * cov**2 +
                     0 * masp**2 +
                     0 * sp**2
                     ).astype(float))

    try:
        return usedf.loc[edisx.idxmin(), 'use'], usedf.loc[edisx.idxmin(), 'coverage']
    except:
        print('asd')


def fit_one_std_2e(_runs, obs, glid, minuse=5, maxuse=10, detrend=False):
    # naja, fast 2c besser
    # 2e: MAE minimieren

    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use, we start with min MAE
    mae1 = mae_weighted(pd.concat([_runs, obs], axis=1), normalised=False)
    use = [mae1.idxmin()]
    cmae = mae1.min()
    minmae = cmae

    for _ in range(maxuse):

        i = 0
        # pareto
        while cmae >= minmae:
            pix = pareto_coverage_2(runs, obs, use, cov=True, mae=False, spread=True).idxmin()

            ens = runs.loc[:, use + [pix]].mean(axis=1)

            # calculate new MAE
            cmae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]

            print('%s: %3d, %.3f, %s' % (glid, len(use)+1, cmae, pix))

            # once used, remove it. If good, added again later
            del runs[pix]

            i+=1
            if i == 10:
                pix = None
                break

        if i == 10:
            break

        # ok, out of loop, we improved performance with pix
        if pix is not None:
            use.append(pix)
            runs[pix] = _runs[pix]
            minmae = cmae

        if (len(use) >= minuse) and (minmae < 100):
            break

    return use, minmae


def fit_one_std_1(runs, obs, glid):

    # list which indices to use
    use = []

    coverage_old = 0

    it = 0

    while len(use) <= 10:

        coverage = coverage_loop(runs, use, obs)
        cmax = coverage.max()

        if cmax > coverage_old:
            # keep if improvement
            use.append(coverage.idxmax())
            coverage_old = cmax

            print('%s: %3d, %.3f' % (glid, len(use), cmax))
            if (len(use) >= 5) and (cmax > 0.95):
                break

        elif len(use) >= 5:
            break

        it += 1
        if it == 100:
            print('iter 100')
            break

    return use, coverage_old


def fit_one_std(df, metric, glid):

    obs = df.loc[:, 'obs'].dropna()

    """
    tst = 0
    mem = 1

    while tst < 0.7:
        ens = df.loc[:, metric[:mem]].mean(axis=1)
        dt_std = detrend_series(ens).std()
        ens_p1 = ens + dt_std
        ens_m1 = ens - dt_std
        tst = ((obs.dropna() > ens_m1.loc[idx]) &
               (obs.dropna() < ens_p1.loc[idx])).sum()/len(idx)
        mem += 1

        if mem == 200:
            print('failed')
            break
    return metric[:mem]
    """

    use = []
    tst = 0

    for idx in metric:

        # correlation
        if len(use) > 0:
            r2 = df.loc[:, use].mean(axis=1).corr(df.loc[:, idx])

        use.append(idx)

        ens = df.loc[:, use].mean(axis=1)
        dt_std = detrend_series(ens).std()
        ens_p1 = ens + dt_std
        ens_m1 = ens - dt_std

        tst2 = ((obs.dropna() > ens_m1.loc[obs.dropna().index]) &
                (obs.dropna() < ens_p1.loc[obs.dropna().index])).sum()/len(obs.dropna())

        if (len(use) > 1) and (tst2 > tst):
            print("gain: %.2f,  r2: %.2f" % (tst2-tst, r2))

        if tst2 > 0.7:
            if len(use) >= 5:
                # good to go
                break
            else:
                # we want an ensemble
                continue
        elif (tst2 - tst) > 0.05:
            # elif tst2 > tst:
            # some improvement
            tst = tst2
        else:
            # that didnt help
            use.remove(idx)


    print('%s: %3d, %.3f' % (glid, len(use), tst2))

    return use


def powerset(iterable, include_zero=False):
    import itertools

    if include_zero is True:
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        z = 0
    else:
        "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        z = 1

    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r)
                                         for r in range(z, len(s)+1))


def best_powerset(df, idx):

    # all combinations of idx
    pdict = dict(zip(range(2**len(idx)), powerset(idx)))

    # observations
    obs = df.loc[:, 'obs'].dropna()

    stats = pd.DataFrame([], columns=['number', 'stats'])

    for key, item in pdict.items():

        # calc ensemble mean
        ens = df.loc[:, item].mean(axis=1)
        dt_std = detrend_series(ens).std()
        ens_p1 = ens + dt_std
        ens_m1 = ens - dt_std
        tst = ((obs > ens_m1.loc[obs.index]) &
               (obs < ens_p1.loc[obs.index])).sum()/len(obs)

        stats.loc[key, 'stats'] = tst
        stats.loc[key, 'number'] = len(item)

    return pdict[stats.sort_values('stats').index[-1]]


def pareto_nach_rye(df, glid):
    # goal:
    # - calc pareto fuer alle runs mir MAE+cummulative error+irgendwas mit SD, correlation etc.
    # - scatter plot (1x3) mit allen combis.
    # - bold rank 0 values
    # - calc coverage and model skill
    # - return rank 0 values

    maes = mae_weighted(df, normalised=False, detrend=False)
    stdq = std_quotient(df, normalised=False)
    stdq = np.abs(1 - stdq)
    maxr = maxerror(df, normalised=False)

    car = cumerror(df, normalised=False).astype(float)

    # euclidian dist
    edisx = np.sqrt(1 * maes ** 2 +
                    1 * maxr**2 +
                    1 * car ** 2
                    ).sort_values()

    # pareto front

    rank = edisx.copy()

    for ix, _ in edisx.iteritems():
        rank[ix] = ((maes < maes[ix]) &
                    (maxr < maxr[ix]) &
                    (car < car[ix])).sum()

    r0 = rank[rank == 0].index

    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=[20, 8])

    ax0.plot(maes, maxr, '.k', color='0.3')
    ax0.plot(maes[r0], maxr[r0], 'or')
    ax0.set_xlabel('MAE')
    ax0.set_xlim([0, maes[r0].max()])
    ax0.set_ylabel('MAX')
    ax0.set_ylim([0, maxr[r0].max()])

    ax1.plot(maes, car, '.k', color='0.3')
    ax1.plot(maes[r0], car[r0], 'or')
    ax1.set_xlabel('MAE')
    ax1.set_xlim([0, maes[r0].max()])
    ax1.set_ylabel('CAR')
    ax1.set_ylim([0, car[r0].max()])

    ax2.plot(maxr, car, '.k', color='0.3')
    ax2.plot(maxr[r0], car[r0], 'or')
    ax2.set_xlabel('MAX')
    ax2.set_xlim([0, maxr[r0].max()])
    ax2.set_ylabel('CAR')
    ax2.set_ylim([0, car[r0].max()])

    rgi_id = glid.split('_')[0]
    fig.suptitle('{} - {}'.format(GLCDICT[rgi_id][2], glid))
    #plt.show()
    fig.savefig('/home/matthias/length_change_1850/neu/pareto{}.png'.format(glid))

    return r0


def pareto_nach_rye2(df, glid):
    # goal:
    # - 1. use min MAE
    # - calc paretoe: MAE, SKILL, (SPREAD?)
    # - use whole paretor front
    # repeat until coverage
    runs = df.loc[:, df.columns != 'obs']
    obs = df.loc[:, 'obs']
    # list which indices to use
    use = [mae_weighted(df, normalised=False).idxmin()]
    print('{}: {:3d} |{:5.2f} |{:7.2f} |{:7.2f} |{:5.2f}'.
          format(glid, len(use), 0, 0,
                 mae_weighted(df, normalised=False).min(), 0))

    cov = 0

    while cov < 1:

        pix = pareto_coverage_3(runs, obs, use,
                                cov=True, mae=True, rmse=False, spread=True,
                                skill=False, only_improve_skill=False,
                                weighted=True, only_improve_coverage=False,
                                maxerr=False,
                                return_minimum=True,
                                return_rank0=False)

        use.append(pix)
        #use += pix.to_list()

        try:
            ens = runs.loc[:, use].mean(axis=1)
        except:
            print(glid)

        maer = ens.sub(obs, axis=0).dropna().abs().mean()
        rms = np.sqrt((ens.sub(obs, axis=0).dropna()**2).mean())
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        rmssprd = rms/sprd

        cov = calc_coverage_2(runs, use, obs)

        print('{}: {:3d} |{:5.2f} |{:7.2f} |{:7.2f} |{:5.2f}'.
              format(glid, len(use), cov, sprd, maer, rmssprd))

        if len(use) == 10:
            break

    return use


def montec(df, glid):
    # goal
    # sample 5-10 random members
    # calc mae, coverage, skill
    # use best set
    import warnings
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    obs = df['obs']

    # we sample from 50 smallest errors (pos and neg bias)
    me = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0). \
        dropna().mean()
    mepl = me[me > 0].sort_values().index[:100]
    memi = me[me < 0].sort_values(ascending=False).index[:100]
    best = mepl.append(memi)

    print('{}: {:3} | {:5} | {:7} | {:7} | {:7} | {:5} | {:6}'.
          format(glid, 'use', 'cov', 'spread', 'MAE', 'MAX', 'skill', 'pareto')),

    usedf = pd.DataFrame([], columns=['mae', 'spread', 'cov', 'skill', 'use',
                                      'pareto', 'max'])

    np.random.seed(42)

    import time
    t1 = time.time()
    # how many times do we try
    for i in range(100000):

        # how many samples
        n = np.random.randint(10, 30)

        # which runs
        runint = np.random.randint(0, len(best), n)
        # sort and unique
        runint.sort()
        runint = np.unique(runint)
        use = best[runint]

        cov = calc_coverage_2(df, use, obs)

        if cov < 0.7:
            continue

        ens = df.loc[:, use].mean(axis=1)

        rms = np.sqrt((ens.sub(obs, axis=0).dropna()**2).mean())
        sprd = np.sqrt(df.loc[:, use].var(axis=1).mean())
        skill = rms/sprd

        if (skill < 0.8) or (skill > 1.2):
            continue

        maew = mae_weighted(pd.concat([obs, ens], axis=1),
                            normalised=False, weighted=True)[0]
        maxe = maxerror(pd.concat([obs, ens], axis=1), normalised=False)[0]

        #print('{}: {:3d} |{:5.2f} |{:7.2f} |{:7.2f} |{:5.2f}'.
        #      format(glid, len(use), cov, sprd, maer, skill))

        usedf.loc[i, 'mae'] = maew
        usedf.loc[i, 'max'] = maxe
        usedf.loc[i, 'spread'] = sprd
        usedf.loc[i, 'cov'] = cov
        usedf.loc[i, 'skill'] = skill
        usedf.loc[i, 'use'] = use

    if len(usedf) == 0:
        return mepl[0:2]

    # reduce size
    usedf_red = usedf.loc[(usedf['cov'] > 0.7) &
                          (usedf['skill'] > 0.8) &
                          (usedf['skill'] < 1.2), :]

    if len(usedf_red) > 1:
        usedf = usedf_red

    maew = usedf['mae'].astype(float)
    # maewn = (maew - maew.min()) / (maew.max() - maew.min())
    maewn = maew / maew.max()
    maxe = usedf['max'].astype(float)
    # maxen = (maxe - maxe.min()) / (maxe.max() - maxe.min())
    maxen = maxe / maxe.max()

    skill = (1-usedf['skill'].astype(float)).abs()
    # skilln = (skill - skill.min()) / (skill.max() - skill.min())
    skilln = skill / skill.max()
    cov = (1 - usedf['cov'].astype(float)).abs()
    # covn = (cov - cov.min()) / (cov.max() - cov.min())
    covn = cov / cov.max()

    # euclidian dist
    edisx = np.sqrt(1 * maewn**2 +
                    0 * skilln**2 +
                    1 * maxen**2 +
                    0 * covn**2
                    )
    usedf['pareto'] = edisx

    ix = usedf.sort_values('pareto').index[0]
    print('{}: {:3d} | {:5.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:5.2f} | {:6.2f}'.
          format(glid, len(usedf.loc[ix, 'use']),
                 usedf.loc[ix, 'cov'],
                 usedf.loc[ix, 'spread'],
                 usedf.loc[ix, 'mae'],
                 usedf.loc[ix, 'max'],
                 usedf.loc[ix, 'skill'],
                 usedf.loc[ix, 'pareto']))

    ix2 = usedf.sort_values('mae').index[0]
    print('{}: {:3d} | {:5.2f} | {:7.2f} | {:7.2f} | {:7.2f} | {:5.2f} | {:6.2f}'.
          format(glid, len(usedf.loc[ix2, 'use']),
                 usedf.loc[ix2, 'cov'],
                 usedf.loc[ix2, 'spread'],
                 usedf.loc[ix2, 'mae'],
                 usedf.loc[ix2, 'max'],
                 usedf.loc[ix2, 'skill'],
                 usedf.loc[ix2, 'pareto']))
    print('time (sec): {}'.format(time.time() - t1))
    return usedf.loc[ix, 'use']


def fit_cov_and_skill(_runs, obs, glid, minuse=5, maxuse=30):
    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    use = []

    usedf = pd.DataFrame([], columns=['mae', 'maxerror', 'rmse', 'spread', 'coverage', 'mae-spread', 'rms-spread', 'rmswg-spread', 'use'])

    for i in range(1, maxuse+1):

        pix = pareto_only_improve(runs, obs, use,
                                  cov=True, mae=True, rmse=False, spread=False,
                                  skill=True, weighted=True,
                                  maxerr=False).idxmin()
        #pix = pareto_coverage_3(runs, obs, use,
        #                        cov=True, mae=True, rmse=False,
        #                        spread=False,
        #                        skill=True, weighted=True, only_improve_skill=False,
        #                        only_improve_coverage=False,
        #                        return_minimum=False,
        #                        maxerr=True).idxmin()
        use.append(pix)

        try:
            ens = runs.loc[:, use].mean(axis=1)
        except:
            print(glid)

        usedf.loc[i, 'mae'] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'maxerror'] = maxerror(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'rmse'] = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False)[0]
        usedf.loc[i, 'spread'] = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        maer = ens.sub(obs, axis=0).dropna().abs().mean()
        rms = np.sqrt((ens.sub(obs, axis=0).dropna()**2).mean())
        rmsewg = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        usedf.loc[i, 'rms-spread'] = rms/sprd
        usedf.loc[i, 'rmswg-spread'] = rmsewg/sprd
        usedf.loc[i, 'mae-spread'] = maer/sprd

        usedf.loc[i, 'coverage'] = calc_coverage_2(runs, use, obs)
        usedf.loc[i, 'use'] = use.copy()

        print('%s: %2d, %.2f, %.2f, %.2f' % (glid, len(use), usedf.loc[i, 'coverage'], usedf.loc[i, 'mae'], usedf.loc[i, 'rmswg-spread']))

        skll = usedf.loc[i, 'rmswg-spread'].astype(float)
        skll = 1 / skll if skll > 1 else skll

        #if (i >= minuse) and (usedf.loc[i, 'coverage'] >= 0.7) and (skll >= 0.8):
        #    print('%s: %2d, %.2f, %.2f, %.2f --- break' % (
        #        glid, len(usedf.loc[i, 'use']), usedf.loc[i, 'coverage'],
        #        usedf.loc[i, 'mae'],
        #        usedf.loc[i, 'rmswg-spread']))
        #    return usedf.loc[i, 'use']

    #return usedf.loc[i, 'use']

    useindex = usedf.loc[minuse:, :].index

    mincov = min(0.7, usedf.loc[useindex, 'coverage'].max())

    useindex = usedf.loc[useindex, :].loc[usedf['coverage']>=mincov].index

    rmsp = usedf.loc[useindex, 'rmswg-spread'].astype(float)
    rmsp[rmsp > 1] = 1 / rmsp[rmsp > 1]

    id1 = rmsp.idxmax()


    print('%s: %2d, %.2f, %.2f, %.2f' % (
    glid, len(usedf.loc[id1, 'use']), usedf.loc[id1, 'coverage'],
    usedf.loc[id1, 'mae'],
    usedf.loc[id1, 'rmswg-spread']))


    #sk = usedf.loc[useindex, 'rmswg-spread'].astype(float)
    sk = usedf.loc[:, 'rmswg-spread'].astype(float)
    sk[sk > 1] = 1/sk[sk > 1]
    sk = (1-sk).abs()
    #sk = sk / sk.max()
    # cv = (1-usedf.loc[useindex, 'coverage'].astype(float)).abs()
    cv = (1 - usedf.loc[:, 'coverage'].astype(float)).abs()
    #cv = cv / cv.max()

    id2 = np.sqrt(cv ** 2 + sk ** 2).idxmin()
    print('%s: %2d, %.2f, %.2f, %.2f' % (
        glid, len(usedf.loc[id2, 'use']), usedf.loc[id2, 'coverage'],
        usedf.loc[id2, 'mae'],
        usedf.loc[id2, 'rmswg-spread']))


    id3 = sk.idxmin()

    print('%s: %2d, %.2f, %.2f, %.2f' % (
        glid, len(usedf.loc[id3, 'use']), usedf.loc[id3, 'coverage'],
        usedf.loc[id3, 'mae'],
        usedf.loc[id3, 'rmswg-spread']))

    return usedf.loc[id1, 'use']


    me = usedf.loc[useindex, 'mae'].astype(float)
    me = me / me.max()

    ma = usedf.loc[minuse:, 'maxerror'].astype(float)
    ma = ma / ma.max()

    #edis = np.sqrt(cv**2 + sk**2)
    edis = np.sqrt(cv ** 2 + sk ** 2 + ma ** 2 + me ** 2)
    idx = edis.idxmin()

    #for i in usedf.loc[idx, 'use']:
    #    print(i)
    return usedf.loc[idx, 'use']


def pareto_only_improve(runs, obs, use, cov=False, rmse=False, mae=True, spread=True, skill=False, weighted=True, maxerr=False):

    if mae:
        mac = mae_coverage_2(runs, use, obs, normalised=True, weighted=False)
    else:
        mac = 0

    if maxerr:
        maxe = maxerror_coverage(runs, use, obs, normalised=True)
    else:
        maxe = 0

    if rmse:
        rms = rmse_coverage_2(runs, use, obs, normalised=True, weighted=weighted)
    else:
        rms = 0

    if spread and (len(use) > 0):
        stc = std_coverage(runs, use, normalised=True)
    else:
        stc = 0

    if cov and (len(use) > 0):
        cover = coverage_loop(runs, use, obs, normalised=False)
        oldcov = calc_coverage_2(runs, use, obs)
        # threshold: everything smaller we neglect
        thresh = min(oldcov+0.01, 0.7)
        cover[cover < thresh] = np.nan
        #cover[cover < oldcov] = np.nan
        # normalise
        cover = cover/cover.max()
    else:
        cover = 0

    if skill and (len(use) > 0):
        rm = rmse_coverage_2(runs, use, obs, normalised=False, weighted=weighted)
        st2 = std_coverage(runs, use, normalised=False)
        rmsp = rm/st2
        # normalise all values to smaller 1
        rmsp[rmsp > 1] = 1 / rmsp[rmsp > 1]

        ens = runs.loc[:, use].mean(axis=1)
        # TODO RMSE_WEIGHTED?
        oldrms = rmse_weighted(pd.concat([obs, ens], axis=1),
                               weighted=weighted)[0]
        oldspr = np.sqrt(runs.loc[:, use].var(axis=1).mean())
        oldskill = oldrms / oldspr
        oldskill = 0 if np.isnan(oldskill) else oldskill
        # normalise values to smaller 1
        oldskill = 1/oldskill if oldskill > 1 else oldskill

        # threshold: everything smaller we neglect
        thresh = min(oldskill, 0.7)
        rmsp[rmsp < thresh] = np.nan
        #rmsp[rmsp < oldskill] = np.nan

        # normalise to zero
        rmsp = (1 - rmsp).abs()
        rmsp = rmsp / rmsp.max()

    else:
        rmsp = 0

    try:
        edisx = np.sqrt(0 * stc**2 +
                        0 * cover**2 +
                        1 * mac**2 +
                        0 * maxe**2 +
                        0 * rmsp**2 +
                        0 * rms**2
                        ).sort_values()
    except AttributeError:
        # if all zero, return MAE
        return mae_coverage_2(runs, use, obs, normalised=True, weighted=weighted).sort_values()

    return edisx


def pareto_only_improve2(runs, obs, use, edisp, maeref, skllref, covref):

    mac = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
    mac2 = mac/mac.max()

    mae3 = mac/maeref

    maxe = maxerror_coverage(runs, use, obs, normalised=False)
    maxe2 = maxe/maxe.max()

    mape = mape_coverage(runs, use, obs)

    # coverage
    if len(use) > 0:
        cover = coverage_loop(runs, use, obs, normalised=False)
        # normalise
        #cover[cover > 0.68] = 1

        if 0:
            oldcov = calc_coverage_2(runs, use, obs)
            # threshold: everything smaller we neglect
            #thresh = min(oldcov + 0.01, 0.7)
            cover[cover < oldcov] = np.nan
            # cover[cover < oldcov] = np.nan

        # normalise
        cover = (1 - cover).abs()
        cover2 = cover / cover.max()
        cover3 = cover / covref

    else:
        cover = 0
        cover2 = 0

    if len(use) > 0:
        rm = rmse_coverage_2(runs, use, obs, normalised=False, weighted=True)
        st2 = std_coverage(runs, use, normalised=False)
        rmsp = rm/st2
        #rmsp = mac / st2
        # normalise all values to smaller 1
        rmsp[rmsp > 1] = 1 / rmsp[rmsp > 1]

        if 0:
            ens = runs.loc[:, use].mean(axis=1)
            # TODO RMSE_WEIGHTED?
            #oldrms = rmse_weighted(pd.concat([obs, ens], axis=1),
            #                       weighted=True)[0]
            oldrms = mae_weighted(pd.concat([obs, ens], axis=1),
                                  weighted=True)[0]
            oldspr = np.sqrt(runs.loc[:, use].var(axis=1).mean())
            oldskill = oldrms / oldspr
            oldskill = 0 if np.isnan(oldskill) else oldskill
            # normalise values to smaller 1
            oldskill = 1 / oldskill if oldskill > 1 else oldskill

            # threshold: everything smaller we neglect
            #thresh = min(oldskill, 0.7)
            rmsp[rmsp < oldskill] = np.nan

        # normalise to zero
        rmsp = (1 - rmsp).abs()
        rmsp2 = rmsp / rmsp.max()
        rmsp3 = rmsp / skllref


    else:
        rmsp = 0
        rmsp2 = 0

    edisx = np.sqrt(1 * cover**2 +
                    0 * mac**2 +
                    1 * rmsp**2
                    ).sort_values()

    edisx2 = np.sqrt(1 * cover2 ** 2 +
                     1 * mac2 ** 2 +
                     0 * maxe2 ** 2 +
                     1 * rmsp2 ** 2
                     ).sort_values()

    edisx[edisx > edisp] = np.nan
    edisx2[edisx2 > edisp] = np.nan

    edisx3 =  rmsp3 + mae3
    #edisx3[edisx3 > edisp] = np.nan
    edisx3[rmsp3 >= edisp] = np.nan

    # return edisx3.idxmin(), edisx3.min()
    if edisx3.idxmin() not in runs.columns:
        rmspold = np.nan
    else:
        rmspold = rmsp3.loc[edisx3.idxmin()]
    return edisx3.idxmin(), rmspold


def pareto_skll_cov(runs, obs, use, edisp):

    mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
    maen = mae / mae.max()

    cover = coverage_loop(runs, use, obs, normalised=False)
    # normalise
    cover = (1 - cover).abs()
    cover=cover/cover.max()

    # skll
    rmse = rmse_coverage_2(runs, use, obs, normalised=False, weighted=True)
    sprd = std_coverage(runs, use, normalised=False)
    skll = rmse/sprd

    # normalise all values to smaller 1
    skll[skll > 1] = 1 / skll[skll > 1]
    # normalise to zero
    skll = (1 - skll).abs()
    skll = skll/skll.max()

    edisx = np.sqrt(cover**2 + skll**2 + maen**2)
    return edisx.idxmin(), edisp

    edisx[edisx > edisp] = np.nan

    # return edisx3.idxmin(), edisx3.min()
    if edisx.idxmin() not in runs.columns:
        edisp = np.nan
    else:
        edisp = edisx.min()




def pareto_skill(runs, obs, use, oldnskll):

    mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
    #mae = mae/mae.max()

    #rmse = rmse_coverage_2(runs, use, obs, normalised=False, weighted=False)
    sprd = std_coverage(runs, use, normalised=False)

    #skll = rmse/sprd
    skll = mae / sprd

    # normalise all values to smaller 1
    skll[skll > 1] = 1 / skll[skll > 1]

    mae[skll < oldnskll+0.02] = np.nan

    # normalise to zero
    #nskll = (1 - skll).abs()
    # normalise skill
    #nskll = nskll / nskll.max()

    #if (nskll.min() < 0.05) and (len(use) >= 5):
    #    return None, np.nan

    # only improve skill
    #nskll[nskll > oldnskll] = np.nan

    #edisx = np.sqrt(0 * nskll**2 + mae**2)

    try:
        oldnskll = skll.loc[mae.idxmin()]
    except:
        oldnskll = np.nan

    return mae.idxmin(), oldnskll


def fit_cov_and_skill2(_runs, obs, glid, minuse=5, maxuse=30):
    runs = deepcopy(_runs)

    if minuse > maxuse:
        raise ValueError

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()
    #rmsestart = rmse_weighted(pd.concat([obs, runs], axis=1), normalised=False, weighted=False).sort_values()

    #use = maestart.to_list()
    use = maestart.index[:5].to_list()
    #use = [maestart.index[0]]

    usedf = pd.DataFrame([], columns=['mae', 'maxerror', 'coverage', 'rmswg-spread','mae-spread', 'use', 'edisp', 'mape'])


    edisp = 1
    for i in range(1, maxuse+1):
        #while True:

        try:
            ens = runs.loc[:, use].mean(axis=1)
        except:
            print(glid)

        usedf.loc[i, 'mae'] = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        #usedf.loc[i, 'maxerror'] = maxerror(pd.concat([obs, ens], axis=1), normalised=False)[0]

        #rmsewg = rmse_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=False)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        #usedf.loc[i, 'rmswg-spread'] = rmsewg/sprd
        usedf.loc[i, 'mae-spread'] = usedf.loc[i,'mae'] / sprd

        usedf.loc[i, 'coverage'] = calc_coverage_2(runs, use, obs)

        #usedf.loc[i, 'mape'] = mean_abs_percentage(pd.concat([obs, ens], axis=1), normalised=False)[0]

        usedf.loc[i, 'use'] = use.copy()

        #skll = usedf.loc[i, 'rmswg-spread'].astype(float)
        #skll = 1 / skll if skll > 1 else skll
        #cov = usedf.loc[i, 'coverage'].astype(float)

        #maen = usedf.loc[i, 'mae']
        #maen = maen/maen.max()

        #edis = np.abs(1-skll) + np.abs(1-cov)
        #usedf.loc[i, 'edis'] = edis

        usedf.loc[i, 'edisp'] = edisp

        print('%s: %2d, %.2f, %.2f, %.2f' % (glid, len(use), usedf.loc[i, 'coverage'], usedf.loc[i, 'mae'], usedf.loc[i, 'mae-spread']))

        #skll = usedf.loc[i, 'rmswg-spread'].astype(float)
        skll = usedf.loc[i, 'mae-spread'].astype(float)
        skll = 1 / skll if skll > 1 else skll

        if (skll > 0.95) and (len(use) >= 5):
            return usedf.loc[i, 'use']

        #pix, edisp = pareto_only_improve2(runs, obs, use, edisp, refmae, refskill, refcov)
        pix, edisp = pareto_skill(runs, obs, use, skll)

        if np.isnan(edisp):
            return use
            #break

        use.append(pix)


        #if (i >= minuse) and (usedf.loc[i, 'coverage'] >= 0.7) and (skll >= 0.8):
        #    print('%s: %2d, %.2f, %.2f, %.2f --- break' % (
        #        glid, len(usedf.loc[i, 'use']), usedf.loc[i, 'coverage'],
        #        usedf.loc[i, 'mae'],
        #        usedf.loc[i, 'rmswg-spread']))
        #    return usedf.loc[i, 'use']

    return use

    skll = usedf.loc[5:, 'rmswg-spread'].astype(float)
    skll[skll > 1] = 1 / skll[skll > 1]
    return usedf.loc[skll.idxmax(), 'use']

    return usedf.loc[usedf.loc[:, 'edisp'].astype(float).idxmin(), 'use']
    return usedf.loc[usedf.loc[:, 'mae'].astype(float).idxmin(), 'use']

    #skll = usedf.loc[:, 'rmswg-spread'].astype(float)
    skll = usedf.loc[minuse:, 'mae-spread'].astype(float)
    skll[skll > 1] = 1 / skll[skll > 1]
    skll = (1 - skll).abs()
    #skll = skll / skll.max()

    cov = usedf.loc[minuse:, 'rmswg-spread'].astype(float)
    cov = (0.68 - cov).abs()
    # cov = cov/cov.max()

    maen = usedf.loc[minuse:, 'mae'].astype(float)/usedf.loc[minuse:, 'mae'].astype(float).max()

    mape = usedf.loc[minuse:, 'mape'].astype(float)

    alledis = np.sqrt(skll**2 + mape**2 + cov**2)

    #alledis = usedf.loc[minuse:, 'rmswg-spread'].astype(float) + usedf.loc[minuse:, 'maxerror'].astype(float) / usedf.loc[minuse:, 'maxerror'].astype(float).max()
    allid = alledis.idxmin()
    return usedf.loc[allid, 'use']

    #print('%s: %2d, %.2f' % (glid, len(usedf.loc[allid, 'use']), alledis.min()))

    #return usedf.loc[allid, 'use']

    edismx = usedf.loc[minuse:, 'edisp'].astype(float).idxmin()
    return usedf.loc[edismx, 'use']

    useindex = usedf.loc[minuse:, :].index

    mincov = min(0.7, usedf.loc[useindex, 'coverage'].max())

    useindex = usedf.loc[useindex, :].loc[usedf['coverage']>=mincov].index

    rmsp = usedf.loc[useindex, 'rmswg-spread'].astype(float)
    rmsp[rmsp > 1] = 1 / rmsp[rmsp > 1]

    id1 = rmsp.idxmax()

    return usedf.loc[id1, 'use']


def mape_coverage(df, use, obs, normalised=False):
    mape = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        mape[col] = mean_abs_percentage(pd.concat([obs, ens], axis=1), normalised=False)[0]

    if normalised:
        return mape / mape.max()
    else:
        return mape


def mean_abs_percentage(_df, normalised=False):

    df = deepcopy(_df)
    # no zero allowed
    df.loc[df['obs'] == 0, 'obs'] = np.nan

    maeall = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0)
    maeall = maeall.divide(df.loc[:, 'obs'],axis=0)
    maeall = maeall.dropna().abs().mean()

    return maeall


def optimize_skill(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    use = maestart.index[:minuse].to_list()
    #use = [maestart.index[0]]

    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        skll = mae / sprd
        skll = 1 / skll if skll > 1 else skll

        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, %.2f, %.2f, %.2f' % (glid, len(use), mae, skll, cov))

        if (skll > 0.90) and (len(use) >= 5):
            return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
        all_sprd = std_coverage(runs, use, normalised=False)

        # skill
        all_skll = all_mae / all_sprd
        # normalise all values to smaller 1
        all_skll[all_skll > 1] = 1 / all_skll[all_skll > 1]
        # increase skill by at least 2%
        thresh1 = min(skll + 0.02, 0.9)

        # coverage
        #all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before
        #thresh2 = min(cov + 0.02, 0.6)

        # filter
        optim = all_mae.copy()
        optim[all_skll < thresh1] = np.nan
        #optim[all_cov < thresh2] = np.nan

        minmaeidx = optim.idxmin()

        if pd.isna(minmaeidx):
            if len(use) < minuse:
                #minmaeidx = all_mae.idxmin()
                minmaeidx = all_skll.idxmax()
                # we run out of good members before we hit maximum skill
            else:
                return use

        if len(use) == 30:
            return use

        # if we are here: add run and go again
        use.append(minmaeidx)



def optimize_skill2(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [maestart.index[0]]

    maebest = maestart.iloc[0]


    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        skll = mae / sprd
        skll = 1 / skll if skll > 1 else skll

        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, %.2f, %.2f, %.2f' % (glid, len(use), mae, skll, cov))

        if (skll > 0.9) and (cov > 0.5) and (len(use) >= 5):
            return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
        all_sprd = std_coverage(runs, use, normalised=False)

        # skill
        all_skll = all_mae / all_sprd
        # normalise all values to smaller 1
        all_skll[all_skll > 1] = 1 / all_skll[all_skll > 1]
        # increase skill by at least 2%
        thresh1 = min(skll + 0.02, 0.9)
        all_skll[all_skll < thresh1] = np.nan

        # coverage
        all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before
        thresh2 = min(cov + 0.02, 0.5)
        all_cov[all_cov < thresh2] = np.nan

        all_mae = all_mae / all_mae.max()
        all_skll = (1-all_skll).abs()
        all_cov = (1-all_cov).abs()

        edis = np.sqrt((all_mae - all_mae.min()) ** 2 +
                       (all_cov - all_cov.min()) ** 2 +
                       (all_skll - all_skll.min()) ** 2)

        #edis = all_scale + all_cov + all_skll

        minmaeidx = edis.idxmin()

        if pd.isna(minmaeidx):
            if len(use) < minuse:
                #if all_mae.idxmin() != all_scale.idxmin():
                #    print('check')
                minmaeidx = all_mae.idxmin()
                # we run out of good members before we hit maximum skill
            else:
                return use

        # if we are here: add run and go again
        use.append(minmaeidx)


def optimize_cov(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [maestart.index[0]]

    maxcov = 0

    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]

        cov = calc_coverage_2(runs, use, obs)

        if (cov > maxcov) and (len(use) >= minuse):
            maxcov = cov
            maxcovuse = use.copy()

        print('%s: %2d, %.2f, %.2f' % (glid, len(use), mae, cov))

        if (cov > 0.68) and (len(use) >= minuse):
            return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)

        # coverage
        all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before
        thresh2 = min(cov + 0.02, 0.6)

        # filter
        optim = all_mae.copy()
        optim[all_cov < thresh2] = np.nan

        minmaeidx = optim.idxmin()

        if pd.isna(minmaeidx) or (optim.min() > mae):
            if (len(use) < minuse) or (cov < 0.68):
                minmaeidx = all_mae.idxmin()
                # we run out of good members before we hit maximum skill
            else:
                return use

        # if we are here: add run and go again
        use.append(minmaeidx)

        if len(use) == 30:
            # that is enough
            return maxcovuse



def optimize_all(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [maestart.index[0]]

    while True:
        ens = runs.loc[:, use].mean(axis=1)
        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())
        skll = mae / sprd
        skll = 1 / skll if skll > 1 else skll
        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, MAE: %.2f, SKL: %.2f, COV: %.2f' % (
        glid, len(use), mae, skll, cov))
        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
        if all_mae.min() < mae:
            use.append(all_mae.idxmin())
        else:
            #return use
            break

    # improve coverage
    while True:
        ens = runs.loc[:, use].mean(axis=1)
        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False,
                           weighted=True)[0]

        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())
        skll = mae / sprd
        skll = 1 / skll if skll > 1 else skll

        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, MAE: %.2f, SKL: %.2f, COV: %.2f' % (glid, len(use), mae, skll, cov))

        if (cov > 0.6) and (cov < 0.9) and (len(use) >= 5):
            return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)

        # coverage
        all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before

        if (all_cov == 1).all() and (len(use)>=5):
            #nothing we can do
            return use

        optim = all_mae.copy()

        if cov < 0.6:
            optim[all_cov < cov+0.02] = np.nan
        elif cov > 0.9:
            optim[all_cov > cov - 0.02] = np.nan


        minmaeidx = optim.idxmin()

        if pd.isna(minmaeidx):
            minmaeidx = all_mae.idxmin()

        # if we are here: add run and go again
        use.append(minmaeidx)


def optimize_all2(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    mae_single = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [mae_single.index[0]]


    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]
        sprd = np.sqrt(runs.loc[:, use].var(axis=1).mean())

        skll = mae / sprd
        skll = 1 / skll if skll > 1 else skll

        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, %.2f, %.2f, %.2f' % (glid, len(use), mae, skll, cov))

        if (skll > 0.9) and (cov > 0.6) and (len(use) >= 5):
            return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)
        all_sprd = std_coverage(runs, use, normalised=False)

        # skill
        all_skll = all_mae / all_sprd
        # normalise all values to smaller 1
        all_skll[all_skll > 1] = 1 / all_skll[all_skll > 1]
        # increase skill by at least 2%
        thresh1 = min(skll + 0.02, 0.9)
        all_skll[all_skll < thresh1] = np.nan

        # coverage
        all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before
        thresh2 = min(cov + 0.02, 0.6)
        all_cov[all_cov < thresh2] = np.nan

        all_mae = all_mae / all_mae.max()
        sing_mae = mae_single.loc[all_mae.index] / mae_single.loc[all_mae.index].max()
        all_skll = (1-all_skll).abs()
        all_cov = (1-all_cov).abs()

        edis = np.sqrt((all_mae - all_mae.min()) ** 2 +
                       (sing_mae - sing_mae.min()) ** 2 +
                       (all_cov - all_cov.min()) ** 2 +
                       (all_skll - all_skll.min()) ** 2)

        #edis = all_scale + all_cov + all_skll

        minmaeidx = edis.idxmin()

        if pd.isna(minmaeidx):
            if len(use) < minuse:
                #if all_mae.idxmin() != all_scale.idxmin():
                #    print('check')
                #minmaeidx = all_mae.idxmin()
                return use
                # we run out of good members before we hit maximum skill
            else:
                return use

        # if we are here: add run and go again
        use.append(minmaeidx)



def optimize_cov2(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1), weighted=True).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [maestart.index[0]]

    maxcov = 0

    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1), normalised=False, weighted=True)[0]

        cov = calc_coverage_2(runs, use, obs)

        print('%s: %2d, %.2f, %.4f' % (glid, len(use), mae, cov))

        #if (cov > 0.68) and (len(use) >= minuse):
        #    return use

        # Add a run:
        all_mae = mae_coverage_2(runs, use, obs, normalised=False, weighted=True)

        # coverage
        all_cov = coverage_loop(runs, use, obs, normalised=False)
        # coverage should be > 0.6 or at least higher than before
        #thresh1 = max(cov - 0.02, 0.9)
        #thresh1 = 0.9
        #thresh2 = min(cov + 0.01, 0.6)
        thresh2 = min(cov, 0.6)

        # filter
        optim = all_mae.copy()
        #optim[all_cov > thresh1] = np.nan
        optim[all_cov <= thresh2] = np.nan

        minmaeidx = optim.idxmin()

        if optim.min() < mae:
            use.append(minmaeidx)
        elif optim.min() > mae:
            if (len(use) < minuse) or (cov < 0.6):
                use.append(minmaeidx)
            else:
                return use
        elif pd.isna(minmaeidx):
            if (len(use) < minuse):
                print('this is not documented in the paper')
                use.append(all_mae.idxmin())
            elif (cov < 0.6):
                print('enough members, did not reach cov=0.6, but thats ok')
                return use
            else:
                print('No increasing members, but already enough and cov>0.6, we are good to go.')
                return use

        if len(use) == 30:
            # that is enough and should not happend anywys
            raise ValueError('that should not happen')
