import numpy as np
import pandas as pd
import ast
from copy import deepcopy

from relic.length_observations import get_length_observations


def relative_length_change(meta, spinup, histrun):
    spin = (spinup.loc[:] - spinup.loc[0]).dropna().iloc[-1]
    dl = spin + meta['dL2003']
    # relative length change
    rel_dl = histrun.loc[:] - histrun.iloc[0] + dl

    return rel_dl


def mae_weighted(_df):

    df = deepcopy(_df)

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
    maeall = maeall.mul(wgh, axis=0).sum()/sum(wgh)

    return maeall


def runs2df(runs):

    # get all glaciers
    glcs = []
    for run in runs:
        glcs += [gl['rgi_id'] for gl in list(run.values())[0]]
    glcs = np.unique(glcs).tolist()

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

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == mrgi][0]
            except IndexError:
                continue

            if np.isnan(rdic['tbias']):
                continue

            rkey = list(run.keys())[0]

            df.loc[rdic['rel_dl'].index, rkey] = rdic['rel_dl'].values

            tbias_series[rkey] = rdic['tbias']

        glcdict[mrgi] = df
        tbiasdict[mrgi] = tbias_series

    return glcdict, tbiasdict


def coverage_loop(df, use, obs):

    coverage = pd.Series()

    for col, val in df.iteritems():

        # don't use anything twice
        if col in use:
            continue

        nucov = calc_coverage(df, use + [col], obs)
        coverage[col] = nucov

    return coverage


def calc_coverage(df, use, obs):

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


def mae_coverage(df, use, obs):
    maecover = pd.Series()
    for col, val in df.iteritems():
        # don't use anything twice
        if col in use:
            continue
        ens = df.loc[:, use + [col]].mean(axis=1)

        maecover[col] = mae_weighted(pd.concat([obs, ens], axis=1))[0]

    return maecover


def optimize_cov(_runs, obs, glid, minuse=5):
    runs = deepcopy(_runs)

    # list which indices to use
    maestart = mae_weighted(pd.concat([obs, runs], axis=1)).sort_values()

    #use = maestart.index[:minuse].to_list()
    use = [maestart.index[0]]

    while True:

        ens = runs.loc[:, use].mean(axis=1)

        mae = mae_weighted(pd.concat([obs, ens], axis=1))[0]

        cov = calc_coverage(runs, use, obs)

        print('%s: %2d, %.2f, %.4f' % (glid, len(use), mae, cov))

        # Add a run:
        all_mae = mae_coverage(runs, use, obs)

        # coverage
        all_cov = coverage_loop(runs, use, obs)
        # coverage should be > 0.6 or at least higher than before
        thresh2 = min(cov, 0.6)

        # filter
        optim = all_mae.copy()
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
                print('No increasing members, but already enough and cov>0.6,',
                      ' we are good to go.')
                return use

        if len(use) == 30:
            # that is enough and should not happend anywys
            raise ValueError('that should not happen')
