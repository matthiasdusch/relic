import numpy as np
import pandas as pd
import ast
from copy import deepcopy

from relic.length_observations import get_length_observations
from relic.preprocessing import GLCDICT, MERGEDICT

from oggm import utils


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


def runs2df(runs, min_pcpsf=0):

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

    tribdict = {}

    for rgi, mrgi in zip(rgi_ids, glcs):
        _meta = meta.loc[rgi].copy()
        _data = data.loc[rgi].copy()

        df = pd.DataFrame([], index=np.arange(1850, 2020))
        df.loc[_data.index, 'obs'] = _data

        tbias_series = pd.Series([])

        trib = pd.DataFrame([], index=np.arange(1850, 2020))

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == mrgi][0]
            except IndexError:
                continue

            if np.isnan(rdic['tbias']):
                continue

            rkey = list(run.keys())[0]

            par = ast.literal_eval('{' + rkey + '}')
            if par['prcp_scaling_factor'] < min_pcpsf:
                print(rkey)
                continue

            df.loc[rdic['rel_dl'].index, rkey] = rdic['rel_dl'].values
            try:
                trib.loc[rdic['trib_dl'].index, rkey] = rdic['trib_dl'].values
            except:
                pass

            tbias_series[rkey] = rdic['tbias']

        glcdict[mrgi] = df
        tbiasdict[mrgi] = tbias_series
        tribdict[mrgi] = trib

    return glcdict, tbiasdict, tribdict


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


def glacier_to_table(outpath):

    df = pd.DataFrame([], index=GLCDICT.keys())

    poldict = {'Switzerland': 'CH',
               'Austria': 'AT',
               'Italy': 'IT',
               'France': 'FR'}

    rgidf = utils.get_rgi_glacier_entities(df.index)
    meta, _ = get_length_observations(df.index)

    for rgi, _ in df.iterrows():
        name = GLCDICT[rgi][2].split('(')[0]
        df.loc[rgi, 'name'] = name
        df.loc[rgi, 'state'] = poldict[GLCDICT[rgi][3]]
        df.loc[rgi, 'lat/lon'] = '{:.2f}/{:.2f}'.\
            format(rgidf.loc[rgidf.RGIId == rgi, 'CenLon'].iloc[0],
                   rgidf.loc[rgidf.RGIId == rgi, 'CenLat'].iloc[0])

        df.loc[rgi, 'merge'] = 'no'
        area = rgidf.loc[rgidf.RGIId == rgi, 'Area'].iloc[0]

        if MERGEDICT.get(rgi):
            df.loc[rgi, 'merge'] = 'yes'
            tribs = MERGEDICT[rgi][0]
            tribdf = utils.get_rgi_glacier_entities(tribs)
            for trib in tribs:
                area += tribdf.loc[tribdf.RGIId == trib, 'Area'].iloc[0]

        df.loc[rgi, 'area [insert km2]'] = '{:.1f}'.\
            format(area)
        df.loc[rgi, '1.obs'] = meta.loc[rgi, 'first']
        df.loc[rgi, '#obs'] = meta.loc[rgi, 'measurements']

    df.loc[:, '1.obs'] = df.loc[:, '1.obs'].astype(int)
    df.loc[:, '#obs'] = df.loc[:, '#obs'].astype(int)

    df = df.sort_values('name')

    # ---------------
    # generate table
    tbl = df.to_latex(na_rep='--', index=False, longtable=True,
                      column_format=2 * 'l' + 'r' + 'l' + 3 * 'r')
    # add title
    titl = ('\n\\caption{A list of all glaciers used for this study. '
            '\\emph{merge} indicates if'
            ' a glacier has additional tributary glaciers merged to it. '
            '\\emph{area} then does include these tributaries. '
            '\\emph{1.obs} refers to the first observation after 1850 and'
            ' the number of observations \\emph{\#obs} is counted until '
            '2020.}\\\\\n'
            '\\label{tbl:glaciers}\\\\\n')
    tbl = tbl.replace('\n', titl, 1)
    with open(outpath, 'w') as tf:
        tf.write(tbl)
