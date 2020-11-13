import numpy as np
import pandas as pd
import ast
from copy import deepcopy
import pickle
import os
import json
import xarray as xr

from relic.length_observations import get_length_observations
from relic.preprocessing import GLCDICT, MERGEDICT
from relic import preprocessing

from oggm import utils, cfg, tasks, GlacierDirectory
from oggm.core.climate import compute_ref_t_stars
from oggm.workflow import execute_entity_task


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


def get_ensemble_length(rgi, histalp_storage, future_storage,
                        ensemble_filename, meta):

    df = pd.DataFrame([], index=np.arange(1850, 3000))

    for i in np.arange(999):

        rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)

        try:
            sp = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_spinup_{:02d}.nc'.format(i)))
            hi = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_histalp_{:02d}.nc'.format(i)))
        except FileNotFoundError:
            break

        sp = sp.length_m.to_dataframe()['length_m']
        hi = hi.length_m.to_dataframe()['length_m']
        df.loc[:, i] = relative_length_change(meta, sp, hi)

    ensemble_count = i
    # future
    for i in np.arange(ensemble_count):

        fut = xr.open_dataset(os.path.join(future_storage, rgi,
                                           ensemble_filename.format(i)))

        fut = fut.length_m.to_dataframe()['length_m']
        fut.index = fut.index + 2014
        df.loc[2015:, i] = (fut - fut.iloc[0] + df.loc[2014, i]).loc[2015:]

    return df


def get_rcp_ensemble_length(rgi, histalp_storage, future_storage,
                            rcp, meta):
    cmip = ['CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'CanESM2',
            'GFDL-CM3', 'GFDL-ESM2G', 'GISS-E2-R', 'IPSL-CM5A-LR',
            'MPI-ESM-LR', 'NorESM1-M']

    dfrcp = pd.DataFrame([], index=np.arange(1850, 2101))

    for i in np.arange(999):

        rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                               rgi[:8], rgi[:11], rgi)

        try:
            sp = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_spinup_{:02d}.nc'.format(i)))
            hi = xr.open_dataset(
                os.path.join(rgipath,
                             'model_diagnostics_histalp_{:02d}.nc'.format(i)))
        except FileNotFoundError:
            break

        sp = sp.length_m.to_dataframe()['length_m']
        hi = hi.length_m.to_dataframe()['length_m']
        dfrcp.loc[:, i] = relative_length_change(meta, sp, hi)

    nr_ensemblemembers = i

    # projection
    for i in np.arange(nr_ensemblemembers):

        df_cm = pd.DataFrame()
        for cmi in cmip:
            try:
                cm = xr.open_dataset(
                    os.path.join(future_storage, rgi,
                                 'model_diagnostics_{}_{}_{:02d}.nc'.
                                 format(cmi, rcp, i)))
            except FileNotFoundError:
                continue

            df_cm.loc[:, cmi] = cm.length_m.to_dataframe()['length_m']

        cm = df_cm.mean(axis=1)
        dfrcp.loc[2015:, i] = \
            (cm - cm.iloc[0] + dfrcp.loc[2014, i]).loc[2015:]

    return dfrcp


def glacier_to_table1(outpath):
    pd.options.display.max_colwidth = 100
    pd.options.display.float_format = '{:.2f}'.format
    df = pd.DataFrame([], index=GLCDICT.keys())

    poldict = {'Switzerland': 'CH',
               'Austria': 'AT',
               'Italy': 'IT',
               'France': 'FR'}

    rgidf = utils.get_rgi_glacier_entities(df.index)
    meta, _ = get_length_observations(df.index)

    for rgi, _ in df.iterrows():
        df.loc[rgi, 'name'] = GLCDICT[rgi][2]
        df.loc[rgi, 'country'] = poldict[GLCDICT[rgi][3]]
        df.loc[rgi, 'lon/lat'] = '{:.2f}/{:.2f}'.\
            format(rgidf.loc[rgidf.RGIId == rgi, 'CenLon'].iloc[0],
                   rgidf.loc[rgidf.RGIId == rgi, 'CenLat'].iloc[0])
        df.loc[rgi, 'lon'] = rgidf.loc[rgidf.RGIId == rgi, 'CenLon'].iloc[0]

        area = rgidf.loc[rgidf.RGIId == rgi, 'Area'].iloc[0]

        df.loc[rgi, 'RGI id'] = rgi
        if MERGEDICT.get(rgi):
            df.loc[rgi, 'RGI id'] = '{}, {}'.\
                format(rgi,
                       str(MERGEDICT[rgi][0]).strip('[]').replace("'", ''))

            tribs = MERGEDICT[rgi][0]
            tribdf = utils.get_rgi_glacier_entities(tribs)
            for trib in tribs:
                area += tribdf.loc[tribdf.RGIId == trib, 'Area'].iloc[0]

        df.loc[rgi, 'areakm2'] = '{:.1f}'.format(area)
        df.loc[rgi, '1.obs'] = '{:.0f}'.format(meta.loc[rgi, 'first'])
        df.loc[rgi, '#obs'] = '{:.0f}'.format(meta.loc[rgi, 'measurements'])

    df = df.sort_values('lon')

    df.index = np.arange(1, 31)

    # ---------------
    titl = ('All glaciers used for this study. Index corresponds '
            'to location in Figure~\\ref{fig:map} and is also indicated in '
            'the title of individual plots. '
            'Multiple RGI ids indicate that the glacier is merged with one or '
            'more tributary glaciers. '
            'For merged glaciers \\emph{area} then also includes the '
            'tributaries. '
            '\\emph{1.obs} refers to the first observation after 1850 and'
            ' the number of observations \\emph{\\#obs} is counted until '
            '2020.'
            )
    # generate table
    tbl = df.to_latex(na_rep='--', index=True, longtable=False,
                      columns=['name', 'country', 'RGI id', 'lon/lat',
                               'areakm2', '1.obs', '#obs'],
                      column_format='r' + 'p{35mm}' + 'l' + 'p{45mm}' + 4*'r',
                      caption=titl,
                      label='tbl:glaciers')

    # set fontsize to footnotesize
    tbl = tbl.replace('\n', '\n\\footnotesize\n', 1)

    # and sone manual line breaks
    tbl = tbl.replace('with Glacier', '\\newline with Glacier')
    tbl = tbl.replace('with Mittel', '\\newline with Mittel')
    tbl = tbl.replace('with Vadret', '\\newline with Vadret')
    tbl = tbl.replace('Obersulzbachkees', 'Obersulzbachkees\\newline')
    tbl = tbl.replace('RGI60-11.00168', '\\newline RGI60-11.00168')
    tbl = tbl.replace('RGI60-11.00213', '\\newline RGI60-11.00213')
    tbl = tbl.replace('country', '')
    tbl = tbl.replace('name', 'name and country')
    tbl = tbl.replace('areakm2', 'area [km$^2$]')
    with open(os.path.join(outpath, 'table1.tex'), 'w') as tf:
        tf.write(tbl)


def glacier_to_table2(histalp_storage, comit_storage, tbiasdictpath,
                      outpath):
    pd.options.display.max_colwidth = 100
    pd.options.display.float_format = '{:.2f}'.format
    wd = utils.get_temp_dir()
    tbiasdict = pickle.load(open(tbiasdictpath, 'rb'))

    # Initialize OGGM
    cfg.initialize()
    cfg.PATHS['working_dir'] = wd
    utils.mkdir(wd, reset=True)
    cfg.PARAMS['baseline_climate'] = 'HISTALP'
    # and set standard histalp values
    cfg.PARAMS['temp_melt'] = -1.75

    rgis = list(GLCDICT.keys())

    allmeta, allobs = get_length_observations(rgis)
    gdirs = preprocessing.configure(wd, rgis, resetwd=True)

    # get glacier
    ref_gdirs = [GlacierDirectory(refid) for
                 refid in preprocessing.ADDITIONAL_REFERENCE_GLACIERS]

    # get glaciers up and running
    compute_ref_t_stars(ref_gdirs + gdirs)
    task_list = [tasks.local_t_star,
                 tasks.mu_star_calibration,
                 tasks.prepare_for_inversion,
                 tasks.mass_conservation_inversion,
                 tasks.filter_inversion_output,
                 tasks.init_present_time_glacier
                 ]

    for task in task_list:
        execute_entity_task(task, gdirs)

    dfout = pd.DataFrame([])
    dfplot = pd.DataFrame([])
    for gdir in gdirs:

        rgi = gdir.rgi_id

        try:
            meta = allmeta.loc[rgi]
        except:
            continue

        if preprocessing.merge_pair_dict(rgi) is not None:
            rgi += '_merged'

        l2003 = gdir.read_pickle('model_flowlines')[-1].length_m
        dfout.loc[rgi, 'ly0'] = (l2003 - meta['dL2003'])/1000

        fn85 = 'model_diagnostics_commitment1885_{:02d}.nc'
        df85 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn85,
                                   meta)
        ensmean = df85.mean(axis=1)
        prelength = ensmean.dropna().iloc[-30:].mean()
        dfout.loc[rgi, 'dl 1885'] = prelength

        fn99 = 'model_diagnostics_commitment1999_{:02d}.nc'
        df99 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn99,
                                   meta)
        ensmean = df99.mean(axis=1)
        postlength = ensmean.dropna().iloc[-30:].mean()
        dfout.loc[rgi, 'dl 1999'] = postlength

        fn70 = 'model_diagnostics_commitment1970_{:02d}.nc'
        df70 = get_ensemble_length(rgi, histalp_storage, comit_storage, fn70,
                                   meta)
        ensmean = df70.mean(axis=1)
        y70length = ensmean.dropna().iloc[-30:].mean()
        dfout.loc[rgi, 'dl 1970'] = y70length

        dfplot.loc[rgi, 'dl 1885-1970'] = (prelength - y70length)/1000

        rndic = pickle.load(
            open(os.path.join(histalp_storage, 'runs_{}.p'.format(rgi)), 'rb'))

        f = gdir.get_filepath('climate_historical')
        with utils.ncDataset(f) as nc:
            time = nc.variables['time'][:]
            clim = pd.DataFrame([], index=pd.Timestamp(
                '1801-01-01') + pd.TimedeltaIndex(time, 'day'))
            clim['prcp'] = nc.variables['prcp'][:]
            clim['temp'] = nc.variables['temp'][:]

        tempjja = clim.loc[clim.index.month.isin([6, 7, 8]), 'temp']
        tempjja = tempjja.groupby(tempjja.index.year).mean()

        tempsom = clim.loc[clim.index.month.isin([5, 6, 7, 8, 9]), 'temp']
        tempsom = tempsom.groupby(tempsom.index.year).mean()

        prcpjja = clim.loc[clim.index.month.isin([6, 7, 8]), 'prcp']
        prcpjja = prcpjja.groupby(prcpjja.index.year).sum()

        prcpsom = clim.loc[clim.index.month.isin([5, 6, 7, 8, 9]), 'prcp']
        prcpsom = prcpsom.groupby(prcpsom.index.year).sum()

        prcpdjf = clim.loc[clim.index.month.isin([1, 2, 12]), 'prcp']
        prcpdjf.index = prcpdjf.index + pd.offsets.MonthBegin(1)
        prcpdjf = prcpdjf.groupby(prcpdjf.index.year).sum()

        tempdjf = clim.loc[clim.index.month.isin([1, 2, 12]), 'temp']
        tempdjf.index = tempdjf.index + pd.offsets.MonthBegin(1)
        tempdjf = tempdjf.groupby(tempdjf.index.year).mean()

        prcpwin = clim.loc[
            clim.index.month.isin([1, 2, 12, 11, 10, 3, 4]), 'prcp']
        prcpwin.index = prcpwin.index + pd.offsets.MonthBegin(3)
        prcpwin = prcpwin.groupby(prcpwin.index.year).sum()

        tempwin = clim.loc[
            clim.index.month.isin([1, 2, 12, 11, 10, 3, 4]), 'temp']
        tempwin.index = tempwin.index + pd.offsets.MonthBegin(3)
        tempwin = tempwin.groupby(tempwin.index.year).mean()

        # reference: 1984-2014
        tjjaref = tempjja.loc[1984:2014].mean()
        tsomref = tempsom.loc[1984:2014].mean()
        twinref = tempwin.loc[1984:2014].mean()
        tdjfref = tempdjf.loc[1984:2014].mean()

        pdjfref = prcpdjf.loc[1984:2014].mean()
        pwinref = prcpwin.loc[1984:2014].mean()
        psomref = prcpsom.loc[1984:2014].mean()
        pjjaref = prcpjja.loc[1984:2014].mean()

        # preindust
        tjjapre = tempjja.loc[1870:1900].mean() - tjjaref
        tsompre = tempsom.loc[1870:1900].mean() - tsomref
        twinpre = tempwin.loc[1870:1900].mean() - twinref
        tdjfpre = tempdjf.loc[1870:1900].mean() - tdjfref

        pdjfpre = prcpdjf.loc[1870:1900].mean() - pdjfref
        pwinpre = prcpwin.loc[1870:1900].mean() - pwinref
        psompre = prcpsom.loc[1870:1900].mean() - psomref
        pjjapre = prcpjja.loc[1870:1900].mean() - pjjaref

        # 1960-1980
        tjja60 = tempjja.loc[1960:1980].mean() - tjjaref
        tsom60 = tempsom.loc[1960:1980].mean() - tsomref
        twin60 = tempwin.loc[1960:1980].mean() - twinref
        tdjf60 = tempdjf.loc[1960:1980].mean() - tdjfref

        pdjf60 = prcpdjf.loc[1960:1980].mean() - pdjfref
        pwin60 = prcpwin.loc[1960:1980].mean() - pwinref
        psom60 = prcpsom.loc[1960:1980].mean() - psomref
        pjja60 = prcpjja.loc[1960:1980].mean() - pjjaref

        # dfout.loc[rgi, 'T JJA 1984-2014'] = tref

        dfout.loc[rgi, 'tjja pre'] = tjjapre
        dfout.loc[rgi, 'tjja 70'] = tjja60
        dfout.loc[rgi, 'pdjf pre'] = pdjfpre
        dfout.loc[rgi, 'pdjf 70'] = pdjf60

        dfplot.loc[rgi, 'dt jja'] = tjjapre - tjja60
        dfplot.loc[rgi, 'dt som'] = tsompre - tsom60
        dfplot.loc[rgi, 'dt win'] = twinpre - twin60
        dfplot.loc[rgi, 'dt djf'] = tdjfpre - tdjf60

        dfplot.loc[rgi, 'dp djf'] = pdjfpre - pdjf60
        dfplot.loc[rgi, 'dp win'] = pwinpre - pwin60
        dfplot.loc[rgi, 'dp som'] = psompre - psom60
        dfplot.loc[rgi, 'dp jja'] = pjjapre - pjja60

        dfout.loc[rgi, 'dt spinup'] = tbiasdict[rgi].loc[
            rndic['ensemble']].mean()

        # dfout.loc[rgi, 'dT spin min'] = tbiasdict[rgi].loc[rndic['ensemble']].min()
        # dfout.loc[rgi, 'dT spin max'] = tbiasdict[rgi].loc[rndic['ensemble']].max(

        mu = pd.DataFrame()
        ela85 = pd.DataFrame()
        ela70 = pd.DataFrame()
        ela99 = pd.DataFrame()
        for i in np.arange(99):
            rgipath = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                                   rgi[:8], rgi[:11], rgi)

            if 'merged' in rgi:
                fname = 'local_mustar_{}.json'.format(rgi.split('_')[0])
            else:
                fname = 'local_mustar.json'

            fp = os.path.join(rgipath, fname)
            try:
                with open(fp, 'r') as f:
                    out = json.load(f)

                # ela
                com85 = xr.open_dataset(
                    os.path.join(
                        comit_storage, rgi,
                        'model_diagnostics_commitment1885_{:02d}.nc'.format(
                            i)))
                com70 = xr.open_dataset(
                    os.path.join(
                        comit_storage, rgi,
                        'model_diagnostics_commitment1970_{:02d}.nc'.format(
                            i)))
                com99 = xr.open_dataset(
                    os.path.join(
                        comit_storage, rgi,
                        'model_diagnostics_commitment1999_{:02d}.nc'.format(
                            i)))

            except FileNotFoundError:
                break

            mu.loc[i, 'mu'] = out['mu_star_flowline_avg']
            ela85.loc[:, i] = com85.to_dataframe()['ela_m']
            ela70.loc[:, i] = com70.to_dataframe()['ela_m']
            ela99.loc[:, i] = com99.to_dataframe()['ela_m']

        dfout.loc[rgi, 'mu mean'] = mu['mu'].mean()
        dfout.loc[rgi, 'ELA 1885'] = int(ela85.mean(axis=1).iloc[-30:].mean())
        dfout.loc[rgi, 'ELA 1970'] = int(ela70.mean(axis=1).iloc[-30:].mean())
        dfout.loc[rgi, 'ELA 1999'] = int(ela99.mean(axis=1).iloc[-30:].mean())

    rgidf = utils.get_rgi_glacier_entities(rgis)
    for rgi, _ in dfout.iterrows():
        rgiid = rgi.split('_')[0]
        name = GLCDICT[rgiid][2].split('(')[0]
        dfout.loc[rgi, 'name'] = name
        dfout.loc[rgi, 'lon'] = rgidf.loc[rgidf.RGIId == rgiid,
                                          'CenLon'].iloc[0]
        dfplot.loc[rgi, 'lon'] = rgidf.loc[rgidf.RGIId == rgiid,
                                           'CenLon'].iloc[0]

    # plot climate vs lengthchange
    from relic.graphics import climate_vs_lengthchange
    climate_vs_lengthchange(dfplot, outpath)

    df = dfout.sort_values('lon')
    df.index = np.arange(1, 31)
    # ---------------
    # formaters

    def _str(x):
        return str(x)

    def _float2(x):
        return '{:.2f}'.format(x)

    def _f2int(x):
        return '{:.0f}'.format(x)

    def _(x):
        return x

    # generate table
    titl = ('$\\Delta l_{1885 (1970, 1999)}$ are the ensemble mean length '
            'changes with respect to the length $l_{y0}$ of the first '
            'observed year (see Tbl.~\\ref{tbl:glaciers}) under a '
            'randomized climate around the years 1870-1900, 1960-1980 and '
            '1984-2014 respectively.'
            '$\\Delta T^{JJA}_{ref-1885 (1970)}$ are the temperature '
            'difference between these periods and a reference period defined '
            'as 1984-2014 as derived from the HISTALP climate data. '
            '$\\Delta T_{spinup}$ is the ensemble mean temperature bias which '
            'was applied '
            'on top of the constant mean climate of the reference period to '
            ' grow the glacier to their post-LIA size. '
            'And $\\overline{ELA_{1885 (1970, 1999)}}$ are the ensemble mean '
            ' equilibrium line altitudes of the randomized climate '
            'simulations around the respective years.'
            )
    tbl = df.to_latex(na_rep='--', index=True, longtable=False,
                      column_format='r' + 'p{18mm}' + 12*'r',
                      columns=['name', 'ly0',
                               'dl 1885', 'dl 1970', 'dl 1999',
                               'tjja pre', 'tjja 70',
                               'pdjf pre', 'pdjf 70',
                               'dt spinup',
                               'ELA 1885', 'ELA 1970', 'ELA 1999'],
                      formatters=[_str, _float2,
                                  _f2int, _f2int, _f2int,
                                  _float2, _float2,
                                  _f2int, _f2int,
                                  _float2,
                                  _f2int, _f2int,
                                  _f2int] + [_ for i in range(2)],
                      caption=titl,
                      label='tbl:climate')

    # fontsize
    tbl = tbl.replace('\n', '\n\\scriptsize\n', 1)

    tbl = tbl.replace('Unterer Grindelwaldgletscher', 'U. Grindelwaldgl.')
    tbl = tbl.replace('Oberer Grindelwaldgletscher', 'O. Grindelwaldgl.')
    tbl = tbl.replace('Großer Aletschgletscher with Mittelaletschgletscher',
                      'Aletschgl.')
    tbl = tbl.replace('Vadret da Morteratsch', 'Morteratsch')
    tbl = tbl.replace('Vadret da Tschierva with Vadret da Roseg', 'Tschierva')
    tbl = tbl.replace('Glacier du Mont Mine with Glacier de Ferpecle',
                      'Mont Mine')
    tbl = tbl.replace('Mer de Glace with Glacier de Leschaux', 'Mer de Glace')
    tbl = tbl.replace('Ghiacciaio dei Forni', 'Forni')
    tbl = tbl.replace('Vadrec del Forno', 'Forno')
    tbl = tbl.replace('Glacier de ', '')
    tbl = tbl.replace('Glacier des', '')
    tbl = tbl.replace('(with tributary)', '')
    tbl = tbl.replace('(with tributaries)', '')

    tbl = tbl.replace('Glacier', 'Gl.')
    tbl = tbl.replace('gletscher', 'gl.')
    tbl = tbl.replace('dT (spinup-jjaref)', 'spinjja')
    tbl = tbl.replace('dT (spinup-somref)', 'spinsom')

    tbl = tbl.replace('{table}', '{sidewaystable}')

    tbl = tbl.replace('dl 1885', '$\\Delta l_{1885}$')
    tbl = tbl.replace('dl 1970', '$\\Delta l_{1970}$')
    tbl = tbl.replace('dl 1999', '$\\Delta l_{1999}$')
    tbl = tbl.replace('tjja pre', '$\\Delta T^{JJA}_{ref-1885}$')
    tbl = tbl.replace('tjja 70', '$\\Delta T^{JJA}_{T_{ref-1970}}$')
    tbl = tbl.replace('pdjf pre', '$\\Delta P^{DJF}_{ref-1885}$')
    tbl = tbl.replace('pdjf 70', '$\\Delta P^{DJF}_{T_{ref-1970}}$')
    tbl = tbl.replace('dt spinup', '$\\Delta T_{spinup}$')
    tbl = tbl.replace('ELA 1885', '$\\overline{ELA_{1885}}$')
    tbl = tbl.replace('ELA 1970', '$\\overline{ELA_{1970}}$')
    tbl = tbl.replace('ELA 1999', '$\\overline{ELA_{1999}}$')

    # add line with units
    tbl = tbl.replace('\\midrule',
                      ('& & [km$^2$] &[m] & [m] & [m] & $[^\\circ C]$ & '
                       '$[^\\circ C]$ & $[^\\circ C]$ & '
                       '[mm] & [mm] & '
                       '[m] & [m] & [m] '
                       '\\\\\n\\midrule'))

    with open(os.path.join(outpath, 'table2.tex'), 'w') as tf:
        tf.write(tbl)


def oldglacier_to_table(outpath):

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
    #with open(os.path.join(outpath, 'table2.tex'), 'w') as tf:
    #    tf.write(tbl)
