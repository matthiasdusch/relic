import os
import pandas as pd
import numpy as np

from relic import preprocessing as pp


def get_wgms(rgiids, firstyear=1850, reconstruction=False):

    if reconstruction:
        wgms = pd.read_csv(os.path.join(
            os.path.dirname(__file__),
            'WGMS-FoG-2019-12-RR-RECONSTRUCTION-FRONT-VARIATION.csv'))
    else:
        wgms = pd.read_csv(os.path.join(
            os.path.dirname(__file__),
            'WGMS-FoG-2019-12-C-FRONT-VARIATION.csv'), decimal=',')

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(firstyear, 2020))

    for rgi in rgiids:
        if reconstruction:
            wid = pp.WGMS_RR.get(rgi)
        else:
            glcd = pp.GLCDICT.get(rgi)
            if glcd[0] != 'wgms':
                # remove from meta and out and continue
                dfdata.drop(rgi)
                dfmeta.drop(rgi)
                continue
            wid = glcd[1]

        # select glacier from wgms data
        glc = wgms.loc[wgms['WGMS_ID'] == wid, :]
        if reconstruction:
            t0 = glc['REFERENCE_YEAR']
            yr = glc['YEAR']
        else:
            t0 = np.floor(glc['REFERENCE_DATE']/10000)
            yr = glc['Year']
        t0 = t0[t0 >= firstyear].iloc[0]

        # secial cases
        if (rgi == 'RGI60-11.00897') and (firstyear <= 1855):
            # Hintereisferner max extent is missing in WGMS
            # Span et al 1997:
            t0 = 1855

        yr = yr[yr > t0]
        dl = glc['FRONT_VARIATION'][yr.index].astype(float).cumsum()

        # new dataframe for complete time series
        df = pd.Series(index=np.arange(firstyear, 2020))
        df.loc[yr] = dl.values
        df.loc[t0] = 0

        # use reconstruction for Bossons and Mer de Glace
        if (not reconstruction) and (
                (rgi == 'RGI60-11.03646') or (rgi == 'RGI60-11.03643')):
            # add reconstruction data
            df = add_observations(df, rgi, firstyear)

        # get first measurement
        dfmeta.loc[rgi, 'first'] = t0

        # write data
        dfdata.loc[rgi] = df

        # meta stuff
        dfmeta.loc[rgi, 'name'] = pp.GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    return dfmeta, dfdata


def get_glamos(rgiids, firstyear=1850):

    glamos = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      'lengthchange.csv'),
                         sep=';', header=4)

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(firstyear, 2020))

    for rgi in rgiids:
        if pp.GLCDICT.get(rgi)[0] != 'glamos':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # GLAMOS ID
        gid = pp.GLCDICT.get(rgi)[1]

        # select glacier from glamos data
        glc = glamos.loc[glamos['glacier id'] == gid, :]
        # save init first year for some few merge scenarios
        t_init = pd.DatetimeIndex(glc['start date of observation']).year[0]

        # select only suitable times
        glc = glc.loc[pd.DatetimeIndex(glc['start date of observation']).
                      year >= firstyear]

        # find first year of observation
        t0 = pd.DatetimeIndex(glc['start date of observation']).year[0]
        dl = glc['length change'].astype(float).cumsum()
        yr = pd.DatetimeIndex(glc['end date of observation']).year

        # new dataframe for complete time series
        df = pd.Series(index=np.arange(firstyear, 2020))
        df.loc[yr] = dl.values
        df.loc[t0] = 0

        # special cases
        if gid == 'A54l/19':
            # Unterer Grindelwald
            # Andreas Bauder Email 02.10.2019
            df.loc[2007] = df.loc[1983] - 230
            df.loc[2008:] -= 230

            # add reconstruction data
            df = add_reconstruction(df, rgi, t0, firstyear)

        elif rgi == 'RGI60-11.01270':
            # Oberer Grindelwald
            # Tecnically there is a data gap between 1999 and 2003
            df.loc[2003] = df.loc[1999]

            # add reconstruction data
            df = add_reconstruction(df, rgi, t0, firstyear)

        elif rgi == 'RGI60-11.02051':
            # Tschierva: add Roseg length
            roseg = glamos.loc[glamos['glacier id'] == 'E23/11', :]
            if firstyear < t_init:
                df = add_merge_length(df, roseg, firstyear)

        elif rgi == 'RGI60-11.02709':
            # Mine: add Ferpecle length
            ferp = glamos.loc[glamos['glacier id'] == 'B72/11', :]
            if firstyear < t_init:
                df = add_merge_length(df, ferp, firstyear)

        # get first measurement
        dfmeta.loc[rgi, 'first'] = df.dropna().index[0]

        # write data
        dfdata.loc[rgi] = df

        # meta stuff
        dfmeta.loc[rgi, 'name'] = pp.GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    return dfmeta, dfdata


def _get_2003_dl(glc, name=None):

    if np.isnan(glc.loc[2003]):
        try:
            minarg = np.abs(glc.loc[1999:2007].dropna().index-2003).argmin()
            dl2003 = glc.loc[1999:2007].dropna().iloc[minarg]
        except ValueError:
            print('No measurement around 2003... %s' % name)
            dl2003 = np.nan
    else:
        dl2003 = glc.loc[2003]

    return dl2003


def add_merge_length(main, merge, firstyear=1850):
    # select only suitable times
    merge = merge.loc[pd.DatetimeIndex(merge['start date of observation']).
                      year >= firstyear]

    # find first year of observation
    t0 = pd.DatetimeIndex(merge['start date of observation']).year[0]
    dl = merge['length change'].astype(float).cumsum()
    yr = pd.DatetimeIndex(merge['end date of observation']).year

    # new dataframe for complete time series
    df = pd.Series(index=np.arange(firstyear, 2020))
    df.loc[yr] = dl.values
    df.loc[t0] = 0

    # find first year of short observation
    t0_short = main.dropna().index[0]

    # find closest point in longer time series
    try:
        _ = df.dropna().loc[t0_short]
        t0_long = t0_short
    except KeyError:
        minarg = np.abs(df.loc[t0_short-5:t0_short+5].
                        dropna().index - t0_short).argmin()
        t0_long = df.loc[t0_short-5:t0_short+5].dropna().index[minarg]
        print('name t0 = %d (instead of %d)' % (t0_long, t0_short))

    # reference lenght at splitting time
    dl_ref = df.loc[t0_long]

    # reset data from t0_short onwards
    df.loc[t0_short:] = np.nan

    # new length is ref + dl_short
    df.loc[t0_short] = dl_ref
    df.loc[main.dropna().index] = dl_ref + main.dropna().values

    assert (df.index.unique() == df.index).all()

    return df


def add_reconstruction(df, rec_id, t0, firstyear):
    meta, data = get_wgms([rec_id], reconstruction=True,
                          firstyear=firstyear)
    diff = data.iloc[0].loc[data.iloc[0].index <= t0]
    df.loc[diff.index] = diff
    df.loc[df.index > t0] += diff.dropna().iloc[-1]

    return df


def add_observations(df, rec_id, firstyear):
    meta, data = get_wgms([rec_id], reconstruction=True,
                          firstyear=firstyear)

    t0 = data.iloc[0].dropna().index[-1]

    diff1 = data.iloc[0].loc[data.iloc[0].index <= t0]
    diff2 = df.loc[df.index >= t0] - df.loc[t0]

    df.loc[diff1.index] = diff1
    df.loc[df.index >= t0] = df.loc[t0] + diff2

    return df


def get_length_observations(rgiids, firstyear=1850):

    # lec_meta, lec_data = get_leclercq(rgiids)
    glam_meta, glam_data = get_glamos(rgiids, firstyear=firstyear)
    wgms_meta, wgms_data = get_wgms(rgiids, firstyear=firstyear)

    # meta, data = select_my_glaciers(meta_all, data_all)
    meta = glam_meta.dropna().append(wgms_meta.dropna())
    data = glam_data.dropna(axis=0, how='all').append(
        wgms_data.dropna(axis=0, how='all'))

    meta['first'] = meta['first'].astype(int)
    meta['measurements'] = meta['measurements'].astype(int)
    meta['dL2003'] = meta['dL2003'].astype(int)
    data.columns = data.columns.astype(int)

    return meta, data
