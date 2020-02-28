import os
import pandas as pd
import urllib
import numpy as np

from relic.preprocessing import GLCDICT


def get_wgms(rgiids, firstyear=1850):

    wgms_c = pd.read_csv(os.path.join(
        os.path.dirname(__file__), 'WGMS-FoG-2018-11-C-FRONT-VARIATION.csv'))

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(firstyear, 2020))

    for rgi in rgiids:
        if GLCDICT.get(rgi)[0] != 'wgms':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # WGMS ID
        wid = GLCDICT.get(rgi)[1]

        # select glacier from wgms data
        glc = wgms_c.loc[wgms_c['WGMS_ID'] == wid, :]
        t0 = np.floor(glc['REFERENCE_DATE']/10000)
        t0 = t0[t0 >= firstyear].iloc[0]

        # secial case
        if (rgi == 'RGI60-11.00897') and (firstyear <= 1855):
            # Hintereisferner max extent is missing in WGMS
            # Span et al 1997:
            t0 = 1855

        yr = glc['Year']
        yr = yr[yr > t0]
        dl = glc['FRONT_VARIATION'][yr.index].cumsum()

        # new dataframe for complete time series
        df = pd.Series(index=np.arange(firstyear, 2020))
        df.loc[yr] = dl.values
        df.loc[t0] = 0

        # get first measurement
        dfmeta.loc[rgi, 'first'] = t0

        # write data
        dfdata.loc[rgi] = df

        # meta stuff
        dfmeta.loc[rgi, 'name'] = GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    return dfmeta, dfdata


def get_glamos(rgiids, firstyear=1850):

    glamos = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      'lengthchange.csv'),
                         header=6)

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(firstyear, 2020))

    for rgi in rgiids:
        if GLCDICT.get(rgi)[0] != 'glamos':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # GLAMOS ID
        gid = GLCDICT.get(rgi)[1]

        # select glacier from glamos data
        glc = glamos.loc[glamos['glacier id'] == gid, :]

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

        elif rgi == 'RGI60-11.02051':
            # Tschierva: add Roseg length
            roseg = glamos.loc[glamos['glacier id'] == 'E23/11', :]
            df = add_merge_length(df, roseg)

        elif rgi == 'RGI60-11.02709':
            # Mine: add Ferpecle length
            ferp = glamos.loc[glamos['glacier id'] == 'B72/11', :]
            df = add_merge_length(df, ferp)

        # get first measurement
        dfmeta.loc[rgi, 'first'] = df.dropna().index[0]

        # write data
        dfdata.loc[rgi] = df

        # meta stuff
        dfmeta.loc[rgi, 'name'] = GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    return dfmeta, dfdata


def get_leclercq(rgiids):

    # --- Download and read Leclercq Files ---
    leclercqhttp = 'https://folk.uio.no/paulwl/downloads/'
    infofile = 'ALL_glacierinfo.txt'
    recordsfile = 'ALL_records.txt'
    urllib.request.urlretrieve(leclercqhttp + infofile, infofile)
    urllib.request.urlretrieve(leclercqhttp + recordsfile, recordsfile)
    lq_info = pd.read_csv(infofile, delimiter='\t')
    lq_rec = pd.read_csv(recordsfile, delimiter='\t', na_values='-99999')

    out = []

    y1850 = np.arange(1850, 2020)

    # loop over all Leclercq glaciers:
    lq_names = lq_rec.columns[0:-1:3]
    for glc in lq_names:

        # ---- 1. get Leclercq length record data
        # column in length record data
        icol = lq_rec.columns.get_loc(glc)
        # actual data
        year = np.array(lq_rec.iloc[1:, icol].dropna(), dtype=int)
        dL = np.array(lq_rec.iloc[1:, icol+1].dropna(), dtype=float)

        # ---- 2. get Leclercq glacier info
        # name and record number
        num = int(glc.split()[0])
        recname = glc.split()[1:]
        current_glc = lq_info.loc[lq_info['num '] == num]
        infoname = current_glc['name '].values[0]
        regio = current_glc['region '].values[0]

        # At the moment: only do Leclercq region 6
        if regio != 6:
            continue

        LID = current_glc['ID '].values[0]

        # check for consistency
        if 1:
            if not all(x in infoname for x in recname):
                raise ValueError('Name Error while processing %s' % infoname)
            elif not current_glc['first '].values[0] == year[0]:
                raise ValueError('First Year Error while processing %s' % infoname)
            elif not current_glc['last '].values[0] == year[-1]:
                raise ValueError('Last Year Error while processing %s' % infoname)

        # Save general information
        tmp = pd.DataFrame([], index=[LID], columns=y1850, dtype=float)
        tmp[year[year >= 1850]] = dL[year >= 1850]
        out.append(tmp)

    data = pd.concat(out)

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=y1850)

    for rgi in rgiids:
        if prepro.GLCDICT_old.get(rgi)[0] != 'leclercq':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # Leclercq ID
        lid = prepro.GLCDICT_old.get(rgi)[1]

        # get first measurement
        dfmeta.loc[rgi, 'first'] = data.loc[lid].dropna().index[0]

        # write data, and make it relative
        dfdata.loc[rgi] = data.loc[lid]
        dfdata.loc[rgi] -= dfdata.loc[rgi, dfmeta.loc[rgi, 'first']]

        # meta stuff
        dfmeta.loc[rgi, 'name'] = prepro.GLCDICT_old.get(rgi)[2]
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


def add_merge_length(main, merge):

    # find first year of observation
    t0 = pd.DatetimeIndex(merge['start date of observation']).year[0]
    dl = merge['length change'].astype(float).cumsum()
    yr = pd.DatetimeIndex(merge['end date of observation']).year

    if np.any(yr < 1850):
        raise ValueError

    # new dataframe for complete time series
    df = pd.Series(index=np.arange(1850, 2020))
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
