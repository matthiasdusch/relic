import os
import pandas as pd
import urllib
import numpy as np

from relic.preprocessing import GLCDICT, GLCDICT_old


def get_wgms(rgiids):

    wgms = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                    'WGMS-FoG-2018-11-C-FRONT-VARIATION.csv'))

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(1850, 2020))

    for rgi in rgiids:
        if GLCDICT.get(rgi)[0] != 'wgms':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # WGMS ID
        wid = GLCDICT.get(rgi)[1]

        # select glacier from glamos data
        glc = wgms.loc[wgms['WGMS_ID'] == wid, :]

        # TODO make sure end-start in GLAMOS is 1 year!!!
        # find first year of observation
        # t0 = pd.DatetimeIndex(glc['start date of observation']).year[0]

        yr = glc['Year']
        yr = yr[yr >= 1850]
        dl = glc['FRONT_VARIATION'][yr.index]
        dl.iloc[0] = 0
        dl = dl.cumsum()

        # new dataframe for complete time series
        df = pd.Series(index=np.arange(1850, 2020))
        df.loc[yr] = dl.values

        # get first measurement
        dfmeta.loc[rgi, 'first'] = yr.iloc[0]

        # write data
        dfdata.loc[rgi] = df

        # meta stuff
        dfmeta.loc[rgi, 'name'] = GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    return dfmeta, dfdata


def get_glamos(rgiids):

    glamos = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      'lengthchange.csv'),
                         header=6)

    dfmeta = pd.DataFrame([], index=rgiids, columns=['name', 'first',
                                                     'measurements', 'dL2003'])
    dfdata = pd.DataFrame([], index=rgiids, columns=np.arange(1850, 2020))

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

        # TODO make sure end-start in GLAMOS is 1 year!!!
        # find first year of observation
        t0 = pd.DatetimeIndex(glc['start date of observation']).year[0]
        dl = glc['length change'].astype(float).cumsum()
        yr = pd.DatetimeIndex(glc['end date of observation']).year

        # TODO make sure first year !>= 1850
        # TODO return some meta data

        # new dataframe for complete time series
        df = pd.Series(index=np.arange(1850, 2020))
        df.loc[yr] = dl.values
        df.loc[t0] = 0

        # get first measurement
        dfmeta.loc[rgi, 'first'] = t0

        # write data
        dfdata.loc[rgi] = df

        # special cases
        # Unterer Grindelwald
        if gid == 'A54l/19':

            # Bauder Email 02.10.2019
            dfdata.loc[rgi, 2007] = dfdata.loc[rgi, 1983] - 230
            dfdata.loc[rgi, 2008:] -= 230

        # meta stuff
        dfmeta.loc[rgi, 'name'] = GLCDICT.get(rgi)[2]
        dfmeta.loc[rgi, 'measurements'] = len(dfdata.loc[rgi].dropna())

        # get 2003 length change
        dfmeta.loc[rgi, 'dL2003'] = _get_2003_dl(dfdata.loc[rgi],
                                                 dfmeta.loc[rgi, 'name'])

    """
            # lets linearely interpolate a 2003 value for U. Grindelwald...
            dfout.loc[nr, 2003] = dfout.loc[nr].interpolate().loc[2003]
            glc.loc[2003] = dfout.loc[nr].interpolate().loc[2003]
    """
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
        if GLCDICT_old.get(rgi)[0] != 'leclercq':
            # remove from meta and out and continue
            dfdata.drop(rgi)
            dfmeta.drop(rgi)
            continue

        # Leclercq ID
        lid = GLCDICT_old.get(rgi)[1]

        # get first measurement
        dfmeta.loc[rgi, 'first'] = data.loc[lid].dropna().index[0]

        # write data, and make it relative
        dfdata.loc[rgi] = data.loc[lid]
        dfdata.loc[rgi] -= dfdata.loc[rgi, dfmeta.loc[rgi, 'first']]

        # meta stuff
        dfmeta.loc[rgi, 'name'] = GLCDICT_old.get(rgi)[2]
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



def add_merge_length(meta, data, ids):

    glamos = pd.read_csv(os.path.join(os.path.dirname(__file__),
                                      'lengthchange.csv'),
                         header=6)
    raise RuntimeError('todo')
    for rgi in ids:

        if rgi == 'RGI60-11.02051':
            name = 'Vadret da Tschierva'
            merge_id = 'RGI60-11.02119'  # Roseg
        elif rgi == 'RGI60-11.02709':
            name = 'Glacier du Mont Miné'
            merge_id = 'RGI60-11.02715'  # Ferpecle
        else:
            raise ValueError('no data implemented')

        # LID of the glacier with the longer observation record
        merge_lid = meta.loc[meta.RGI_ID == merge_id].index[0]

        glc = get_glamos(name)

        # find first year of short observation
        t0_short = glc.dropna().index[0]

        # new dataframe for complete time series from the other glacier
        tmp = data.loc[merge_lid].copy()

        # find closest point in longer time series
        try:
            _ = data.loc[merge_lid].dropna().loc[t0_short]
            t0_long = t0_short
        except KeyError:
            minarg = np.abs(data.loc[merge_lid, t0_short-5:t0_short+5].
                            dropna().index - t0_short).argmin()
            t0_long = data.loc[merge_lid, t0_short-5:t0_short+5].\
                dropna().index[minarg]
            print('name t0 = %d (instead of %d)' % (t0_long, t0_short))

        # reference lenght at splitting time
        dl_ref = tmp.loc[t0_long]

        # reset data from t0_short onwards
        tmp.loc[t0_short:] = np.nan

        # new length is ref + dl_short
        tmp.loc[t0_short] = dl_ref
        tmp.loc[glc.dropna().index] = dl_ref + glc.dropna().values

        assert (tmp.index.unique() == tmp.index).all()

        # find 2003 length
        try:
            dl2003 = tmp.dropna().loc[2003]
        except KeyError:
            minarg = np.abs(tmp.loc[2000:2006].dropna().index-2003).argmin()
            dl2003 = tmp.loc[2000:2006].dropna().iloc[minarg]

        meta.loc[-1*int(rgi.split('.')[-1])] = {'name': name,
                                                'lon': -99,
                                                'lat': -99,
                                                'first': tmp.dropna().index[0],
                                                'measurements': len(tmp),
                                                'dL2003': dl2003,
                                                'RGI_ID': rgi}

        data.loc[-1*int(rgi.split('.')[-1])] = tmp.values

    return meta, data


def get_length_observations(rgiids):
    lec_meta, lec_data = get_leclercq(rgiids)
    gla_meta, gla_data = get_glamos(rgiids)

    meta, data = select_my_glaciers(meta_all, data_all)

    meta, data = add_merge_length(meta, data,
                                  ['RGI60-11.02051', 'RGI60-11.02709'])

    meta['first'] = meta['first'].astype(int)
    data.columns = data.columns.astype(int)

    return meta, data
