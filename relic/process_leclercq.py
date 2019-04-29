import pandas as pd
import urllib
import numpy as np


def download_leclercq():

    # --- Download and read Leclercq Files ---
    leclercqhttp = 'https://folk.uio.no/paulwl/downloads/'
    infofile = 'ALL_glacierinfo.txt'
    recordsfile = 'ALL_records.txt'
    urllib.request.urlretrieve(leclercqhttp + infofile, infofile)
    urllib.request.urlretrieve(leclercqhttp + recordsfile, recordsfile)
    lq_info = pd.read_csv(infofile, delimiter='\t')
    lq_rec = pd.read_csv(recordsfile, delimiter='\t', na_values='-99999')

    out = []
    meta = []

    y1850 = np.arange(1850, 2011)

    # loop over all Leclercq glaciers:
    lq_names = lq_rec.columns[0:-1:3]
    for glc in lq_names:

        # ---- 1. get Leclercq length record data
        # column in length record data
        icol = lq_rec.columns.get_loc(glc)
        # actual data
        year = np.array(lq_rec.iloc[1:, icol].dropna(), dtype=int)
        dL = np.array(lq_rec.iloc[1:, icol+1].dropna(), dtype=int)
        src = np.array(lq_rec.iloc[1:, icol+2].dropna(), dtype=int)

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

        # important data
        LID = current_glc['ID '].values[0]
        LAT = current_glc['lat '].values[0]
        LON = current_glc['lon '].values[0]

        # check for consistency
        if 1:
            if not all(x in infoname for x in recname):
                raise ValueError('Name Error while processing %s' % infoname)
            elif not current_glc['first '].values[0] == year[0]:
                raise ValueError('First Year Error while processing %s' % infoname)
            elif not current_glc['last '].values[0] == year[-1]:
                raise ValueError('Last Year Error while processing %s' % infoname)

        # Save general information

        meta.append(pd.DataFrame([[infoname, LON, LAT]], index=[LID],
                                 columns=['name', 'lon', 'lat']))
        tmp = pd.DataFrame([], index=[LID], columns=y1850)
        tmp[year[year >= 1850]] = dL[year >= 1850]
        out.append(tmp)

    dfout = pd.concat(out)
    dfmeta = pd.concat(meta)
    dfout.sort_index(axis=1, inplace=True)
    dfmeta.index.name = 'LID'
    dfout.index.name = 'LID'

    for nr, glc in dfout.iterrows():
        dfmeta.loc[nr, 'first'] = glc.dropna().index[0]
        dfout.loc[nr] -= glc.loc[dfmeta.loc[nr, 'first']]
        dfmeta.loc[nr, 'measurements'] = glc.dropna().size

        if np.isnan(glc.loc[2003]):
            try:
                minarg = np.abs(glc.loc[2000:2006].dropna().index-2003).argmin()
                dfmeta.loc[nr, 'dL2003'] = glc.loc[2000:2006].dropna().iloc[minarg]
            except ValueError:
                print('No measurement around 2003... %s' % dfmeta.loc[nr, 'name'])
                dfmeta.loc[nr, 'dL2003'] = np.nan

        else:
            dfmeta.loc[nr, 'dL2003'] = glc.loc[2003]

    return dfmeta, dfout


def select_my_glaciers(meta, data):

    # TODO do something smart and automated here!

    meta.loc[23, 'RGI_ID'] = 'RGI60-11.03646'
    meta.loc[137, 'RGI_ID'] = 'RGI60-11.01238'
    meta.loc[133, 'RGI_ID'] = 'RGI60-11.00106'
    meta.loc[79, 'RGI_ID'] = 'RGI60-11.00897'
    meta.loc[109, 'RGI_ID'] = 'RGI60-11.03643'
    meta.loc[66, 'RGI_ID'] = 'RGI60-11.01450'
    meta.loc[123, 'RGI_ID'] = 'RGI60-11.01270'
    meta.loc[140, 'RGI_ID'] = 'RGI60-11.02119'
    meta.loc[62, 'RGI_ID'] = 'RGI60-11.00746'
    meta.loc[7, 'RGI_ID'] = 'RGI60-11.03638'
    meta.loc[52, 'RGI_ID'] = 'RGI60-11.02245'
    meta.loc[51, 'RGI_ID'] = 'RGI60-11.01974'
    meta.loc[432, 'RGI_ID'] = 'RGI60-11.02916'
    meta.loc[99, 'RGI_ID'] = 'RGI60-11.00929'
    meta.loc[73, 'RGI_ID'] = 'RGI60-11.00887'
    meta.loc[47, 'RGI_ID'] = 'RGI60-11.02715'  # ferpacle

    # --- neu ---
    # meta.loc[183, 'RGI_ID'] = 'RGI60-11.01328'  # unteraar
    meta.loc[118, 'RGI_ID'] = 'RGI60-11.00992'  # nierderjoch
    meta.loc[196, 'RGI_ID'] = 'RGI60-11.02630'  # Zinal, vorsicht mit Zuordnung
    meta.loc[144, 'RGI_ID'] = 'RGI60-11.02793'  # Saleina
    meta.loc[49, 'RGI_ID'] = 'RGI60-11.01478'  # Fiescher
    meta.loc[98, 'RGI_ID'] = 'RGI60-11.01698'  # Langgletscher
    # meta.loc[82, 'RGI_ID'] = 'RGI60-11.00872'  # Hüfi
    meta.loc[64, 'RGI_ID'] = 'RGI60-11.02822'  # Gorner
    meta.loc[4, 'RGI_ID'] = 'RGI60-11.02704'  # Allalin
    meta.loc[176, 'RGI_ID'] = 'RGI60-11.02755'  # Tsidjore Nouve
    meta.loc[171, 'RGI_ID'] = 'RGI60-11.02740'  # Trient

    meta = meta.loc[meta.loc[:, 'RGI_ID'].dropna().index]
    data = data.loc[meta.index]

    return meta, data
