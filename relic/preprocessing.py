import xarray as xr
import numpy as np
import pandas as pd

from oggm import utils, cfg, workflow, tasks
from oggm.workflow import execute_entity_task
from oggm import entity_task
from oggm.utils import get_ref_mb_glaciers_candidates

from relic import length_observations


import logging
log = logging.getLogger(__name__)

MERGEDICT = {
             # Tschierva: Roseg
             'RGI60-11.02051': [['RGI60-11.02119'], 8],
             # Mine: ferpecle
             'RGI60-11.02709': [['RGI60-11.02715'], 2.5],
             # Venedigerkees (Obersulzbachkees)
             'RGI60-11.00116': [['RGI60-11.00141', 'RGI60-11.00168',
                                 'RGI60-11.00127'], 10.5],
             # Mer de Glace: Leschaux
             'RGI60-11.03643': [['RGI60-11.03642'], 7.5],
             # Großer Aletsch: Mittelaletsch
             'RGI60-11.01450': [['RGI60-11.01797'], 6],
             # HEF: KWF
             'RGI60-11.00897': [['RGI60-11.00787'], 5],
             # Pasterze: Waserfall, Hofmann
             'RGI60-11.00106': [['RGI60-11.00122', 'RGI60-11.00213'], 8.5],
             # Huefifirn
             'RGI60-11.00872': [['RGI60-11.00981'], 2.5]
            }

# stores [observation source, source ID, Plotname]
GLCDICT = {
    'RGI60-11.00106': ['wgms', 566, 'Pasterze', 'Austria'],
    'RGI60-11.00116': ['wgms', 583, 'Obersulzbachkees', 'Austria'],
    'RGI60-11.00687': ['wgms', 519, 'Taschachferner', 'Austria'],
    'RGI60-11.00746': ['wgms', 522, 'Gepatschferner', 'Austria'],
    'RGI60-11.00887': ['wgms', 511, 'Gurgler Ferner', 'Austria'],
    'RGI60-11.00897': ['wgms', 491, 'Hintereisferner (with Kesselwandferner)',
                       'Austria'],

    'RGI60-11.01238': ['glamos', 'B43/03', 'Rhonegletscher', 'Switzerland'],
    'RGI60-11.01270': ['glamos', 'A54l/04', 'Oberer Grindelwaldgletscher',
                       'Switzerland'],
    'RGI60-11.01328': ['glamos', 'A54g/11', 'Unteraargletscher',
                       'Switzerland'],
    'RGI60-11.01346': ['glamos', 'A54l/19',
                       'Unterer Grindelwaldgletscher', 'Switzerland'],
    'RGI60-11.01450': ['glamos', 'B36/26',
                       'Großer Aletschgletscher (with Mittelaletsch Gl.)',
                       'Switzerland'],
    'RGI60-11.01478': ['glamos', 'B40/07', 'Fieschergletscher', 'Switzerland'],
    'RGI60-11.01698': ['glamos', 'B31/04', 'Langgletscher', 'Switzerland'],
    'RGI60-11.01946': ['glamos', 'E22/03', 'Vadret da Morteratsch',
                       'Switzerland'],

    'RGI60-11.01974': ['wgms', 670, 'Ghiacciaio dei Forni', 'Italy'],

    'RGI60-11.02051': ['glamos', 'E23/06',
                       'Vadret da Tschierva (with Roseg)', 'Switzerland'],
    'RGI60-11.02709': ['glamos', 'B72/15',
                       'Glacier du Mont Mine (with Ferpecle)', 'Switzerland'],
    'RGI60-11.02245': ['glamos', 'C83/12', 'Vadrec del Forno', 'Switzerland'],
    'RGI60-11.02630': ['glamos', 'B63/05', 'Glacier de Zinal', 'Switzerland'],
    'RGI60-11.02704': ['glamos', 'B52/29', 'Allalingletscher', 'Switzerland'],
    'RGI60-11.02740': ['glamos', 'B90/02', 'Glacier du Trient', 'Switzerland'],
    'RGI60-11.02755': ['glamos', 'B73/16', 'Glacier de Tsijiore Nouve',
                       'Switzerland'],
    'RGI60-11.02766': ['glamos', 'B83/03', 'Glacier de Corbassiere',
                       'Switzerland'],
    'RGI60-11.02793': ['glamos', 'B85/16', 'Glacier de Saleinaz',
                       'Switzerland'],
    'RGI60-11.02822': ['glamos', 'B56/07', 'Gornergletscher', 'Switzerland'],
    'RGI60-11.00872': ['glamos', 'A51d/10', 'Huefifirn', 'Switzerland'],

    'RGI60-11.03638': ['wgms', 354, 'Glacier de Argentiere', 'France'],
    'RGI60-11.03643': ['wgms', 353, 'Mer de Glace (with Leschaux)',
                       'France'],
    'RGI60-11.03646': ['wgms', 355, 'Glacier des Bossons', 'France'],
    'RGI60-11.03684': ['wgms', 351, 'Glacier Blanc', 'France']
}

ADDITIONAL_REFERENCE_GLACIERS = []


def glacier_to_table(outpath):

    df = pd.DataFrame([], index=GLCDICT.keys())

    poldict = {'Switzerland': 'CH',
               'Austria': 'AT',
               'Italy': 'IT',
               'France': 'FR'}

    rgidf = utils.get_rgi_glacier_entities(df.index)
    meta, _ = length_observations.get_length_observations(df.index)

    for rgi, _ in df.iterrows():
        name = GLCDICT[rgi][2].split('(')[0]
        df.loc[rgi, 'name'] = name
        #df.loc[rgi, 'RGI id'] = rgi
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


def configure(workdir, glclist, baselineclimate='HISTALP', resetwd=False):
    global MERGEDICT
    global GLCDICT
    global ADDITIONAL_REFERENCE_GLACIERS

    # Initialize OGGM
    cfg.initialize()
    cfg.PATHS['working_dir'] = workdir

    # Local working directory (where OGGM will write its output)
    utils.mkdir(workdir, reset=resetwd)

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = False

    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_rgi_area'] = True

    # set negative flux filtering to false. should be standard soon
    cfg.PARAMS['filter_for_neg_flux'] = False
    cfg.PARAMS['correct_for_neg_flux'] = True

    # here in relic we want to run the mb calibration every time
    cfg.PARAMS['run_mb_calibration'] = True

    # check if we want to merge a glacier
    mglclist = []
    for glc in glclist:
        mglc = merge_pair_dict(glc)
        if mglc is not None:
            mglclist += mglc[0]

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = 160

    gdirs = workflow.init_glacier_regions(glclist + mglclist,
                                          from_prepro_level=3)

    # and we want to use all glaciers for the MB calibration
    refids = get_ref_mb_glaciers_candidates()
    # right now we only do Alpine glaciers
    refids = [rid for rid in refids if '-11.' in rid]
    # but do leave out the actual glaciers
    refids = [rid for rid in refids if rid not in glclist + mglclist]
    # I SAID ALPS, NOT PYRENEES
    refids.remove('RGI60-11.03232')
    refids.remove('RGI60-11.03209')
    refids.remove('RGI60-11.03241')
    # initialize the reference glaciers with a small border
    ref_gdirs = workflow.init_glacier_regions(rgidf=refids,
                                              from_prepro_level=3,
                                              prepro_border=10)
    # save these ids for later
    ADDITIONAL_REFERENCE_GLACIERS = refids

    # climate
    if baselineclimate == 'CRU':
        cfg.PARAMS['prcp_scaling_factor'] = 2.5
        cfg.PARAMS['temp_melt'] = -1.0

    if baselineclimate == 'HISTALP':
        cfg.PARAMS['baseline_climate'] = baselineclimate
        # and set standard histalp values
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.75
        # run histalp climate on all glaciers!
        execute_entity_task(tasks.process_histalp_data, gdirs + ref_gdirs)

    # TODO: if I do use custom climate stuff like histalp_annual_mean:
    #   ->>>> look back at commits before 1.10.2019

    return gdirs


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def histalp_annual_mean(gdir, y0=None, years=None):
    """
    docstring

    Parameters
    ----------
    gdir

    Returns
    -------

    """

    # read original histalp climate monthly
    cm = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    time = pd.DatetimeIndex(cm.time.values)
    temp = cm.temp.values
    prcp = cm.prcp.values.copy()
    hgt = cm.ref_hgt
    lon = cm.ref_pix_lon
    lat = cm.ref_pix_lat
    igrad = None
    if cfg.PARAMS['temp_use_local_gradient']:
        raise RuntimeError('Have to think about it')

    if y0 is None:
        meanpcp = cm.prcp.groupby('time.month').mean().copy()
    else:
        assert isinstance(y0, int)
        sy0 = pd.to_datetime('%d' % y0)
        if years is None:
            meanpcp = cm.prcp.loc[sy0:].groupby('time.month').mean().copy()
        else:
            assert isinstance(years, int)
            sy1 = pd.to_datetime('%d' % (y0+years))
            meanpcp = cm.prcp.loc[sy0:sy1].groupby('time.month').mean().copy()

    for mo in np.arange(1, 13):
        prcp[time.month == mo] = meanpcp.sel(month=mo).values

    gdir.write_monthly_climate_file(time, prcp, temp, hgt, lon, lat,
                                    gradient=igrad)

    # metadata
    out = {'baseline_climate_source': 'HISTALP annual mean',
           'baseline_hydro_yr_0': time[0].year + 1,
           'baseline_hydro_yr_1': time[-1].year}
    gdir.write_pickle(out, 'climate_info')


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def annual_temperature_from_summer_temp(gdir, y0=1950, years=30):
    """

    Parameters
    ----------
    gdir
    y0
    years

    Returns
    -------

    """

    # read original histalp climate monthly
    cm = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    time = pd.DatetimeIndex(cm.time.values)
    temp = cm.temp.copy()
    prcp = cm.prcp.values
    hgt = cm.ref_hgt
    lon = cm.ref_pix_lon
    lat = cm.ref_pix_lat
    igrad = None
    if cfg.PARAMS['temp_use_local_gradient']:
        raise RuntimeError('Have to think about it')

    # get temeratures of reference period
    tref = temp.loc[pd.to_datetime('%d' % y0):
                    pd.to_datetime('%d' % (y0+years))][:-1].copy()

    # monthly mean temperature of reference period
    ref_monthly = tref.groupby('time.month').mean()
    # summer mean temperature of reference period -> one value
    ref_jja = ref_monthly.loc[(ref_monthly.month >= 6) &
                              (ref_monthly.month <= 8)].mean().values

    # remove non-summer temperatures
    temp.loc[(temp.time.dt.month > 8) | (temp.time.dt.month < 6)] = np.nan

    # yearly series of summer mean temeratures
    tmean = temp.groupby('time.year').mean()

    # and create monthly time series with yearly bias
    for y in tmean:
        bias = ref_jja - y.values
        temp.loc[temp.time.dt.year == y.year] = ref_monthly.loc[
            temp.loc[temp.time.dt.year == y.year].time.dt.month] - bias

    temp = temp.values

    # write climate data
    gdir.write_monthly_climate_file(time, prcp, temp, hgt, lon, lat,
                                    gradient=igrad)

    # metadata
    out = {'baseline_climate_source': 'HISTALP jja temps',
           'baseline_hydro_yr_0': time[0].year + 1,
           'baseline_hydro_yr_1': time[-1].year}
    gdir.write_pickle(out, 'climate_info')


def merge_pair_dict(mainglc):
    return MERGEDICT.get(mainglc.strip('_merged'))
