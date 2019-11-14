import xarray as xr
import numpy as np
import pandas as pd

from oggm import utils, cfg, workflow, tasks
from oggm.core.climate import compute_ref_t_stars
from oggm.workflow import execute_entity_task
from oggm import entity_task


import logging
log = logging.getLogger(__name__)

MERGEDICT = {
             # Tschierva: Roseg
             'RGI60-11.02051': ['RGI60-11.02119', 'glamos', 'E23/11', 8],
             # Mine: ferpecle
             'RGI60-11.02709': ['RGI60-11.02715', 'glamos', 'B72/11', 2.5]
            }


def configure(workdir, glclist, baselineclimate='HISTALP'):
    # Initialize OGGM
    cfg.initialize()

    # Local working directory (where OGGM will write its output)
    utils.mkdir(workdir, reset=True)
    cfg.PATHS['working_dir'] = workdir

    # Use multiprocessing?
    cfg.PARAMS['use_multiprocessing'] = True

    # How many grid points around the glacier?
    # Make it large if you expect your glaciers to grow large
    cfg.PARAMS['border'] = 160

    # Set to True for operational runs
    cfg.PARAMS['continue_on_error'] = False

    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_rgi_area'] = True

    # set negative flux filtering to false. should be standard soon
    cfg.PARAMS['filter_for_neg_flux'] = False
    cfg.PARAMS['correct_for_neg_flux'] = True

    gdirs = workflow.init_glacier_regions(glclist, from_prepro_level=3)

    # climate
    if baselineclimate == 'CRU':
        cfg.PARAMS['prcp_scaling_factor'] = 2.5
        cfg.PARAMS['temp_melt'] = -1.0

    if baselineclimate == 'HISTALP':
        cfg.PARAMS['baseline_climate'] = baselineclimate
        # and set standard histalp values
        cfg.PARAMS['prcp_scaling_factor'] = 1.75
        cfg.PARAMS['temp_melt'] = -1.75
        execute_entity_task(tasks.process_histalp_data, gdirs)

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
