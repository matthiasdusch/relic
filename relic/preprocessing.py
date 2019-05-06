import xarray as xr
import numpy as np
import pandas as pd

from oggm import utils, cfg, workflow, tasks
from oggm.core.climate import compute_ref_t_stars
from oggm.workflow import execute_entity_task
from oggm import entity_task

from relic.process_leclercq import download_leclercq, select_my_glaciers

import logging
log = logging.getLogger(__name__)


def configure(workdir, glclist, glena_factor=1.5, baselineclimate='HISTALP',
              annual_mean_prcp=False, jja_temp=False, prcp_sf=None,
              y0=None, years=None):
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

    # and set standard histalp values
    cfg.PARAMS['prcp_scaling_factor'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75

    gla = cfg.PARAMS['glen_a']
    # set glan a to nan just do make sure there are no global problems
    cfg.PARAMS['glen_a'] = gla*glena_factor

    gdirs = workflow.init_glacier_regions(glclist, from_prepro_level=3)

    # climate
    if baselineclimate == 'CRU':
        cfg.PARAMS['prcp_scaling_factor'] = 2.5
        cfg.PARAMS['temp_melt'] = -1.0

    if baselineclimate == 'HISTALP':
        cfg.PARAMS['baseline_climate'] = baselineclimate
        execute_entity_task(tasks.process_histalp_data, gdirs)

    if annual_mean_prcp is True:
        execute_entity_task(histalp_annual_mean, gdirs, y0=y0, years=years)

    if jja_temp is True:
        execute_entity_task(annual_temperature_from_summer_temp, gdirs)

    if prcp_sf is not None:
        cfg.PARAMS['run_mb_calibration'] = True
        cfg.PARAMS['prcp_scaling_factor'] = prcp_sf
        compute_ref_t_stars(gdirs)

    execute_entity_task(tasks.local_t_star, gdirs)
    execute_entity_task(tasks.mu_star_calibration, gdirs)
    execute_entity_task(tasks.init_present_time_glacier, gdirs)

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


def get_leclercq_observations(firstyear=None):
    meta_all, data_all = download_leclercq(firstyear=firstyear)
    meta, data = select_my_glaciers(meta_all, data_all)
    meta['first'] = meta['first'].astype(int)
    data.columns = data.columns.astype(int)
    return meta, data
