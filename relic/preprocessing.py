import xarray as xr
import numpy as np
import pandas as pd

from oggm import utils, cfg, workflow, tasks
from oggm.core.climate import compute_ref_t_stars
from oggm.workflow import execute_entity_task
from oggm import entity_task

import logging
log = logging.getLogger(__name__)


def configure(workdir, glclist, glena_factor=1.5, baselineclimate='HISTALP'):
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
    if baselineclimate == 'HISTALP':
        cfg.PARAMS['baseline_climate'] = baselineclimate
        execute_entity_task(tasks.process_histalp_data, gdirs)

    if baselineclimate == 'HISTALP_ANNUAL_MEAN':
        cfg.PARAMS['baseline_climate'] = 'HISTALP'
        execute_entity_task(tasks.process_histalp_data, gdirs)

        cfg.PARAMS['baseline_climate'] = baselineclimate
        execute_entity_task(histalp_annual_mean, gdirs)

        cfg.PARAMS['run_mb_calibration'] = True
        compute_ref_t_stars(gdirs)

    execute_entity_task(tasks.local_t_star, gdirs)
    execute_entity_task(tasks.mu_star_calibration, gdirs)
    execute_entity_task(tasks.init_present_time_glacier, gdirs)

    return gdirs


@entity_task(log, writes=['climate_monthly', 'climate_info'])
def histalp_annual_mean(gdir):
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

    meanpcp = cm.prcp.groupby('time.month').mean().copy()

    for mo in np.arange(1, 13):
        prcp[time.month == mo] = meanpcp.sel(month=mo).values

    gdir.write_monthly_climate_file(time, prcp, temp, hgt, lon, lat,
                                    gradient=igrad)

    # metadata
    out = {'baseline_climate_source': 'HISTALP annual mean',
           'baseline_hydro_yr_0': time[0].year + 1,
           'baseline_hydro_yr_1': time[-1].year}
    gdir.write_pickle(out, 'climate_info')
