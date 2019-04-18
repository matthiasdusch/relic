import numpy as np
import pandas as pd
import xarray as xr

from oggm import tasks, cfg
from oggm.core.flowline import FileModel
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance)
from oggm.workflow import execute_entity_task
from oggm.core.climate import compute_ref_t_stars

from relic.spinup import spinup_with_tbias, minimize_dl


def minimize_glena(glena, gdir, meta, obs_ye, obs_dl, optimization):

    # finish OGGM tasks
    tasks.mass_conservation_inversion(gdir, glen_a=glena)
    tasks.filter_inversion_output(gdir)
    tasks.init_present_time_glacier(gdir)

    # --------- HOW SHALL WE SPIN ---------------

    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003'].iloc[0]

    tbias = spinup_with_tbias(gdir, fls, dl, len2003, glena=glena)
    tmp_mod = FileModel(gdir.get_filepath('model_run',
                                          filesuffix='_spinup_%.3e' % glena))
    tmp_mod.run_until(tmp_mod.last_yr)

    # --------- HIST IT DOWN ---------------
    tasks.run_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                                init_model_fls=tmp_mod.fls,
                                output_filesuffix='_histalp_%.3e' % glena,
                                glen_a=glena)

    # assert that global glen_a is still unused
    assert np.isnan(cfg.PARAMS['glen_a'])

    # compare hist run to observations
    ds = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_histalp_%.3e' % glena))
    histalp_dl = ds.length_m.values[-1] - ds.length_m.values[0]
    delta = (histalp_dl - obs_dl)**2
    print('glenA: %.2e  delta: %.4f' % (glena, delta))
    if optimization is True:
        return delta
    else:
        return tbias
    """
    else:
        dlhist = ds.length_m.to_dataframe()['length_m']
        # set first year to 0
        dlhist -= dlhist.iloc[0]
        # add actual difference between spinup and obs
        spin_offset = (tmp_mod.length_m-len2003) - dl
        # spin_offset is positive if spinoff is to large
        dlhist += spin_offset

        return dlhist
    """


def simple_spinup(gdir, meta):

    # --------- HOW SHALL WE SPIN ---------------

    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003'].iloc[0]

    tbias = spinup_with_tbias(gdir, fls, dl, len2003)
    return tbias


def given_spinup_tbias(gdir, meta=None, data=None):

    obs = data.loc[meta.index].iloc[0].copy()
    obs_ye = obs.dropna().index[-1]
    obs_dl = obs.dropna().iloc[-1]

    # --------- HOW SHALL WE SPIN ---------------
    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003'].iloc[0]
    # mass balance model
    mb = MultipleFlowlineMassBalance(gdir, fls=fls,
                                     mb_model_class=ConstantMassBalance)

    # for HEF, do something else afterwards
    tbias = -0.47
    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, None, gdir, False)

    tmp_mod = FileModel(gdir.get_filepath('model_run',
                                          filesuffix='_spinup'))
    tmp_mod.run_until(tmp_mod.last_yr)

    # --------- HIST IT DOWN ---------------
    tasks.run_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                                init_model_fls=tmp_mod.fls,
                                output_filesuffix='_histalp')

    # compare hist run to observations
    ds = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_histalp'))
    histalp_dl = ds.length_m.values[-1] - ds.length_m.values[0]
    delta = (histalp_dl - obs_dl)**2
    print('delta: %.4f' % delta)
    return tbias


def simple_spinup_plus_histalp(gdir, meta=None, obs=None):

    # select meta and obs
    meta = meta.loc[meta['RGI_ID'] == gdir.rgi_id].copy()
    obs = obs.loc[meta.index].iloc[0].copy()
    obs_ye = obs.dropna().index[-1]

    try:
        # --------- SPIN IT UP ---------------
        tbias = simple_spinup(gdir, meta)

        # --------- GET SPINUP STATE ---------------
        tmp_mod = FileModel(gdir.get_filepath('model_run',
                                              filesuffix='_spinup'))
        tmp_mod.run_until(tmp_mod.last_yr)

        # --------- HIST IT DOWN ---------------
        tasks.run_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                                    init_model_fls=tmp_mod.fls,
                                    output_filesuffix='_histalp')

        ds1 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                                filesuffix='_histalp'))
        ds2 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                                filesuffix='_spinup'))
        rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'].iloc[0],
                'histalp': ds1.length_m.to_dataframe()['length_m'],
                'spinup': ds2.length_m.to_dataframe()['length_m'],
                'tbias': tbias}
    except (FloatingPointError, RuntimeError):

        rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'].iloc[0],
                'histalp': np.nan,
                'spinup': np.nan,
                'tbias': np.nan}
        pass

    return rval


def vary_precipitation_sf(gdirs, meta, obs, pcpsf=None):

    cfg.PARAMS['run_mb_calibration'] = True

    if pcpsf is None:
        pcpsf = np.arange(0.25, 5.25, 0.25)

    rval_dict = {}

    for sf in pcpsf:

        cfg.PARAMS['prcp_scaling_factor'] = sf

        # finish OGGM tasks
        compute_ref_t_stars(gdirs)
        task_list = [tasks.local_t_star,
                     tasks.mu_star_calibration,
                     tasks.prepare_for_inversion,
                     tasks.mass_conservation_inversion,
                     tasks.filter_inversion_output,
                     tasks.init_present_time_glacier
                     ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # actual spinup and histalp
        rval_dict[sf] = execute_entity_task(simple_spinup_plus_histalp,
                                            gdirs,
                                            meta=meta, obs=obs)
    return rval_dict




