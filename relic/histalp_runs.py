import numpy as np
import pandas as pd
import xarray as xr
import logging

from oggm import tasks, cfg
from oggm.core.flowline import FileModel, robust_model_run
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance)
from oggm.workflow import execute_entity_task
from oggm.core.climate import compute_ref_t_stars
from oggm import entity_task
from oggm.exceptions import InvalidParamsError

from relic.spinup import spinup_with_tbias, minimize_dl, systematic_spinup
from relic.postprocessing import relative_length_change, mae, r2

# Module logger
log = logging.getLogger(__name__)


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


@entity_task(log)
def relic_from_climate_data(gdir, ys=None, ye=None, min_ys=None,
                            store_monthly_step=False,
                            climate_filename='climate_monthly',
                            climate_input_filesuffix='', output_filesuffix='',
                            init_model_filesuffix=None, init_model_yr=None,
                            init_model_fls=None, zero_initial_glacier=False,
                            mass_balance_bias=None,
                            **kwargs):
    """ copy of flowline.run_from_climate_data
    """

    if ys is None:
        try:
            ys = gdir.rgi_date.year
        except AttributeError:
            ys = gdir.rgi_date
    if ye is None:
        raise InvalidParamsError('Need to set the `ye` kwarg!')
    if min_ys is not None:
        ys = ys if ys < min_ys else min_ys

    if init_model_filesuffix is not None:
        fp = gdir.get_filepath('model_run', filesuffix=init_model_filesuffix)
        with FileModel(fp) as fmod:
            if init_model_yr is None:
                init_model_yr = fmod.last_yr
            fmod.run_until(init_model_yr)
            init_model_fls = fmod.fls

    if (mass_balance_bias is not None) and ('merged' in gdir.rgi_id):
        raise ValueError('Need to think about this...')

    mb = MultipleFlowlineMassBalance(gdir, mb_model_class=PastMassBalance,
                                     filename=climate_filename,
                                     input_filesuffix=climate_input_filesuffix,
                                     bias=mass_balance_bias)

    return robust_model_run(gdir, output_filesuffix=output_filesuffix,
                            mb_model=mb, ys=ys, ye=ye,
                            store_monthly_step=store_monthly_step,
                            init_model_fls=init_model_fls,
                            zero_initial_glacier=zero_initial_glacier,
                            **kwargs)


def simple_spinup_plus_histalp(gdir, meta=None, obs=None, mb_bias=None,
                               use_systematic_spinup=False):

    # select meta and obs
    meta = meta.loc[meta['RGI_ID'] == gdir.rgi_id].copy()
    obs = obs.loc[meta.index].iloc[0].copy()
    obs_ye = obs.dropna().index[-1]

    # --------- SPIN IT UP ---------------
    if use_systematic_spinup:
        tbias = systematic_spinup(gdir, meta)

        if tbias == -999:

            rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'].iloc[0],
                    'histalp': np.nan,
                    'spinup': np.nan,
                    'tbias': np.nan, 'tmean': np.nan, 'pmean': np.nan}
            return rval
    else:
        tbias = simple_spinup(gdir, meta)

    # --------- GET SPINUP STATE ---------------
    tmp_mod = FileModel(gdir.get_filepath('model_run',
                                          filesuffix='_spinup'))
    tmp_mod.run_until(tmp_mod.last_yr)

    # --------- HIST IT DOWN ---------------
    relic_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                            init_model_fls=tmp_mod.fls,
                            output_filesuffix='_histalp',
                            mass_balance_bias=mb_bias)

    ds1 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_histalp'))
    ds2 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_spinup'))
    # store mean temperature and precipitation
    yindex = np.arange(meta['first'].iloc[0], obs_ye+1)
    cm = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    tmean = cm.temp.groupby('time.year').mean().loc[yindex].to_pandas()
    pmean = cm.prcp.groupby('time.year').mean().loc[yindex].to_pandas()

    rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'].iloc[0],
            'histalp': ds1.length_m.to_dataframe()['length_m'],
            'spinup': ds2.length_m.to_dataframe()['length_m'],
            'tbias': tbias, 'tmean': tmean, 'pmean': pmean}

    # relative length change
    rval['rel_dl'] = relative_length_change(meta, rval['spinup'],
                                            rval['histalp'])

    rval['mae'] = mae(obs, rval['rel_dl'])
    rval['r2'] = r2(obs, rval['histalp'])
    """
    except (FloatingPointError, RuntimeError) as err:

        rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'].iloc[0],
                'histalp': np.nan,
                'spinup': np.nan,
                'tbias': np.nan, 'tmean': np.nan, 'pmean': np.nan}
        pass
    """

    return rval


def vary_mass_balance_bias(gdirs, meta, obs, mbbias=None):

    if mbbias is None:
        print('use optimization to find a mbbias')
    else:

        rval_dict = {}
        for mb in mbbias:
            # actual spinup and histalp
            rval_dict[mb] = execute_entity_task(simple_spinup_plus_histalp,
                                                gdirs, meta=meta, obs=obs,
                                                mb_bias=mb)

    return rval_dict


def vary_precipitation_sf(gdirs, meta, obs, pcpsf=None,
                          use_systematic_spinup=False):

    cfg.PARAMS['run_mb_calibration'] = True

    if pcpsf is None:
        pcpsf = np.arange(0.5, 3.25, 0.25)

    rval_dict = {}

    for sf in pcpsf:

        cfg.PARAMS['prcp_scaling_factor'] = sf
        log.info('Precipitation sf = %.1e' % sf)

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
                                            gdirs, meta=meta, obs=obs,
                                            use_systematic_spinup=use_systematic_spinup
                                            )
    return rval_dict


def vary_precipitation_gradient(gdirs, meta, obs, prcp_gradient=None,
                                use_systematic_spinup=False):

    if prcp_gradient is None:
        # vary gradient between 0% and 100% per 1000m
        prcp_gradient = np.nan  # np.arange(0, 1.1, 0.1)*1e-3

    rval_dict = {}
    for grad in prcp_gradient:
        # actual spinup and histalp
        cfg.PARAMS['prcp_gradient'] = grad
        log.info('Precipitation gradient = %.1e' % grad)
        rval_dict[grad] = execute_entity_task(simple_spinup_plus_histalp,
                                              gdirs, meta=meta, obs=obs,
                                              use_systematic_spinup=use_systematic_spinup
                                              )

    return rval_dict
