import numpy as np
import pandas as pd
import xarray as xr
import logging
import itertools
import ast

from oggm import tasks, cfg
from oggm.core.flowline import FileModel, robust_model_run
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance,
                                   PastMassBalance)
from oggm.workflow import (execute_entity_task, init_glacier_regions,
                           merge_glacier_tasks)
from oggm.core.climate import compute_ref_t_stars
from oggm import entity_task
from oggm.exceptions import InvalidParamsError
from oggm.utils import get_ref_mb_glaciers_candidates

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

    if mass_balance_bias is not None:
        if '_merged' in gdir.rgi_id:
            fls = gdir.read_pickle('model_flowlines')
            for fl in fls:
                flsfx = '_' + fl.rgi_id
                df = gdir.read_json('local_mustar', filesuffix=flsfx)
                df['bias'] += mass_balance_bias
                gdir.write_json(df, 'local_mustar', filesuffix=flsfx)
            # we write this to the local_mustar file so we do not need to
            # pass it on to the MultipleFlowlineMassBalance model
            mass_balance_bias = None
        else:
            df = gdir.read_json('local_mustar')
            mass_balance_bias += df['bias']

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

    # take care of merged glaciers
    rgi_id = gdir.rgi_id.split('_')[0]

    # select meta and obs
    meta = meta.loc[meta['RGI_ID'] == rgi_id].copy()
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
    try:
        relic_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                                init_model_fls=tmp_mod.fls,
                                output_filesuffix='_histalp',
                                mass_balance_bias=mb_bias)
    except RuntimeError as err:
        if err.args[0] == 'Glacier exceeds domain boundaries.':
            log.info('(%s) histalp run exceeded domain bounds' % gdir.rgi_id)
            return

    ds1 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_histalp'))
    ds2 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_spinup'))
    # store mean temperature and precipitation
    yindex = np.arange(meta['first'].iloc[0], obs_ye+1)

    try:
        cm = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    except FileNotFoundError:
        cm = xr.open_dataset(gdir.get_filepath('climate_monthly',
                                               filesuffix='_' + rgi_id))

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

    # if merged, store tributary flowline change as well
    if '_merged' in gdir.rgi_id:

        trib = rval['histalp'].copy() * np.nan

        # choose the correct flowline index, use model_fls as they have rgiids
        fls = gdir.read_pickle('model_flowlines')
        flix = np.where([fl.rgi_id != rgi_id for fl in fls])[0][-1]

        fmod = FileModel(gdir.get_filepath('model_run', filesuffix='_histalp'))
        assert fmod.fls[flix].nx == fls[flix].nx, ('filemodel and gdir '
                                                   'flowlines do not match')
        for yr in rval['histalp'].index:
            fmod.run_until(yr)
            trib.loc[yr] = fmod.fls[flix].length_m

        #assert trib.iloc[0] == trib.max(), ('the tributary was not connected '
        #                                    'to the main glacier at the start '
        #                                    'of this histalp run')

        trib -= trib.iloc[0]
        rval['trib_dl'] = trib

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

    cfg.PARAMS['run_mb_calibration'] = True

    if prcp_gradient is None:
        # vary gradient between 0% and 100% per 1000m
        prcp_gradient = np.nan  # np.arange(0, 1.1, 0.1)*1e-3

    rval_dict = {}
    for grad in prcp_gradient:
        # actual spinup and histalp
        cfg.PARAMS['prcp_gradient'] = grad
        log.info('Precipitation gradient = %.1e' % grad)

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

        rval_dict[grad] = execute_entity_task(simple_spinup_plus_histalp,
                                              gdirs, meta=meta, obs=obs,
                                              use_systematic_spinup=use_systematic_spinup
                                              )

    return rval_dict


def multi_parameter_run(paramdict, gdirs, meta, obs, rgiregion=11,
                        use_systematic_spinup=False):

    # we want to run the mb calibration every time
    cfg.PARAMS['run_mb_calibration'] = True
    # and we want to use all glaciers for that (from one region)
    refids = get_ref_mb_glaciers_candidates()
    if rgiregion is not None:
        refids = [rid for rid in refids if '-%d.' % rgiregion in rid]
    # but do leave out the actual glaciers
    gids = [gd.rgi_id for gd in gdirs]
    refids = [rid for rid in refids if rid not in gids]

    # initialize the reference glaciers
    ref_gdirs = init_glacier_regions(rgidf=refids,
                                     from_prepro_level=3, prepro_border=10)

    # get us all parameters
    keys = paramdict.keys()
    values = paramdict.values()
    paramcombi = [dict(zip(keys, combination)) for
                  combination in itertools.product(*values)]
    log.info('Multi parameter run with >>> %s <<< parameters started.' %
             len(paramcombi))

    # set mass balance bias to None, will be changed if passed as a parameter
    mbbias = None

    # default glena
    default_glena = cfg.PARAMS['glen_a']

    # default sliding
    default_fs = 5.7e-20

    rval_dict = {}

    # loop over all combinations
    for nr, combi in enumerate(paramcombi):

        # set all parameters
        for key, val in combi.items():

            # here we se cfg.PARAMS values
            if key == 'glena_factor':
                cfg.PARAMS['glen_a'] = val * default_glena
                cfg.PARAMS['inversion_glen_a'] = val * default_glena
            # set mass balance bias
            elif key == 'mbbias':
                mbbias = val
            elif key == 'prcp_scaling_factor':
                cfg.PARAMS['prcp_scaling_factor'] = val
            elif key == 'sliding_factor':
                cfg.PARAMS['fs'] = val * default_fs
                cfg.PARAMS['inversion_fs'] = val * default_fs
            else:
                raise ValueError('Parameter not understood')

        log.info('Current parameter combination: %s' % str(combi))
        log.info('This is combination %d out of %d.' % (nr+1, len(paramcombi)))

        # do the mass balance calibration
        compute_ref_t_stars(ref_gdirs + gdirs)
        task_list = [tasks.local_t_star,
                     tasks.mu_star_calibration,
                     tasks.prepare_for_inversion,
                     tasks.mass_conservation_inversion,
                     tasks.filter_inversion_output,
                     tasks.init_present_time_glacier
                     ]
        for task in task_list:
            execute_entity_task(task, gdirs)

        # check for glaciers to merge:
        id_pairs = [['RGI60-11.02119', 'RGI60-11.02051', 8],   # ferpecle, mine
                    ['RGI60-11.02715', 'RGI60-11.02709', 2.5]]  # roseg,tschier
        gdirs_merged = []
        gdirs2sim = gdirs.copy()
        for ids in id_pairs:
            if (ids[0] in gids) and (ids[1] in gids):
                gd2merge = [gd for gd in gdirs2sim if gd.rgi_id in ids]
                gdirs2sim = [gd for gd in gdirs2sim if gd.rgi_id not in ids]
                gdir_merged = merge_glacier_tasks(gd2merge, ids[0],
                                                  buffer=ids[2])
                """
                # uncomment to visually inspect the merged glacier
                import matplotlib.pyplot as plt
                from oggm import graphics
                f, ax = plt.subplots(1, 1, figsize=(12, 12))
                graphics.plot_centerlines(gdir_merged,
                                          use_model_flowlines=True, ax=ax)
                plt.show()
                """
                gdirs_merged.append(gdir_merged)

        gdirs2sim += gdirs_merged

        # do the actual simulations
        rval = execute_entity_task(simple_spinup_plus_histalp,
                                   gdirs2sim, meta=meta, obs=obs,
                                   use_systematic_spinup=
                                   use_systematic_spinup,
                                   mb_bias=mbbias
                                   )
        # remove possible Nones
        rval = [rl for rl in rval if rl is not None]

        rval_dict[str(combi)] = rval

    return rval_dict
