import numpy as np
import pandas as pd
import xarray as xr
import logging
import itertools

from oggm import tasks, cfg
from oggm.core.flowline import FileModel, robust_model_run
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   PastMassBalance)
from oggm.workflow import execute_entity_task, merge_glacier_tasks
from oggm.core.climate import compute_ref_t_stars
from oggm import entity_task, GlacierDirectory
from oggm.exceptions import InvalidParamsError

from relic.spinup import systematic_spinup
from relic.preprocessing import merge_pair_dict
from relic.postprocessing import relative_length_change

from relic import preprocessing

# Module logger
log = logging.getLogger(__name__)


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
            flids = np.unique([fl.rgi_id for fl in fls])
            for fl in flids:
                flsfx = '_' + fl
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


def spinup_plus_histalp(gdir, meta=None, obs=None, mb_bias=None):
    # take care of merged glaciers
    rgi_id = gdir.rgi_id.split('_')[0]

    # select meta and obs
    meta = meta.loc[rgi_id].copy()
    # obs = obs.loc[rgi_id].copy()
    # we want to simulate as much as possible -> histalp till 2014
    # obs_ye = obs.dropna().index[-1]
    obs_ye = 2014

    # --------- SPIN IT UP ---------------
    tbias = systematic_spinup(gdir, meta)

    if tbias == -999:

        rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'],
                'histalp': np.nan,
                'spinup': np.nan,
                'tbias': np.nan, 'tmean': np.nan, 'pmean': np.nan}
        return rval

    # --------- GET SPINUP STATE ---------------
    tmp_mod = FileModel(gdir.get_filepath('model_run',
                                          filesuffix='_spinup'))
    tmp_mod.run_until(tmp_mod.last_yr)

    # --------- HIST IT DOWN ---------------
    try:
        relic_from_climate_data(gdir, ys=meta['first'], ye=obs_ye,
                                init_model_fls=tmp_mod.fls,
                                output_filesuffix='_histalp',
                                mass_balance_bias=mb_bias)
    except RuntimeError as err:
        if 'Glacier exceeds domain boundaries' in err.args[0]:
            log.info('(%s) histalp run exceeded domain bounds' % gdir.rgi_id)
            return
        else:
            raise RuntimeError('other error')

    ds1 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_histalp'))
    ds2 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_spinup'))
    # store mean temperature and precipitation
    yindex = np.arange(meta['first'], obs_ye+1)

    try:
        cm = xr.open_dataset(gdir.get_filepath('climate_monthly'))
    except FileNotFoundError:
        cm = xr.open_dataset(gdir.get_filepath('climate_monthly',
                                               filesuffix='_' + rgi_id))

    tmean = cm.temp.groupby('time.year').mean().loc[yindex].to_pandas()
    pmean = cm.prcp.groupby('time.year').mean().loc[yindex].to_pandas()

    rval = {'rgi_id': gdir.rgi_id, 'name': meta['name'],
            'histalp': ds1.length_m.to_dataframe()['length_m'],
            'spinup': ds2.length_m.to_dataframe()['length_m'],
            'tbias': tbias, 'tmean': tmean, 'pmean': pmean}

    # relative length change
    rval['rel_dl'] = relative_length_change(meta, rval['spinup'],
                                            rval['histalp'])

    # TODO: EXTRACT and ADD thickness information here

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

        # assert trib.iloc[0] == trib.max(), ('the tributary was not connected'
        #                                    'to the main glacier at the start'
        #                                    'of this histalp run')

        trib -= trib.iloc[0]
        rval['trib_dl'] = trib

    return rval


def multi_parameter_run(paramdict, gdirs, meta, obs, runid=None):
    # get us all parameters
    keys = paramdict.keys()
    values = paramdict.values()
    paramcombi = [dict(zip(keys, combination)) for
                  combination in itertools.product(*values)]
    log.info('Multi parameter run with >>> %s <<< parameters started.' %
             len(paramcombi))

    # set some default parameters which might be changed by the paramdict
    # set mass balance bias to None, will be changed if passed as a parameter
    mbbias = None
    # default glena
    default_glena = 2.4e-24
    # default sliding
    default_fs = 5.7e-20

    # if a runid is passed, run only this item in the paramcombi
    # runids (= SLURM JOBID) start at 1 !
    if runid is not None:
        paramcombi = [paramcombi[runid-1]]

    # rval_dict is our output
    rval_dict = {}
    # TODO think of something nicer! NetCDF or a like

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

        if runid is not None:
            nr = runid-1

        log.info('Current parameter combination: %s' % str(combi))
        log.info('This is combination %d out of %d.' % (nr+1, len(paramcombi)))

        # ok, we need the ref_glaciers here for calibration
        # they should be initialiced so, just recreate them from the directory
        ref_gdirs = [GlacierDirectory(refid) for
                     refid in preprocessing.ADDITIONAL_REFERENCE_GLACIERS]

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
        gdirs_merged = []
        gdirs2sim = gdirs.copy()
        for gid in meta.index:
            merg = merge_pair_dict(gid)
            if merg is not None:
                # main and tributary glacier
                gd2merge = [gd for gd in gdirs if gd.rgi_id in [gid] + merg[0]]

                # actual merge task
                gdir_merged = merge_glacier_tasks(gd2merge, gid,
                                                  buffer=merg[1])

                # remove the entity glaciers from the simulation list
                gdirs2sim = [gd for gd in gdirs2sim if
                             gd.rgi_id not in [gid] + merg[0]]

                """
                # uncomment to visually inspect the merged glacier
                import matplotlib.pyplot as plt
                from oggm import graphics
                import os
                f, ax = plt.subplots(1, 1, figsize=(12, 12))
                graphics.plot_centerlines(gdir_merged,
                                          use_model_flowlines=True, ax=ax)
                f.savefig(os.path.join(cfg.PATHS['working_dir'], gid) + '.png')
                """

                gdirs_merged.append(gdir_merged)

        # add merged glaciers to the left over entity glaciers
        gdirs2sim += gdirs_merged

        # do the actual simulations
        rval = execute_entity_task(spinup_plus_histalp,
                                   gdirs2sim, meta=meta, obs=obs,
                                   mb_bias=mbbias
                                   )
        # remove possible Nones
        rval = [rl for rl in rval if rl is not None]

        rval_dict[str(combi)] = rval

    return rval_dict
