import numpy as np
import xarray as xr
import logging
import itertools
import ast
import os
import shutil

from oggm import tasks, cfg
from oggm.core.flowline import (FileModel, robust_model_run,
                                run_from_climate_data)
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   PastMassBalance, ConstantMassBalance)
from oggm.workflow import execute_entity_task, merge_glacier_tasks
from oggm.core.climate import compute_ref_t_stars
from oggm import entity_task, GlacierDirectory
from oggm.exceptions import InvalidParamsError
from oggm.utils import copy_to_basedir, mkdir, include_patterns

from relic.spinup import systematic_spinup, final_spinup
from relic.preprocessing import merge_pair_dict
from relic.postprocessing import relative_length_change

from relic import preprocessing

# Module logger
log = logging.getLogger(__name__)


def spinup_plus_histalp(gdir, meta=None, mb_bias=None, runsuffix=''):
    # take care of merged glaciers
    rgi_id = gdir.rgi_id.split('_')[0]

    # select meta
    meta = meta.loc[rgi_id].copy()
    # we want to simulate as much as possible -> histalp till 2014
    obs_ye = 2014

    # --------- SPIN IT UP ---------------
    tbias = systematic_spinup(gdir, meta, mb_bias=mb_bias)

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

    # --------- HIST IT DOWN --------------
    try:
        run_from_climate_data(gdir, ys=meta['first'], ye=obs_ye,
                              init_model_fls=tmp_mod.fls,
                              climate_filename='climate_monthly',
                              output_filesuffix='_histalp' + runsuffix,
                              bias=mb_bias)
    except RuntimeError as err:
        if 'Glacier exceeds domain boundaries' in err.args[0]:
            log.info('(%s) histalp run exceeded domain bounds' % gdir.rgi_id)
            return
        else:
            raise RuntimeError('other error')

    ds1 = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                            filesuffix='_histalp' + runsuffix))
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

    # TODO: EXTRACT and ADD thickness information here if need be

    # if merged, store tributary flowline change as well
    if '_merged' in gdir.rgi_id:

        trib = rval['histalp'].copy() * np.nan

        # choose the correct flowline index, use model_fls as they have rgiids
        fls = gdir.read_pickle('model_flowlines')
        flix = np.where([fl.rgi_id != rgi_id for fl in fls])[0][-1]

        fmod = FileModel(gdir.get_filepath('model_run',
                                           filesuffix='_histalp' + runsuffix))
        assert fmod.fls[flix].nx == fls[flix].nx, ('filemodel and gdir '
                                                   'flowlines do not match')
        for yr in rval['histalp'].index:
            fmod.run_until(yr)
            trib.loc[yr] = fmod.fls[flix].length_m

        trib -= trib.iloc[0]
        rval['trib_dl'] = trib

    return rval


def multi_parameter_run(paramdict, gdirs, meta, obs, runid=None, runsuffix=''):
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
                log.warning('DeprecationWarning: If downloadlink is updated ' +
                            'to gdirs_v1.2, remove filename kwarg')
                gdir_merged = merge_glacier_tasks(gd2merge, gid,
                                                  buffer=merg[1],
                                                  filename='climate_monthly')

                # remove the entity glaciers from the simulation list
                gdirs2sim = [gd for gd in gdirs2sim if
                             gd.rgi_id not in [gid] + merg[0]]

                # uncomment to visually inspect the merged glacier
                """
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
                                   gdirs2sim, meta=meta,
                                   mb_bias=mbbias,
                                   runsuffix=runsuffix
                                   )
        # remove possible Nones
        rval = [rl for rl in rval if rl is not None]

        rval_dict[str(combi)] = rval

    return rval_dict


def run_ensemble(allgdirs, rgi_id, ensemble, tbiasdict, allmeta,
                 storedir, runsuffix='', spinup_y0=1999):

    # default glena
    default_glena = 2.4e-24

    # loop over all combinations
    for nr, run in enumerate(ensemble):

        pdict = ast.literal_eval('{' + run + '}')
        cfg.PARAMS['glen_a'] = pdict['glena_factor'] * default_glena
        cfg.PARAMS['inversion_glen_a'] = pdict['glena_factor'] * default_glena
        mbbias = pdict['mbbias']
        cfg.PARAMS['prcp_scaling_factor'] = pdict['prcp_scaling_factor']

        log.info('Current parameter combination: %s' % str(run))
        log.info('This is combination %d out of %d.' % (nr+1, len(ensemble)))

        # ok, we need the ref_glaciers here for calibration
        # they should be initialiced so, just recreate them from the directory
        ref_gdirs = [GlacierDirectory(refid) for
                     refid in preprocessing.ADDITIONAL_REFERENCE_GLACIERS]

        # do the mass balance calibration
        compute_ref_t_stars(ref_gdirs + allgdirs)
        task_list = [tasks.local_t_star,
                     tasks.mu_star_calibration,
                     tasks.prepare_for_inversion,
                     tasks.mass_conservation_inversion,
                     tasks.filter_inversion_output,
                     tasks.init_present_time_glacier
                     ]

        for task in task_list:
            execute_entity_task(task, allgdirs)

        # check for glaciers to merge:
        gdirs_merged = []
        gdirs2sim = allgdirs.copy()
        for gid in allmeta.index:
            merg = merge_pair_dict(gid)
            if merg is not None:
                # main and tributary glacier
                gd2merge = [gd for gd in allgdirs if gd.rgi_id in [gid] + merg[0]]

                # actual merge task
                log.warning('DeprecationWarning: If downloadlink is updated ' +
                            'to gdirs_v1.2, remove filename kwarg')
                gdir_merged = merge_glacier_tasks(gd2merge, gid,
                                                  buffer=merg[1],
                                                  filename='climate_monthly')

                # remove the entity glaciers from the simulation list
                gdirs2sim = [gd for gd in gdirs2sim if
                             gd.rgi_id not in [gid] + merg[0]]

                gdirs_merged.append(gdir_merged)

        # add merged glaciers to the left over entity glaciers
        gdirs2sim += gdirs_merged

        # now only select the 1 glacier
        gdir = [gd for gd in gdirs2sim if gd.rgi_id == rgi_id][0]
        rgi_id0 = rgi_id.split('_')[0]
        meta = allmeta.loc[rgi_id0].copy()

        # do the actual simulations

        # spinup
        fls = gdir.read_pickle('model_flowlines')
        delta = fls[-1].dx_meter
        len2003 = fls[-1].length_m
        dl = -meta['dL2003']

        try:
            final_spinup(tbiasdict[run], mbbias, spinup_y0,
                         fls, dl, len2003, delta, gdir,
                         filesuffix='spinup_{:02d}'.format(nr))
        except RuntimeError:
            log.warning('Delta > 1x fl dx ({:.2f}), using 2x'.format(delta))
            final_spinup(tbiasdict[run], mbbias, spinup_y0,
                         fls, dl, len2003,
                         delta*2,
                         gdir, filesuffix='spinup_{:02d}'.format(nr))

        # histalp
        # --------- GET SPINUP STATE ---------------
        tmp_mod = FileModel(
            gdir.get_filepath('model_run',
                              filesuffix='spinup_{:02d}'.format(nr)))

        tmp_mod.run_until(tmp_mod.last_yr)

        # --------- HIST IT DOWN ---------------
        histrunsuffix = 'histalp{}_{:02d}'.format(runsuffix, nr)

        # now actual simulation
        run_from_climate_data(gdir, ys=meta['first'], ye=2014,
                              init_model_fls=tmp_mod.fls,
                              output_filesuffix=histrunsuffix,
                              climate_filename='climate_monthly',
                              bias=mbbias)

        # save the calibration parameter to the climate info file
        out = gdir.get_climate_info()
        out['ensemble_calibration'] = pdict
        gdir.write_json(out, 'climate_info')

        # copy stuff to storage
        basedir = os.path.join(storedir, rgi_id)
        ensdir = os.path.join(basedir, '{:02d}'.format(nr))
        mkdir(ensdir, reset=True)

        deep_path = os.path.join(ensdir, rgi_id[:8], rgi_id[:11], rgi_id)

        # copy whole GDir
        copy_to_basedir(gdir, base_dir=ensdir, setup='run')

        # copy run results
        fn1 = 'model_diagnostics_spinup_{:02d}.nc'.format(nr)
        shutil.copyfile(
            gdir.get_filepath('model_diagnostics',
                              filesuffix='spinup_{:02d}'.format(nr)),
            os.path.join(deep_path, fn1))

        fn2 = 'model_diagnostics_{}.nc'.format(histrunsuffix)
        shutil.copyfile(
            gdir.get_filepath('model_diagnostics', filesuffix=histrunsuffix),
            os.path.join(deep_path, fn2))

        fn3 = 'model_run_spinup_{:02d}.nc'.format(nr)
        shutil.copyfile(
            gdir.get_filepath('model_run',
                              filesuffix='spinup_{:02d}'.format(nr)),
            os.path.join(deep_path, fn3))

        fn4 = 'model_run_{}.nc'.format(histrunsuffix)
        shutil.copyfile(
            gdir.get_filepath('model_run', filesuffix=histrunsuffix),
            os.path.join(deep_path, fn4))

        log.warning('DeprecationWarning: If downloadlink is updated to ' +
                    'gdirs_v1.2 remove this copyfile:')
        # copy (old) climate monthly files which
        for fn in os.listdir(gdir.dir):
            if 'climate_monthly' in fn:
                shutil.copyfile(os.path.join(gdir.dir, fn),
                                os.path.join(deep_path, fn))
