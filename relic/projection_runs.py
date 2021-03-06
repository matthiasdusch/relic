import os

import numpy as np
import shutil
import logging

from oggm import cfg, utils, GlacierDirectory, tasks
from oggm.core import gcm_climate
from oggm.workflow import init_glacier_regions, execute_entity_task
from oggm.core.flowline import FileModel, run_from_climate_data

from relic.preprocessing import merge_pair_dict

log = logging.getLogger(__name__)


def run_and_store_from_disk(rgi, histalp_storage, storage):

    cmip = ['CCSM4', 'CNRM-CM5', 'CSIRO-Mk3-6-0', 'CanESM2',
            'GFDL-CM3', 'GFDL-ESM2G', 'GISS-E2-R', 'IPSL-CM5A-LR',
            'MPI-ESM-LR', 'NorESM1-M']

    bp = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-ng/pr/pr_mon_{}_{}_r1i1p1_g025.nc'
    bt = 'https://cluster.klima.uni-bremen.de/~oggm/cmip5-ng/tas/tas_mon_{}_{}_r1i1p1_g025.nc'

    for i in np.arange(999):
        # Local working directory (where OGGM will write its output)
        storage_dir = os.path.join(histalp_storage, rgi, '{:02d}'.format(i),
                                   rgi[:8], rgi[:11], rgi)
        new_dir = os.path.join(cfg.PATHS['working_dir'], 'per_glacier',
                               rgi[:8], rgi[:11], rgi)

        # make sure directory is empty:
        try:
            shutil.rmtree(new_dir)
        except FileNotFoundError:
            pass
        # if path does not exist, we handled all ensemble members:
        try:
            shutil.copytree(storage_dir, new_dir)
        except FileNotFoundError:
            log.info('processed {:02d} ensemble members'.format(i))
            break

        gdir = GlacierDirectory(rgi)

        pdict = gdir.get_climate_info()['ensemble_calibration']

        cfg.PARAMS['prcp_scaling_factor'] = pdict['prcp_scaling_factor']
        default_glena = 2.4e-24
        cfg.PARAMS['glen_a'] = pdict['glena_factor'] * default_glena
        cfg.PARAMS['inversion_glen_a'] = pdict['glena_factor'] * default_glena
        mbbias = pdict['mbbias']

        tmp_mod = FileModel(
            gdir.get_filepath('model_run',
                              filesuffix='_histalp_{:02d}'.format(i)))
        tmp_mod.run_until(tmp_mod.last_yr)

        for cm in cmip:
            for rcp in ['rcp26', 'rcp45', 'rcp60', 'rcp85']:

                ft = utils.file_downloader(bt.format(cm, rcp))
                fp = utils.file_downloader(bp.format(cm, rcp))
                if ft is None:
                    log.warning('no {} for model {}'.format(rcp, cm))
                    continue

                filesuffix = '_{}_{}'.format(cm, rcp)

                # bias correct them
                if '_merged' in rgi:
                    process_cmip_for_merged_glacier(gdir, filesuffix, ft, fp)
                else:
                    gcm_climate.process_cmip5_data(gdir,
                                                   filesuffix=filesuffix,
                                                   fpath_temp=ft,
                                                   fpath_precip=fp)

                rid = '_{}_{}'.format(cm, rcp)
                rid_out = '{}_{:02d}'.format(rid, i)

                run_from_climate_data(gdir,
                                      ys=2014, ye=2100,
                                      climate_filename='gcm_data',
                                      climate_input_filesuffix=rid,
                                      init_model_fls=tmp_mod.fls,
                                      output_filesuffix=rid_out,
                                      bias=mbbias
                                      )

                fn1 = 'model_diagnostics{}.nc'.format(rid_out)
                shutil.copyfile(
                    gdir.get_filepath('model_diagnostics',
                                      filesuffix=rid_out),
                    os.path.join(storage, fn1))

                fn4 = 'model_run{}.nc'.format(rid_out)
                shutil.copyfile(
                    gdir.get_filepath('model_run',
                                      filesuffix=rid_out),
                    os.path.join(storage, fn4))


def process_cmip_for_merged_glacier(gdir, filesuffix, ft, fp):

    rgi = gdir.rgi_id.split('_')[0]

    rgis = merge_pair_dict(rgi)[0] + [rgi]

    gdirs = init_glacier_regions(rgis, prepro_border=10, from_prepro_level=1)
    execute_entity_task(tasks.process_histalp_data, gdirs)

    execute_entity_task(gcm_climate.process_cmip5_data, gdirs,
                        filesuffix=filesuffix, fpath_temp=ft, fpath_precip=fp)

    for gd in gdirs:
        # copy climate files
        shutil.copyfile(
            gd.get_filepath('gcm_data', filesuffix=filesuffix),
            gdir.get_filepath('gcm_data',
                              filesuffix='_{}{}'.format(gd.rgi_id, filesuffix)
                              ))
