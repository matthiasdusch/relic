import os

import numpy as np
import shutil
import logging

from oggm import cfg, GlacierDirectory
from oggm.core.flowline import FileModel, robust_model_run
from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   RandomMassBalance)

log = logging.getLogger(__name__)


def run_and_store_from_disk(rgi, histalp_storage, commit_storage, y0=1999,
                            years=300, seed=None, unique_samples=False,
                            halfsize=15):

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

        mb = MultipleFlowlineMassBalance(gdir,
                                         mb_model_class=RandomMassBalance,
                                         filename='climate_monthly',
                                         bias=mbbias, y0=y0, seed=seed,
                                         unique_samples=unique_samples,
                                         halfsize=halfsize)

        robust_model_run(gdir,
                         output_filesuffix='commitment{:04d}_{:02d}'.format(
                             y0, i),
                         mb_model=mb, ys=0, ye=years,
                         init_model_fls=tmp_mod.fls)

        fn1 = 'model_diagnostics_commitment{:04d}_{:02d}.nc'.format(y0, i)
        shutil.copyfile(
            gdir.get_filepath('model_diagnostics',
                              filesuffix='commitment{:04d}_{:02d}'.format(
                                  y0, i)),
            os.path.join(commit_storage, fn1))

        fn4 = 'model_run_commitment{:04d}_{:02d}.nc'.format(y0, i)
        shutil.copyfile(
            gdir.get_filepath('model_run',
                              filesuffix='commitment{:04d}_{:02d}'.format(
                                  y0, i)),
            os.path.join(commit_storage, fn4))
