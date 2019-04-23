
from oggm import utils, cfg, workflow, tasks

import logging
log = logging.getLogger(__name__)


def configure(workdir, glclist, glena_factor=1.5):
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
    cfg.PARAMS['continue_on_error'] = True

    cfg.PARAMS['use_intersects'] = False
    cfg.PARAMS['use_rgi_area'] = True

    # set negative flux filtering to false. should be standard soon
    cfg.PARAMS['filter_for_neg_flux'] = False
    cfg.PARAMS['correct_for_neg_flux'] = True

    # and set standard histalp values
    cfg.PARAMS['prcp_scaling_factor'] = 1.75
    cfg.PARAMS['temp_melt'] = -1.75

    # use histalp
    cfg.PARAMS['baseline_climate'] = 'HISTALP'

    gla = cfg.PARAMS['glen_a']
    # set glan a to nan just do make sure there are no global problems
    cfg.PARAMS['glen_a'] = gla*glena_factor

    gdirs = workflow.init_glacier_regions(glclist, from_prepro_level=3)
    if cfg.PARAMS['baseline_climate'] == 'HISTALP':
        workflow.execute_entity_task(tasks.process_histalp_data, gdirs)
        workflow.execute_entity_task(tasks.local_t_star, gdirs)
        workflow.execute_entity_task(tasks.mu_star_calibration, gdirs)
    workflow.execute_entity_task(tasks.init_present_time_glacier, gdirs)

    return gdirs
