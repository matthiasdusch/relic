import os

from relic.preprocessing import GLCDICT, merge_pair_dict
from relic.commitment_runs import run_and_store_from_disk as com_run
from relic.projection_runs import run_and_store_from_disk as pro_run


from oggm import cfg, utils


if __name__ == '__main__':

    # Local working directory (where OGGM will write its output)
    WORKING_DIR = os.environ.get('OGGM_WORKDIR')

    myglcs = list(GLCDICT.keys())
    myglcs.sort()

    histalp_storage = '/home/users/mdusch/storage/'


    jobid = int(os.environ.get('JOBID')) - 1
    rgi_id = myglcs[jobid]

    if merge_pair_dict(rgi_id) is not None:
        rgi_id += '_merged'


    # Initialize OGGM
    cfg.initialize()
    cfg.PATHS['working_dir'] = WORKING_DIR
    cfg.PARAMS['baseline_climate'] = 'HISTALP'
    # and set standard histalp values
    cfg.PARAMS['temp_melt'] = -1.75

    #
    commit_storage1 = os.path.join(histalp_storage, 'commitmentruns_noseed',
                                   rgi_id)
    utils.mkdir(commit_storage1)

    com_run(rgi_id, histalp_storage, commit_storage1, y0=1999,
            years=300, seed=23)
    com_run(rgi_id, histalp_storage, commit_storage1, y0=1885,
            years=300, seed=23)
    com_run(rgi_id, histalp_storage, commit_storage1, y0=1970,
            years=300, seed=23, halfsize=10)

    #
    commit_storage2 = os.path.join(histalp_storage, 'commitmentruns_noseed',
                                   rgi_id)
    utils.mkdir(commit_storage2)

    com_run(rgi_id, histalp_storage, commit_storage2, y0=1999,
            years=300, seed=None)
    com_run(rgi_id, histalp_storage, commit_storage2, y0=1885,
            years=300, seed=None)
    com_run(rgi_id, histalp_storage, commit_storage2, y0=1970,
            years=300, seed=None, halfsize=10)

    #
    proj_storage = os.path.join(histalp_storage, 'projections', rgi_id)
    utils.mkdir(proj_storage)

    pro_run(rgi_id, histalp_storage, proj_storage)
