import os
import pickle
import numpy as np

from relic.preprocessing import configure, GLCDICT, merge_pair_dict
from relic.length_observations import get_length_observations
from relic.histalp_runs import run_ensemble


import logging
log = logging.getLogger(__name__)


if __name__ == '__main__':

    # Local working directory (where OGGM will write its output)
    WORKING_DIR = os.environ.get('OGGM_WORKDIR')

    myglcs = list(GLCDICT.keys())
    myglcs.sort()

    while True:
        try:
            allgdirs = configure(WORKING_DIR, myglcs, baselineclimate='HISTALP')
            break
        except:
            log.warning('fiona error')
            pass

    # read length data observations
    meta, obs = get_length_observations(myglcs)

    stor = '/home/users/mdusch/storage/'


    alltbiasdict = pickle.load(open(os.path.join(stor, 'tbiasdict.p'), 'rb'))


    jobid = int(os.environ.get('JOBID')) - 1


    rgi_id = myglcs[jobid]

    if merge_pair_dict(rgi_id) is not None:
        rgi_id += '_merged'

    print('------------------------------- current glacier: {}'.format(rgi_id))

    rundictpath = os.path.join(stor, 'runs_%s.p' % rgi_id)
    ensemble = pickle.load(open(rundictpath, 'rb'))['ensemble']
    tbiasdict = {}
    for run in ensemble:
        tbiasdict[run] = alltbiasdict[rgi_id].loc[run]

    run_ensemble(allgdirs, rgi_id, ensemble, tbiasdict, meta, stor, runsuffix='')

