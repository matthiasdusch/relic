import os
import pickle
import numpy as np

from relic.histalp_runs import multi_parameter_run
from relic.preprocessing import configure, GLCDICT
from relic.length_observations import get_length_observations

import logging
log = logging.getLogger(__name__)


if __name__ == '__main__':

    # Local working directory (where OGGM will write its output)
    WORKING_DIR = os.environ.get('OGGM_WORKDIR')

    myglcs = list(GLCDICT.keys())

    while True:
        try:
            gdirs = configure(WORKING_DIR, myglcs, baselineclimate='HISTALP')
            break
        except:
            log.warning('fiona error')
            pass

    # read length data observations
    meta, obs = get_length_observations(myglcs)

    pcpsf = np.arange(0.5, 4.1, 0.25)
    glenas = np.arange(1.0, 4.1, 0.5)
    mbbias = np.append(np.arange(-1400, 1100, 200), np.array([-100, 100]))

    pdict = {'prcp_scaling_factor': pcpsf,
             'glena_factor': glenas,
             'mbbias': mbbias}

    jobid = int(os.environ.get('JOBID'))
    rval = multi_parameter_run(pdict, gdirs, meta, obs, runid=jobid)

    out = os.path.join('/home/users/mdusch/length_change/finito/all/out', 'out_%d.p' % jobid)
    pickle.dump(rval, open(out, 'wb'))
    log.warning('finito')
