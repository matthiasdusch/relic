import os
import pickle

from relic.preprocessing import GLCDICT, merge_pair_dict
from relic.length_observations import get_length_observations
from relic.postprocessing import glacier_to_table1, glacier_to_table2, runs2df
from relic import graphics

myglcs = list(GLCDICT.keys())
allmeta, allobs = get_length_observations(myglcs)

base = '/home/matthias/length_change_1850/finito'

histalp_storage = os.path.join(base, 'storage')
comit_storage = os.path.join(histalp_storage, 'commitmentruns')
comit_storage_noseed = os.path.join(histalp_storage, 'commitmentruns_noseed')
proj_storage = os.path.join(histalp_storage, 'projections')
tbiasdictpath = os.path.join(base, 'tbiasdict.p')
glcdictpath = os.path.join(base, 'glcdict.p')
pin = os.path.join(base, 'clusterdata')
pout = os.path.join(base, 'plots')


# do histalp and calibration plots. takes more time
if 0:
    try:
        glcdict = pickle.load(open(glcdictpath, 'rb'))
    except:
        runs = []
        for fname in os.listdir(pin):
            fl = pickle.load(open(os.path.join(pin, fname), 'rb'))

            for grad in list(fl.keys()):
                lbl = str(grad).strip('{}')
                runs.append({'%s' % lbl: fl[grad]})

        glcdict, tbdict, tribdict = runs2df(runs)
        pickle.dump(glcdict, open(glcdictpath, 'wb'))
        pickle.dump(tbdict, open(tbiasdictpath, 'wb'))

    graphics.past_simulation_and_params(glcdict, pout, y_len=3)

# do plots
if 0:
    glcdict = pickle.load(open(glcdictpath, 'rb'))

    for rgi in myglcs:
        meta = allmeta.loc[rgi]
        if merge_pair_dict(rgi) is not None:
            rgi += '_merged'

        graphics.past_simulation_and_commitment(rgi, allobs, allmeta,
                                                histalp_storage, comit_storage,
                                                comit_storage_noseed,
                                                pout, y_len=3)
        graphics.past_simulation_and_projection(rgi, allobs, allmeta,
                                                histalp_storage, proj_storage,
                                                comit_storage_noseed, pout,
                                                y_len=3)
        graphics.elevation_profiles(rgi, meta, histalp_storage, pout)
        graphics.paramplots(glcdict[rgi], rgi, pout, y_len=10)

# do tables
if 0:
    glacier_to_table1(base)
    glacier_to_table2(histalp_storage, comit_storage_noseed, tbiasdictpath,
                      base)

if 0:
    glcdict = pickle.load(open(glcdictpath, 'rb'))
    graphics.grey_madness({'RGI60-11.01270': glcdict['RGI60-11.01270']},
                           pout, y_len=3)
if 0:
    graphics.run_and_plot_merged_montmine(pout)

if 0:
    # do histogram here
    graphics.histogram(histalp_storage, base)
