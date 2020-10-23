import numpy as np
import pandas as pd

from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance)
from oggm.core.flowline import FluxBasedModel

import logging
log = logging.getLogger(__name__)


def minimize_dl(tbias, mb, fls, dl, len2003, gdir, optimization, runsuffix=''):
    # Mass balance
    mb.temp_bias = tbias

    model = FluxBasedModel(fls, mb_model=mb)

    try:
        model.run_until_equilibrium(ystep=10, max_ite=200)
    except ValueError:
        log.info('(%s) tbias of %.2f did exceed max iterations (def 1000yrs)' %
                 (gdir.rgi_id, tbias))
        return len2003**2
    except FloatingPointError:
        if optimization is True:
            log.info('(%s) tbias of %.2f gave length: %.2f' %
                     (gdir.rgi_id, tbias, model.length_m))
            return len2003**2
        else:
            raise RuntimeError('This should never happen...')
    except RuntimeError as err:
        if (optimization is True) and\
           ('Glacier exceeds domain boundaries' in err.args[0]):
            log.info('(%s) tbias of %.2f exceeds domain boundaries' %
                     (gdir.rgi_id, tbias))
            return len2003**2
        elif 'CFL error' in err.args[0]:
            log.info('(%s) tbias of %.2f leads to CFL error' %
                     (gdir.rgi_id, tbias))
            print(err)
            return len2003**2
        else:
            print(err)
            raise RuntimeError('This should never happen 2...')

    if optimization is True:
        if model.length_m < fls[-1].dx_meter:
            log.info('(%s) tbias of %.2f gave length: %.2f' %
                     (gdir.rgi_id, tbias, model.length_m))
            return len2003**2

        dl_spinup = model.length_m - len2003
        delta = (dl - dl_spinup)**2
        print('%s: tbias: %.2f  delta: %.4f' % (gdir.rgi_id, tbias, delta))
        return delta
    else:

        filesuffix = '_spinup' + runsuffix

        run_path = gdir.get_filepath('model_run',
                                     filesuffix=filesuffix,
                                     delete=True)
        diag_path = gdir.get_filepath('model_diagnostics',
                                      filesuffix=filesuffix,
                                      delete=True)
        model2 = FluxBasedModel(fls, mb_model=mb)
        model2.run_until_and_store(model.yr, run_path=run_path,
                                   diag_path=diag_path)


def systematic_spinup(gdir, meta, mb_bias=None, y0=1999):

    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    # TODO maybe not use 2003 as fixed date, but rather ask for the RGI date
    #   this then needs to be considered in meta as well
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003']
    # mass balance model
    log.warning('DeprecationWarning: If downloadlink is updated to ' +
                'gdirs_v1.2 remove filename kwarg')
    mb = MultipleFlowlineMassBalance(gdir, fls=fls,
                                     mb_model_class=ConstantMassBalance,
                                     filename='climate_monthly',
                                     y0=y0,
                                     bias=mb_bias)

    # coarse first test values
    totest = np.arange(-8, 3.1)

    # dataframe for results
    rval = pd.DataFrame([], columns=['delta'], dtype=float)

    # linespace counter
    lsc = 0
    # linespace shift
    lss = [0.5, 0.25, 0.125, 0.06, 0.02]

    while True:
        # dont do anything twice
        totest = totest[~np.isin(totest, rval.index)]
        for tb in totest:
            delta = minimize_dl(tb, mb, fls, dl, len2003, gdir, True)
            if delta == len2003**2:
                delta = np.nan

            rval.loc[tb, 'delta'] = delta
            if np.sqrt(delta) < fls[-1].dx_meter:
                break

        if np.sqrt(delta) < fls[-1].dx_meter:
            break

        if lsc == len(lss):
            log.info('SPINUP WARNING (%s): use best result so far!' %
                     gdir.rgi_id)
            break

        if rval['delta'].isna().all():
            log.info('SPINUP ERROR (%s): could not find working tbias!' %
                     gdir.rgi_id)
            return -999

        # no fit so far, get new tbias to test:
        # current minima
        cmin = rval['delta'].idxmin()

        totest = np.linspace(cmin-lss[lsc], cmin+lss[lsc], 5).round(2)
        lsc += 1

    tbias = rval.dropna().idxmin().iloc[0]
    delta = np.sqrt(rval.loc[tbias, 'delta'])

    log.info('(%s) delta = %.2f (flowline spacing = %.2f)' %
             (gdir.rgi_id, delta, fls[-1].dx_meter))

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, None, None, gdir, False)
    return tbias
