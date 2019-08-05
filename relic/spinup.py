import numpy as np
import pandas as pd
import os
import pickle
import scipy
import xarray as xr

from oggm.core.massbalance import (MultipleFlowlineMassBalance,
                                   ConstantMassBalance)
from oggm.core.flowline import FileModel, FluxBasedModel
from oggm.utils import cfg

import logging
log = logging.getLogger(__name__)


def relic_run_until_equilibrium(model, rate=1e-4, ystep=10, max_ite=100):

    ite = 0
    was_close_zero = 0
    t_rate = 1
    while (t_rate > rate) and (ite <= max_ite) and (was_close_zero < 5):
        ite += 1
        v_bef = model.volume_m3
        for _y in np.arange(ystep):
            model.run_until(model.yr + 1)
        v_af = model.volume_m3
        if np.isclose(v_bef, 0., atol=1):
            t_rate = 1
            was_close_zero += 1
        else:
            t_rate = np.abs(v_af - v_bef) / v_bef
    if ite > max_ite:
        raise ValueError('Did not find equilibrium.')


def minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, optimization):
    # Mass balance
    mb.temp_bias = tbias

    model = FluxBasedModel(fls, mb_model=mb,
                           time_stepping='default',
                           glen_a=glena)

    try:
        relic_run_until_equilibrium(model)
    except ValueError:
        pass
    except FloatingPointError:
        if optimization is True:
            log.info('(%s) tbias of %.2f gave length: %.2f' %
                     (gdir.rgi_id, tbias, model.length_m))
            return len2003**2
        else:
            raise RuntimeError('This should never happen...')
    except RuntimeError as err:
        if (optimization is True) and\
           (err.args[0] == 'Glacier exceeds domain boundaries.'):
            log.info('(%s) tbias of %.2f exceeds domain boundaries' %
                     (gdir.rgi_id, tbias))
            return len2003**2
        else:
            raise RuntimeError('This should never happen...')

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

        if glena is None:
            filesuffix = '_spinup'
        else:
            filesuffix = '_spinup_%.3e' % glena

        run_path = gdir.get_filepath('model_run',
                                     filesuffix=filesuffix,
                                     delete=True)
        diag_path = gdir.get_filepath('model_diagnostics',
                                      filesuffix=filesuffix,
                                      delete=True)
        model2 = FluxBasedModel(fls, mb_model=mb,
                                time_stepping='default',
                                glen_a=glena)
        model2.run_until_and_store(model.yr, run_path=run_path,
                                   diag_path=diag_path)


def spinup_with_tbias(gdir, fls, dl, len2003, glena=None):
    """

    Parameters
    ----------
    gdir
    fls
    dl
    len2003
    glena

    Returns
    -------

    """

    # first tbias guess: Relative length change compared to todays length
    fg = np.abs(dl/len2003)
    # limit to reasonable first guesses
    fg = np.clip(fg, 0, 2)
    # force minus
    fg *= -1

    print('first guess: %.2f' % fg)

    # mass balance model
    mb = MultipleFlowlineMassBalance(gdir, fls=fls,
                                     mb_model_class=ConstantMassBalance)

    optilist = []
    optisuccess = False

    for nr in np.arange(1, 4):

        if nr == 1:
            br1 = fg
            br2 = fg-1
        else:
            optilist.append(opti)
            br1 = opti.x - 0.5
            br2 = opti.x - 1

        opti = scipy.optimize.minimize_scalar(minimize_dl,
                                              bracket=(br1, br2),
                                              tol=1e-2,
                                              args=(mb, fls, dl, len2003, glena,
                                                    gdir, True),
                                              options={'maxiter': 30}
                                              )

        log.info('%d. opti: %s\n%s' % (nr, gdir.rgi_id, opti))

        if np.sqrt(opti.fun) <= fls[-1].dx_meter:
            optisuccess = True
            break

    if optisuccess is False:
        for ol in optilist:
            if ol.fun < opti.fun:
                opti = ol
        log.info('%s optim did not work, continue with smallest error\n%s'
                 % (gdir.rgi_id, opti))

    tbias = opti.x

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)

    return tbias


def systematic_spinup(gdir, meta, glena=None):
    import numpy.polynomial.polynomial as poly

    # --------- HOW SHALL WE SPIN ---------------

    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003'].iloc[0]

    # first tbias guess: Relative length change compared to todays length
    fg = np.abs(dl/len2003)
    # limit to reasonable first guesses
    fg = np.clip(fg, 0, 2)
    # force minus
    fg *= -1

    print('first guess: %.2f' % fg)

    # mass balance model
    mb = MultipleFlowlineMassBalance(gdir, fls=fls,
                                     mb_model_class=ConstantMassBalance)

    # values to test systematicall
    totest = np.geomspace(fg, fg*3, 4)
    totest = np.unique(np.round(np.append(totest, fg-(totest-fg)), 2))
    totest = np.clip(totest, -3, 1)

    rval = pd.DataFrame([], columns=['delta'])

    counter = 0
    maxcount = 40

    found_fit = False

    while True:
        # dont do anything twice
        totest = totest[~np.isin(totest, rval.index)]

        for tb in totest:
            delta = minimize_dl(tb, mb, fls, dl, len2003, glena, gdir, True)
            counter += 1
            if delta == len2003**2:
                delta = np.nan
            rval.loc[tb, 'delta'] = delta
            if np.sqrt(delta) < fls[-1].dx_meter:
                found_fit = True
                break
        if found_fit is True:
            break

        rval = rval.astype(float)
        rval.sort_index(inplace=True)

        # no fit so far, get new tbias to test:
        # current minima
        cmin = rval['delta'].idxmin()

        if np.isnan(cmin):
            log.info('SPINUP ERROR: (%s) got all nans!' % gdir.rgi_id)
            fg -= 0.5
            totest = np.geomspace(fg, fg*3, 4)
            totest = np.unique(np.round(np.append(totest, fg-(totest-fg)), 2))
            if counter > maxcount:
                log.info('SPINUP ERROR: (%s) maximum counter reached!' %
                         gdir.rgi_id)
                break
            else:
                continue

        # if cmin left or right of values only test there
        if np.sum(rval.index > cmin) == 0:
            totest = np.round(np.linspace(cmin, cmin+0.5, 5)[1:], 2)
            continue
        if np.sum(rval.index < cmin) == 0:
            totest = np.round(np.linspace(cmin, cmin-0.5, 5)[1:], 2)
            continue

        # if our minimum is between two values, test there
        idx = np.where(cmin == rval.index)[0][0]
        totest = np.unique(np.round(np.linspace(rval.index[idx-1],
                                                rval.index[idx+1], 5),
                                    2))
        totest = totest[~np.isin(totest, rval.index)]

        # if we get nothing here, try a polyfit
        if len(totest) == 0:
            log.info('SPINUP (%s): USing polyfit!' %
                     gdir.rgi_id)
            y = rval.dropna().delta.values
            x = rval.dropna().index
            x_new = np.arange(x.min()-1, x.max()+1, 0.01)
            coef = poly.polyfit(x, y, 2)
            fit2d = poly.polyval(x_new, coef)

            if fit2d[0] < 0:
                log.info('SPINUP ERROR: (%s) negative fit should not happen!' %
                         gdir.rgi_id)
                fit2d = np.abs(fit2d)

            pmin = x_new[fit2d.argmin()]
            totest = np.geomspace(pmin, pmin*3, 5)
            totest = np.unique(np.round(np.append(totest, pmin-(totest-pmin)),
                                        2))
            counter += 1

        # limit to some values from experience
        totest = np.unique(np.clip(totest, -4, 2))
        totest = totest[~np.isnan(totest)]
        totest = totest[~np.isin(totest, rval.index)]

        if len(totest) == 0:
            # we need to do something

            try:
                totest = np.linspace(rval.dropna().index[0],
                                     rval.dropna().index[-1], 10)
                totest = np.unique(np.round(totest, 2))
                counter += 1
            except IndexError:
                totest = []
                counter += 1

        if counter > maxcount:
            log.info('SPINUP ERROR: (%s) maximum counter reached!' %
                     gdir.rgi_id)
            break

    # use minimal delta from rval
    rval = rval.astype(float)

    if rval['delta'].isna().all():
        return -999

    tbias = rval.dropna().idxmin().iloc[0]
    delta = np.sqrt(rval.loc[tbias, 'delta'])
    log.info('(%s) delta = %.2f (counter=%d)' % (gdir.rgi_id, delta, counter))

    if delta > fls[-1].dx_meter:
        log.info('SPINUP ERROR: (%s) minimum fit spinup failed!' %
                 gdir.rgi_id)

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)
    return tbias
