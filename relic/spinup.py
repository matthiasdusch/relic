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
        dl_spinup = model.length_m - len2003
        delta = (dl - dl_spinup)**2
        print('tbias: %.2f  delta: %.4f' % (tbias, delta))
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

    found_fit = False
    left = 1
    right = 1

    while True:
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

        # if cmin left or right of values only test there
        if np.sum(rval.index > cmin) == 0:
            totest = np.round(np.linspace(cmin, cmin+0.5, 5)[1:], 2)
            break
        if np.sum(rval.index < cmin) == 0:
            totest = np.round(np.linspace(cmin, cmin-0.5, 5)[1:], 2)
            break

        # if our minimum is between two values, test there
        idx = np.where(cmin == rval.index)[0][0]
        totest = np.unique(np.round(np.linspace(rval['delta'].iloc[idx-1],
                                                rval['delta'].iloc[idx+1], 5),
                                    2))
        # if this is to small, we are on the wrong track, try a polyfit
        if len(totest) <= 2:
            y = rval.dropna().delta.values
            x = rval.dropna().index
            x_new = np.arange(x.min()-1, x.max()+1, 0.01)
            coef = poly.polyfit(x, y, 2)
            fit2d = poly.polyval(x_new, coef)

            if (fit2d < 0).any():
                log.info('SPINUP ERROR: (%s) negative fit should not happen!' %
                         gdir.rgi_id)
                fit2d = np.abs(fit2d)

            pmin = x_new[fit2d.argmin()]
            totest = np.geomspace(pmin, pmin*3, 5)
            totest = np.unique(np.round(np.append(totest, pmin-(totest-pmin)),
                                        2))

        if counter == 30:
            log.info('SPINUP ERROR: (%s) maximum counter reached!' %
                     gdir.rgi_id)
            break

    """
        # we want at least 2 values left/right of the current minima
        if np.sum(rval.index > cmin) < 2:
            totest = np.linspace(cmin, cmin+right, 5)[1:]
            right += 1  # to hopefully avoid endless loops
            continue
        if np.sum(rval.index < cmin) < 2:
            totest = np.linspace(cmin, cmin-left, 5)[1:]
            left += 1
            continue
        left = 1
        right = 1

        # if we have enough values, let's try and fit a polynomial

        y = rval.dropna().delta.values
        x = rval.dropna().index
        x_new = np.arange(x.min()-1, x.max()+1, 0.01)
        coef = poly.polyfit(x, y, 2)
        fit2d = poly.polyval(x_new, coef)

        if (fit2d < 0).any():
            log.info('SPINUP ERROR: (%s) negative fit should not happen!' %
                     gdir.rgi_id)
            fit2d = np.abs(fit2d)

        pmin = x_new[fit2d.argmin()]

        totest = np.geomspace(pmin, pmin*3, 5)
        totest = np.unique(np.round(np.append(totest, pmin-(totest-pmin)), 2))

        if counter == 30:
            log.info('SPINUP ERROR: (%s) maximum counter reached!' %
                     gdir.rgi_id)
            break

    """
    """
    # first test
    for tb in totest:

        delta = minimize_dl(tb, mb, fls, dl, len2003, glena, gdir, True)
        if delta == len2003**2:
            delta = np.nan
        else:
            rval.loc[tb, 'delta'] = delta

    totest = np.array([])
    # current minima
    cmin = rval.idxmin().iloc[0]

    # we want at least 2 values left/right of the current minima
    if np.sum(rval.index > cmin) < 2:
        totest = np.append(totest, np.linspace(cmin, cmin+1, 5)[1:])
    if np.sum(rval.index < cmin) < 2:
        totest = np.append(totest, np.linspace(cmin, cmin-1, 5)[1:])

    # second test
    for tb in totest:

        delta = minimize_dl(tb, mb, fls, dl, len2003, glena, gdir, True)
        if delta == len2003**2:
            delta = np.nan
        else:
            rval.loc[tb, 'delta'] = delta

    # we need at least some good runs
    if len(rval.dropna()) < 4:
        log.info('SPINUP ERROR: (%s) only %d good spinups!' %
                 (gdir.rgi_id, len(rval.dropna())))

    # fit a polynom to the values we have
    y = rval.dropna().delta.values
    x = rval.dropna().index
    x_new = np.arange(x.min()-1, x.max()+1, 0.01)
    coef = poly.polyfit(x, y, 2)
    fit2d = poly.polyval(x_new, coef)

    # check if positive
    if fit2d[0] < 0:
        log.info('SPINUP ERROR: (%s) negative fit, that should not happen!' %
                 gdir.rgi_id)
        fit2d = np.abs(fit2d)

    # check if minimim value is not at the border
    if (np.argmin(fit2d) == 0) | (np.argmin(fit2d) == (len(fit2d)-1)):
        log.info('SPINUP ERROR: (%s) minimum fit at the border!' %
                 gdir.rgi_id)

    # tbias from polyfit
    tbias = x_new[fit2d.argmin()]
    delta = minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, True)
    log.info('(%s) delta = %.2f' % (gdir.rgi_id, np.sqrt(delta)))

    # oder direkt von rval
    if delta == len2003**2:
        log.info('SPINUP ERROR: (%s) minimum fit spinup failed!' %
                 gdir.rgi_id)
        tbias = rval.dropna().idxmin().iloc[0]
    """

    # use minimal delta from rval
    tbias = rval.dropna().idxmin().iloc[0]
    delta = np.sqrt(rval.loc[tbias, 'delta'])
    log.info('(%s) delta = %.2f (counter=%d)' % (gdir.rgi_id, delta, counter))

    if delta > fls[-1].dx_meter:
        log.info('SPINUP ERROR: (%s) minimum fit spinup failed!' %
                 gdir.rgi_id)

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)
    return tbias
