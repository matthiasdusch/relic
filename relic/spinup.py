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

    for nr in [1, 2, 3]:

        if nr == 1:
            br1 = fg
            br2 = fg-1
        else:
            # opti_old = opti.copy()
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
            break
        elif nr == 3:
            raise RuntimeError('Optimization failed....')

    tbias = opti.x

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)

    return tbias
