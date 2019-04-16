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
        raise RuntimeError('Did not find equilibrium.')


def minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, optimization):
    # Mass balance
    mb.temp_bias = tbias

    model = FluxBasedModel(fls, mb_model=mb,
                           time_stepping='default',
                           glen_a=glena)

    for _y in np.arange(150):
        model.run_until(model.yr + 1)
    try:
        relic_run_until_equilibrium(model)
    except (RuntimeError, ValueError):
        pass

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

    opti = scipy.optimize.minimize_scalar(minimize_dl,
                                          bracket=(fg, fg-1),
                                          tol=1e-2,
                                          args=(mb, fls, dl, len2003, glena,
                                                gdir, True),
                                          options={'maxiter': 30}
                                          )

    print('First optimization result:')
    print(opti)

    if np.sqrt(opti.fun) > 100:
        # try again, a bit colder...
        opti2 = scipy.optimize.minimize_scalar(minimize_dl,
                                               bracket=(opti.x-0.5, opti.x-1),
                                               tol=1e-2,
                                               args=(mb, fls, dl, len2003, glena,
                                                     gdir, True),
                                               options={'maxiter': 30}
                                              )
        print('Second optimization result:')
        print(opti2)

        if opti2.fun < 100:
            opti = opti2
        else:
            raise RuntimeError('Optimization failed....')

    # delta = opti.fun
    # just go with it and assert in the postprocessing

    assert np.sqrt(opti.fun) < 100

    tbias = opti.x

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)

    return tbias
