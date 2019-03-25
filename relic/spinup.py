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


def minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, optimization):
    # Mass balance
    mb.temp_bias = tbias

    model = FluxBasedModel(fls, mb_model=mb,
                           time_stepping='default',
                           glen_a=glena)
    model.run_until(150)
    model.run_until_equilibrium(rate=1e-4, ystep=10, max_ite=100)

    if optimization is True:
        dl_spinup = model.length_m - len2003
        delta = (dl - dl_spinup)**2
        print('tbias: %.2f  delta: %.4f' % (tbias, delta))
        return delta
    else:
        run_path = gdir.get_filepath('model_run',
                                     filesuffix='_spinup_%.3e' % glena,
                                     delete=True)
        diag_path = gdir.get_filepath('model_diagnostics',
                                      filesuffix='_spinup_%.3e' % glena,
                                      delete=True)
        model.run_until_and_store(model.yr+1, run_path=run_path,
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
                                          bracket=(fg, fg-0.2),
                                          tol=1e-4,
                                          args=(mb, fls, dl, len2003, glena,
                                                gdir, True),
                                          options={'maxiter': 10}
                                          )

    print(opti)
    delta = opti.fun
    assert np.sqrt(delta) < 100
    tbias = opti.x

    # --------- SPIN IT UP FOR REAL ---------------
    minimize_dl(tbias, mb, fls, dl, len2003, glena, gdir, False)
