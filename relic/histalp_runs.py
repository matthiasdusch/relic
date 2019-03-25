import numpy as np
import xarray as xr

from oggm import tasks, cfg
from oggm.core.flowline import FileModel

from relic.spinup import spinup_with_tbias


def minimize_glena(glena, gdir, meta, obs_ye, obs_dl, optimization):

    # finish OGGM tasks
    tasks.mass_conservation_inversion(gdir, glen_a=glena)
    tasks.filter_inversion_output(gdir)
    tasks.init_present_time_glacier(gdir)

    # --------- HOW SHALL WE SPIN ---------------

    # how long are we at initialization
    fls = gdir.read_pickle('model_flowlines')
    len2003 = fls[-1].length_m
    # how long shall we go? MINUS for positive length change!
    dl = -meta['dL2003'].iloc[0]

    spinup_with_tbias(gdir, fls, dl, len2003, glena=glena)
    tmp_mod = FileModel(gdir.get_filepath('model_run', filesuffix='_spinup'))

    # --------- HIST IT DOWN ---------------
    tasks.run_from_climate_data(gdir, ys=meta['first'].iloc[0], ye=obs_ye,
                                init_model_fls=tmp_mod.fls,
                                output_filesuffix='_histalp_%.3e' % glena,
                                glen_a=glena)

    # assert that global glen_a is still unused
    assert np.isnan(cfg.PARAMS['glen_a'])

    # compare hist run to observations
    ds = xr.open_dataset(gdir.get_filepath('model_diagnostics',
                                           filesuffix='_histalp_%.3e' % glena))
    histalp_dl = ds.length_m.values[-1] - ds.length_m.values[0]
    delta = (histalp_dl - obs_dl)**2
    print('glenA: %.2e  delta: %.4f' % (glena, delta))
    if optimization is True:
        return delta
    """
    else:
        dlhist = ds.length_m.to_dataframe()['length_m']
        # set first year to 0
        dlhist -= dlhist.iloc[0]
        # add actual difference between spinup and obs
        spin_offset = (tmp_mod.length_m-len2003) - dl
        # spin_offset is positive if spinoff is to large
        dlhist += spin_offset

        return dlhist
    """
