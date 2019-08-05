import numpy as np
import pandas as pd


def calc_acdc(_obs, spinup, model, meta, col):

    obs = _obs.dropna().copy()

    try:
        spin = (spinup.loc[:, col] - spinup.loc[0, col]).dropna().iloc[-1][0]
    except IndexError:
        return pd.Series([np.nan])

    dl = spin + meta['dL2003'].iloc[0]
    # relative length change
    hist = model.loc[:, col] - model.loc[:, col].iloc[0] + dl

    # only look at points with measurements
    hist = hist.loc[obs.index].squeeze()

    # accumulated difference change
    acdc = (obs-hist).diff().abs().cumsum()
    acdc.iloc[0] = 0

    return acdc


def relative_length_change(meta, spinup, histrun):
    spin = (spinup.loc[:] - spinup.loc[0]).dropna().iloc[-1]
    dl = spin + meta['dL2003'].iloc[0]
    # relative length change
    rel_dl = histrun.loc[:] - histrun.iloc[0] + dl

    return rel_dl


def mae(obs, model):
    return np.mean(np.abs(obs-model).dropna())


def r2(obs, model):
    ix = obs.dropna().index
    return np.corrcoef(obs.dropna(), model[ix])[0, 1]
