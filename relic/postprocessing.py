import numpy as np
import pandas as pd

from relic.process_length_observations import add_custom_length
from relic.preprocessing import get_leclercq_observations


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


def mae_all(df, normalised=False):

    maeall = df.loc[:, df.columns != 'obs'].sub(df.loc[:, 'obs'], axis=0).\
        dropna().abs().mean()

    if normalised:
        return maeall/maeall.max()
    else:
        return maeall


def mae_diff_all(df, yr=10, normalised=False):
    # MAE of XX year difference
    df_dt = df.copy()
    df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
    rundif = df_dt.resample("%dY" % yr).mean().diff()

    maediff = rundif.loc[:, rundif.columns != 'obs'].\
        sub(rundif.loc[:, 'obs'], axis=0).dropna().abs().mean()

    if normalised:
        return maediff/maediff.max()
    else:
        return maediff


def r2(obs, model):
    ix = obs.dropna().index
    return np.corrcoef(obs.dropna(), model[ix])[0, 1]


def dummy_dismantel_multirun():
    # just to remember
    import ast
    # rvaldict keys are strings, transform to dict with
    # combinationdict=ast.literal_eval(rvaldictkey)


def merged_ids(mainid):
    # merged glacier id matches
    mids = {'RGI60-11.02715_merged': 'RGI60-11.02709',
            'RGI60-11.02119_merged': 'RGI60-11.02051'
            }

    return mids[mainid]


def glcnames(glid):
    namedict = {
        'RGI60-11.03646': 'Bosson',
        'RGI60-11.01238': 'Rhone',
        'RGI60-11.00106': 'Pasterze glacier (Austria)',
        'RGI60-11.00897': 'Hintereisferner',
        'RGI60-11.03643': 'Mer de Glace',
        'RGI60-11.01450': 'Great Aletsch glacier',
        'RGI60-11.01270': 'Upper Grindelwald glacier',
        'RGI60-11.02119': 'Roseg',
        'RGI60-11.02051': 'Tschierva',
        'RGI60-11.02119_merged': 'Roseg and Tschierva glacier (Switzerland)',
        'RGI60-11.00746': 'Gepatschferner (Austria)',
        'RGI60-11.03638': 'Argentiere glacier (France)',
        'RGI60-11.02245': 'Forno',
        'RGI60-11.01974': 'Forni',
        'RGI60-11.02916': 'Pre de bard',
        'RGI60-11.00929': 'Langtaler',
        'RGI60-11.00887': 'Gurgler',
        'RGI60-11.02715': 'Ferpecle',
        'RGI60-11.02709': 'Mont Mine glacier',
        'RGI60-11.02715_merged': 'Ferpecle and Mont Mine glacier (Switzerland)',
        'RGI60-11.01328': 'Unteraar',
        'RGI60-11.00992': 'Nierderjoch',
        'RGI60-11.02630': 'Zinal',
        'RGI60-11.02793': 'Saleina',
        'RGI60-11.01478': 'Fiescher',
        'RGI60-11.01698': 'Langgletscher',
        'RGI60-11.00872': 'Hüfi',
        'RGI60-11.02822': 'Gorner',
        'RGI60-11.02704': 'Allalin',
        'RGI60-11.02755': 'Tsidjore Nouve',
        'RGI60-11.02740': 'Trient',
        'RGI60-11.01946': 'Morteratsch glacier',
        'RGI60-11.01346': 'Lower Grindelwald glacier (Switzerland)',
    }
    return namedict[glid]


def runs2df(runs):
    meta, data = get_leclercq_observations()
    meta, data = add_custom_length(meta, data,
                                   ['RGI60-11.02051', 'RGI60-11.02709'])

    # get all glaciers
    glcs = []
    for run in runs:
        glcs += [gl['rgi_id'] for gl in list(run.values())[0]]
        #_glcs = [gl['rgi_id'] for gl in list(run.values())[0]]
        #if len(_glcs) > len(glcs):
        #    glcs = _glcs.copy()
    glcs = np.unique(glcs).tolist()

    # store results per glacier in a dict
    glcdict = {}

    for glid in glcs:
        # take care of merged glaciers
        rgi_id = glid.split('_')[0]
        _meta = meta.loc[meta['RGI_ID'] == rgi_id].copy()
        _data = data.loc[_meta.index[0]].copy()

        df = pd.DataFrame([], index=np.arange(1850, 2011))
        df.loc[_data.index, 'obs'] = _data

        if '_merged' in glid:
            mid = merged_ids(glid)

            _mmeta = meta.loc[meta['RGI_ID'] == mid].copy()
            _mdata = data.loc[_mmeta.index[0]].copy()
            dfmerge = pd.DataFrame([], index=np.arange(1850, 2011))
            dfmerge.loc[_mdata.index, 'obs'] = _mdata

        for nr, run in enumerate(runs):
            rlist = list(run.values())[0]
            try:
                rdic = [gl for gl in rlist if gl['rgi_id'] == glid][0]
            except IndexError:
                continue

            if np.isnan(rdic['tbias']):
                continue

            rkey = list(run.keys())[0]

            df.loc[rdic['rel_dl'].index, rkey] = rdic['rel_dl']

            if '_merged' in glid:
                dfmerge.loc[rdic['trib_dl'].index, rkey] = rdic['trib_dl']

        glcdict[glid] = df
        if '_merged' in glid:
            glcdict[mid] = dfmerge

    return glcdict


def pareto(glcdict):
    paretodict = {}

    for glc in glcdict.keys():

        # get my measures
        maes = mae_all(glcdict[glc], normalised=True)
        maediff = mae_diff_all(glcdict[glc], yr=10, normalised=True)

        # utopian
        up = [maes.min(), maediff.min()]

        # euclidian dist
        edisx = np.sqrt((maes-up[0])**2 + (maediff-up[1])**2).idxmin()

        if '_merged' in glc:
            mid = merged_ids(glc)
            # get my measures
            maes2 = mae_all(glcdict[mid], normalised=True)
            maediff2 = mae_diff_all(glcdict[mid], yr=10, normalised=True)
            up += [maes2.min(), maediff2.min()]

            edisx = np.sqrt((maes-up[0])**2 +
                            (maediff-up[1])**2 +
                            (maes2-up[2])**2 +
                            (maediff2-up[3])**2).idxmin()

        """
        import matplotlib.pyplot as plt
        plt.plot(maes, maediff, 'o')
        plt.plot(up[0], up[1], 'og')
        plt.plot(maes.loc[edisx], maediff.loc[edisx], '.r')
        plt.show()
        """
        paretodict[glc] = edisx

    return paretodict
