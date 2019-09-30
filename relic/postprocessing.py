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
        # return maeall/maeall.max()
        return (maeall - maeall.min())/(maeall.max()-maeall.min())
    else:
        return maeall


def mae_diff_mean(df, yr, normalised=False):
    # MAE of XX year difference
    df_dt = df.copy()
    df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
    rundif = df_dt.resample("%dY" % yr).mean().diff()

    maediff = rundif.loc[:, rundif.columns != 'obs']. \
        sub(rundif.loc[:, 'obs'], axis=0).dropna().abs().mean()

    if normalised:
        return (maediff - maediff.min())/(maediff.max()-maediff.min())
    else:
        return maediff


def mae_diff_yearly(df, yr, normalised=False):
    # MAE of XX year difference
    df_dt = df.copy()
    df_dt.index = pd.to_datetime(df_dt.index, format="%Y")
    # rundif = df_dt.resample("%dY" % yr).mean().diff()
    # rundif = df.diff(yr).copy()
    rundif = df.copy() * np.nan

    obsyrs = pd.Series(df.obs.dropna().index)
    for y1 in obsyrs.iloc[:-yr]:
        y2 = obsyrs[((obsyrs-y1-yr) >= 0)].iloc[0]
        rundif.loc[y1, :] = df.loc[y2, :] - df.loc[y1, :]

    maediff = rundif.loc[:, rundif.columns != 'obs'].\
        sub(rundif.loc[:, 'obs'], axis=0).dropna().abs().mean()

    if normalised:
        return (maediff - maediff.min())/(maediff.max()-maediff.min())
    else:
        return maediff


def diff_corr(df, yr=10, normalised=False):
    # MAE of XX year difference
    rundif = df.copy() * np.nan

    obsyrs = pd.Series(df.obs.dropna().index)
    for y1 in obsyrs.iloc[:-yr]:
        y2 = obsyrs[((obsyrs-y1-yr) >= 0)].iloc[0]
        rundif.loc[y1, :] = df.loc[y2, :] - df.loc[y1, :]

    corre = rundif.dropna().corr()['obs'].loc[~rundif.columns.isin(['obs'])]

    if normalised:
        return 1 - (corre - corre.min())/(corre.max() - corre.min())
    else:
        return corre


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
    mids = {'RGI60-11.02709_merged': 'RGI60-11.02715',
            'RGI60-11.02051_merged': 'RGI60-11.02119'
            }

    return mids[mainid]


def glcnames(glid):
    namedict = {
        'RGI60-11.03646': 'Bossons glacier (France)',
        'RGI60-11.01238': 'Rhone',
        'RGI60-11.00106': 'Pasterze glacier (Austria)',
        'RGI60-11.00897': 'Hintereisferner',
        'RGI60-11.03643': 'Mer de Glace',
        'RGI60-11.01450': 'Great Aletsch glacier',
        'RGI60-11.01270': 'Upper Grindelwald glacier',
        'RGI60-11.02119': 'Roseg',
        'RGI60-11.02051': 'Tschierva',
        'RGI60-11.02051_merged': 'Tschierva (with Roseg)',
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
        'RGI60-11.02709_merged': 'Mont Mine (with Ferpecle)',
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

        if 'XXX_merged' in glid:
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

            if 'XXX_merged' in glid:
                dfmerge.loc[rdic['trib_dl'].index, rkey] = rdic['trib_dl']

        glcdict[glid] = df
        if 'XXX_merged' in glid:
            glcdict[mid] = dfmerge

    return glcdict


def pareto(glcdict, maedyr):
    paretodict = {}

    for glc in glcdict.keys():

        # get my measures
        maes = mae_all(glcdict[glc], normalised=True)
        maediff = mae_diff_yearly(glcdict[glc], maedyr, normalised=True)
        # corre = diff_corr(glcdict[glc], yr=maedyr, normalised=True)

        # utopian
        up = [maes.min(), maediff.min()]
        # up = [maes.min(), maediff.min(), corre.min()]

        # euclidian dist
        edisx = np.sqrt(5*(maes-up[0])**2 +
                        (maediff-up[1])**2).idxmin()
        #                (corre-up[2])**2).idxmin()

        if 'XXX_merged' in glc:
            mid = merged_ids(glc)
            # get my measures
            maes2 = mae_all(glcdict[mid], normalised=True)
            maediff2 = mae_diff_yearly(glcdict[mid], maedyr, normalised=True)
            # corre2 = diff_corr(glcdict[mid], yr=maedyr, normalised=True)
            up += [maes2.min(), maediff2.min()]
            # up += [maes2.min(), maediff2.min(), corre2.min()]

            edisx = np.sqrt((maes-up[0])**2 +
                            (maediff-up[1])**2 +
                            (maes2-up[2])**2 +
                            (maediff2-up[3])**2).idxmin()

        paretodict[glc] = edisx

        plot_pareto(glc, edisx, maes, maediff)

    return paretodict


def plot_pareto(glc, edisx, maes, maediff):
    import matplotlib.pyplot as plt
    import ast
    import os
    from colorspace import diverging_hcl
    from matplotlib.colors import ListedColormap

    fig1, ax1 = plt.subplots(figsize=[15, 8])

    ax1.plot(0, 0, '*k', markersize=20, label='utopian solution')

    ax1.plot(maes.loc[edisx], maediff.loc[edisx], '*r', color='C2',
             markersize=25, markeredgecolor='C3',
             label='choosen as best solution')

    ax1.plot(maes.loc[maes.idxmin()], maediff.loc[maes.idxmin()],
             '*r', color='C0', markersize=20,
             label='best result for Objective 1')
    ax1.plot(maes.loc[maediff.idxmin()], maediff.loc[maediff.idxmin()],
             '*r', color='C1', markersize=20,
             label='best result for Objective 2')

    df = pd.DataFrame([], columns=['mae', 'maedif', 'mb', 'prcp'])

    # prcp
    cmap = diverging_hcl(h=[260, 0], c=80, l=[30, 90], power=2).cmap(n=18)
    cmap = cmap(np.arange(cmap.N))
    cmap[:, -1] = np.append(np.linspace(1, 0.5, 9)**2, np.linspace(0.5, 1, 9)**2)
    prcpcmp = ListedColormap(cmap)

    # mbbias
    cmap = diverging_hcl(h=[130, 43], c=100, l=[70, 90], power=1.5).cmap(n=8)
    cmap = cmap(np.arange(cmap.N))
    cmap[:, -1] = np.append(np.linspace(1, 0.5, 4)**2, np.linspace(0.5, 1, 4)**2)
    mbbcmp = ListedColormap(cmap)

    for run in maes.index:
        prm = ast.literal_eval('{' + run + '}')

        df = df.append({'mae': maes.loc[run],
                        'maedif': maediff.loc[run],
                        'mb': prm['mbbias'],
                        'prcp': prm['prcp_scaling_factor']},
                       ignore_index=True)

        """
        plt.plot(maes.loc[run], maediff.loc[run], 'ok',
                 color=mbdicc[prm['mbbias']],
                 alpha=mbdica[prm['mbbias']])

        plt.plot(maes.loc[run], maediff.loc[run], 'ok',
                 color=pcdicc[prm['prcp_scaling_factor']],
                 alpha=pcdica[prm['prcp_scaling_factor']])
        """

    sc2 = ax1.scatter(df.mae, df.maedif, c=df.mb, cmap=mbbcmp, label='')
    sc1 = ax1.scatter(df.mae, df.maedif, c=df.prcp, cmap=prcpcmp.reversed(),
                      label='')

    cx1 = fig1.add_axes([0.71, 0.38, 0.05, 0.55])
    cb1 = plt.colorbar(sc1, cax=cx1)
    cb1.set_label('Precipitation scaling factor', fontsize=16)

    cx2 = fig1.add_axes([0.85, 0.38, 0.05, 0.55])
    cb2 = plt.colorbar(sc2, cax=cx2)
    cb2.set_label('Mass balance bias', fontsize=16)

    name = glcnames(glc)
    ax1.set_title(name, fontsize=30)
    ax1.tick_params(axis='both', which='major', labelsize=22)

    ax1.set_ylabel('5yr difference MAE (normalised)',
                   fontsize=26)
    ax1.set_xlabel('MAE of relative length change (normalised)', fontsize=26)

    ax1.legend(bbox_to_anchor=(1.04, 0),
               fontsize=18, loc="lower left", ncol=1)

    ax1.grid(True)
    fig1.tight_layout()
    pout = '/home/matthias/length_change_1850/multi/array/190926/pareto'
    fn1 = os.path.join(pout, 'pareto_%s.png' % glc)
    # fn2 = os.path.join(pout, 'pareto_%s.pdf' % name.split()[0])
    fig1.savefig(fn1)
    # fig1.savefig(fn2)
