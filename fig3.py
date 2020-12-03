import pandas as pd
import os
import sys
from utils.helpers import save_fig, paper_plot, cached, add_colorbar
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import numpy as np
import matplotlib.transforms
import matplotlib.ticker

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

DATA_DIR = 'data'

@paper_plot
def plot_fig3abc():
    # read data
    df_fleiss = pd.read_csv(os.path.join(DATA_DIR, 'fig3_fleiss.csv'))
    df_var = pd.read_csv(os.path.join(DATA_DIR, 'fig3_variability.csv'), index_col='label')
    df_dis = pd.read_csv(os.path.join(DATA_DIR, 'fig3_label_imbalance.csv'), index_col='label')

    # plot stuff
    height_ratios = [2.5, .6, 3]
    fig, axes = plt.subplots(3, 1, figsize=(2.5, 4.5), sharex=True, gridspec_kw=dict(hspace=0, height_ratios=height_ratios))
    cbar_opts = [dict(pad=.08, shrink=.4 * 2.5/hr, aspect=15) for hr in height_ratios] 
    plot_cbar = True
    common_args = dict(lw=.2, ec='white', annot=False, square=True)

    # plot distribution
    cmap = 'Purples'
    ax = axes[0]
    heatmap = sns.heatmap(data=df_dis, cmap=cmap, cbar=plot_cbar, cbar_kws=dict(label="Corpus size", **cbar_opts[0]), ax=ax, fmt='d', annot_kws=dict(fontsize=5), **common_args)
    ax.set_xticklabels(df_dis.columns.tolist())
    ax.tick_params(axis='both', direction='out')
    ax.set_title('Label imbalance', fontsize=7)

    # plot fleiss
    cmap = 'Reds'
    ax = axes[1]
    labels = [f'{l:.2f}'.lstrip('0') for l in df_fleiss.fleiss_kappa.values]
    heatmap = sns.heatmap(data=[df_fleiss.fleiss_kappa], cbar=plot_cbar, cmap=cmap, cbar_kws=dict(label="Fleiss' Kappa", **cbar_opts[1]), ax=ax, fmt='', annot_kws=dict(fontsize=5), **common_args)
    ax.tick_params(axis='both', direction='out')
    ax.set_yticklabels(['all'], rotation=0)
    ax.set_title('Annotator agreement', fontsize=7)

    # plot variance
    cmap = 'Greens'
    ax = axes[2]
    heatmap = sns.heatmap(data=df_var, cmap=cmap, cbar=plot_cbar, center=df_var.loc['all'].mean(), cbar_kws=dict(label='Embedding variance', **cbar_opts[2]), ax=ax, **common_args)
    ax.set_title('Corpus variability', fontsize=7)
    # ticks formatting
    ax.tick_params(axis='both', direction='out')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right')
    offset = matplotlib.transforms.ScaledTranslation(.06, 0, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    # common cosmetics
    for ax, cbar_ticks in zip(axes, [3, 3, 4]):
        # set labels invisible
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        # colorbar cosmetics
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(axis='y', direction='out')
        cbar.ax.yaxis.label.set_rotation(90)
        cbar.ax.yaxis.label.set_ha('center')
        cbar.ax.yaxis.label.set_va('top')
        cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(cbar_ticks))

    save_fig(fig, 'fig3abc', version=1, plot_formats=['png', 'pdf'], dpi=800)

@paper_plot
def plot_fig3d(cached=True):
    sim_matrix = {}
    keys = ['all', 'positive', 'neutral', 'negative']
    for k in keys:
        sim_matrix[k] = pd.read_csv(os.path.join(DATA_DIR, f'fig3d_{k}.csv'), index_col=0)
    num_plots = len(sim_matrix.keys())
    fig, _axes = plt.subplots(2, 2, figsize=(4, 4), sharex=True, sharey=True)
    axes = []
    for ax_row in _axes:
        for ax in ax_row:
            axes.append(ax)

    min_vals = []
    max_vals = []
    for key, df in sim_matrix.items():
        min_vals.append(df.values[np.triu_indices_from(df,k=1)].min())
        max_vals.append(df.values[np.triu_indices_from(df,k=1)].max())
    min_val = min(min_vals)
    max_val = max(max_vals)

    for key, ax in zip(keys, axes):
        df = sim_matrix[key]
        # normalize
        df = (df - min_val) / (max_val - min_val)
        mask = np.zeros(df.shape, dtype=bool)
        mask[np.tril_indices(len(df), k=-1)] = True
        cmap = 'Blues_r'
        sns.heatmap(data=df, mask=mask, cmap=cmap, cbar=False, cbar_kws=dict(label='Normalized\ncosine similarity', fraction=.025, pad=.08), vmax=1, center=.5, lw=.2, ec='white', ax=ax, square=True)

        # move axis labels
        ax.tick_params(axis='both', direction='out')

        ax.set_xticklabels(ax.get_xticklabels(), rotation=75, ha='right')
        offset = matplotlib.transforms.ScaledTranslation(.05, 0, fig.dpi_scale_trans)
        for label in ax.xaxis.get_majorticklabels():
            label.set_transform(label.get_transform() + offset)

        # title
        ax.set_title(key, fontsize=7)

    # add colorbar
    cbar = add_colorbar(fig, ax, x=.9, y=.42, length=.013, width=.2, vmin=0, vmax=1,
            label='Normalized\ncosine similarity', cmap=cmap, orientation='vertical')
    cbar.ax.tick_params(axis='y', direction='out')
    cbar.ax.yaxis.label.set_ha('center')
    cbar.ax.yaxis.label.set_va('top')
    cbar.ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))

    fig.subplots_adjust(hspace=.2, wspace=0)
    fig.suptitle('Corpus similarity', fontsize=7, y=.96)

    # save
    save_fig(fig, 'fig3d', version=1, plot_formats=['png', 'pdf'], dpi=800)

if __name__ == "__main__":
    # plot_fig3abc()
    plot_fig3d()
