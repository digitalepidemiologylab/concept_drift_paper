import os
import sys
import json
from utils.helpers import save_fig, paper_plot, cached, cached_parquet
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm
import sklearn.metrics
import numpy as np
from matplotlib.lines import Line2D
import logging
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


dataset_version = 1
repeats = 50

def read_data_fasttext(dataset_version=dataset_version):
    f_path = os.path.join('data', 'fig2_fasttext.csv')
    df = pd.read_csv(f_path, parse_dates=['centroid_day', 'centroid_day_train'])
    df = df[df['repeat'].isin(list(range(repeats)))]
    return df

def read_data_bert():
    f_path = os.path.join('data', 'fig2_bert.csv')
    df = pd.read_csv(f_path, parse_dates=['centroid_day', 'centroid_day_train'])
    return df

def compute_concept_drift_score(df, metric='f1_macro'):
    for (repeat, centroid_day_train), grp in df.groupby(['repeat', 'centroid_day_train']):
        # get initial performance
        initial_performance = grp[(grp.centroid_day_train == grp.centroid_day)][metric].values
        try:
            assert len(initial_performance) == 1
        except:
            logger.warning(f'Found {len(initial_performance)} values of same repeat {repeat} and train day {centroid_day_train}')
        initial_performance = initial_performance[0]
        # compute model drift score
        df.loc[(df.repeat == repeat) & (df.centroid_day_train == centroid_day_train), 'concept_drift'] = grp[metric] - initial_performance
        df.loc[(df.repeat == repeat) & (df.centroid_day_train == centroid_day_train), 'rel_concept_drift'] = 100*(grp[metric] - initial_performance)/initial_performance
    return df

@paper_plot
def plot_fig2():
    df_fasttext = read_data_fasttext()
    df_bert = read_data_bert()

    # constants
    ms = 2
    lw = .8
    ms_square = 2.5
    metric = 'f1_macro'

    # plot stuff
    fig, all_axes = plt.subplots(2, 2, sharex=True, figsize=(3.5, 3))
    for i, (df, (ax1, ax2), title) in enumerate(zip([df_fasttext, df_bert], [[all_axes[0][i], all_axes[1][i]] for i in range(len(all_axes))], ['FastText', 'BERT'])):
        palette = sns.color_palette('inferno', n_colors=df.centroid_day_train.nunique())

        # compute drift score
        df = compute_concept_drift_score(df, metric=metric)

        # train markers
        df_markers = df.groupby(['centroid_day_train', 'centroid_day'])[[metric, 'rel_concept_drift']].mean().reset_index().copy()
        df_markers = df_markers[df_markers.centroid_day_train == df_markers.centroid_day]

        # performance score
        df = df[['centroid_day_train', 'centroid_day', metric, 'repeat', 'rel_concept_drift']]
        df['train_day'] = df['centroid_day_train'].apply(lambda s: s.strftime('%Y-%m-%d'))
        df = df.sort_values(['train_day', 'centroid_day'])
        sns.lineplot(x='centroid_day', y=metric, hue='train_day', ci=95, err_style='band', marker='o', lw=lw, ms=ms, mec='none', palette=palette, data=df, legend=i == 1, ax=ax1)
        sns.lineplot(x='centroid_day', y=metric, hue='centroid_day_train', ci=None, lw=0, marker='s', mec='none', ms=ms_square, palette=palette, data=df_markers, ax=ax1, legend=False)

        # model drift score
        sns.lineplot(x='centroid_day', y='rel_concept_drift', hue='centroid_day_train', data=df, palette=palette, marker='o', lw=lw, ms=ms, mec='none', legend=False, ax=ax2)
        sns.lineplot(x='centroid_day', y='rel_concept_drift', hue='centroid_day_train', ci=None, lw=0, marker='s', mec='none', ms=ms_square, palette=palette, data=df_markers, ax=ax2, legend=False)

        # axis labels
        if i == 0:
            ax1.set_ylabel('F1-macro')
            ax2.set_ylabel('% Relative\nperforamnce change')
        else:
            ax1.yaxis.label.set_visible(False)
            ax2.yaxis.label.set_visible(False)
            ax1.yaxis.set_ticklabels([])
            ax2.yaxis.set_ticklabels([])

        ax2.locator_params(axis='y', nbins=5)
        ax1.set_ylim((.32, .65))
        ax2.set_ylim((-28, 13))

        # titles
        ax1.set_title(title)

        # common formatting
        for ax in [ax1, ax2]:
            ax.grid(True)
            ax.xaxis.set_minor_locator(mdates.MonthLocator())
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

            # lims
            ax.set_xlim((datetime(2018, 7, 1), datetime(2020, 9, 1)))
            ax.xaxis.label.set_visible(False)

        # create legends
        if i == 1:
            handles, labels = ax1.get_legend_handles_labels()
            legend_opts = dict(
                    loc='center left',
                    bbox_to_anchor=(1.1, .2),
                    borderaxespad=0.,
                    handlelength=1,
                    handletextpad=.8,
                    frameon=False
                    )
            leg = ax1.legend(handles, labels, title='Trained on data up to', **legend_opts)
            leg._legend_box.align = "left"

            leg2 = [
                    Line2D([0], [0], lw=0, marker='s', mec='none', ms=3, color=palette[0], label='Train & evaluate'),
                    Line2D([0], [0], lw=0, marker='o', mec='none', ms=3, color=palette[0], label='Evaluate')]
            ax2.legend(handles=leg2, **legend_opts)

    fig.subplots_adjust(hspace=.1, wspace=.08)

    # save
    save_fig(plt.gcf(), 'fig2', version=1, plot_formats=['png', 'pdf'])


if __name__ == "__main__":
    plot_fig2()
