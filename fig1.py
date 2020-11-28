import sys
import logging
import seaborn as sns
from utils.helpers import save_fig, paper_plot
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import glob
import os
from pandas.plotting import register_matplotlib_converters
import matplotlib.ticker


register_matplotlib_converters()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

# constants
interval_days = 90
num_train_bins = 4
train_test_frac = .6
num_train_samples_per_bin = 400
num_test_samples_per_bin = 150

DATA_DIR = 'data'

def read_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'fig1.csv'), parse_dates=['centroid_day', 'start_day', 'end_day'])
    return df

@paper_plot
def main():
    # read data
    df = read_data()
    df = df.set_index('centroid_day')

    # first datapoint was skipped (by accident)
    df = df.iloc[1:]

    palette = sns.color_palette('Blues', n_colors=len(df))

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 1.8))

    # compute baselines
    df['base'] = 0
    df.loc[df.iloc[1:5].index, 'base'] = num_train_samples_per_bin
    df['train'] = 0
    df.loc[df.iloc[:1].index, 'train'] = num_train_samples_per_bin
    df.loc[df.iloc[5:].index, 'train'] = num_train_samples_per_bin
    df['test'] = 0
    df.loc[df.iloc[3:].index, 'test'] = num_test_samples_per_bin
    df['other'] = df['base'] + df['train'] + df['test']
    df['total'] = df['all'].copy()
    df['all'] -= df['other']

    # plot
    width = 40
    ax.bar(df.index.to_pydatetime(), df['base'].values.tolist(), width=width, color='C0', hatch='/////',  label='Training data $b_1$')
    ax.bar(df.index.to_pydatetime(), df['train'].values.tolist(), width=width, color='C0', label=f'Train')
    ax.bar(df.index.to_pydatetime(), df['test'].values.tolist(), bottom=df['train']+df['base'], color='C3', width=width, label=f'Eval')
    ax.bar(df.index.to_pydatetime(), df['all'].values.tolist(), bottom=df['other'].values.tolist(), color='.8', width=width, label='Unused')

    # annotate
    ax.annotate('$b_0$', (df.index[3], df.iloc[3].total),  ha='center', va='bottom', xytext=(0, 1), textcoords='offset points')
    ax.annotate('$b_1$', (df.index[4], df.iloc[4].total),  ha='center', va='bottom', xytext=(0, 1), textcoords='offset points')
    ax.annotate('$b_{8}$', (df.index[11], df.iloc[11].total),  ha='center', va='bottom', xytext=(0, 1), textcoords='offset points')

    # legend
    leg = plt.legend(loc='center left', title='Annotation type', bbox_to_anchor=(1.05, .5), frameon=False, handleheight=.4, handlelength=1.2)
    leg._legend_box.align = "left"

    # tick frequency
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(base=400))

    # tick direction
    ax.tick_params(axis='x', direction='out', which='minor', zorder=2, size=2)
    ax.tick_params(axis='x', direction='out', which='major', zorder=2, size=4)

    ax.set_ylim((0, 1500))
    ax.set_xlim((datetime(2017, 7, 1), datetime(2020, 9, 1)))
    ax.grid(True)

    # labels
    ax.set_ylabel('Number of annotations')

    # cosmetics
    sns.despine()

    # save
    save_fig(fig, 'fig1', version=1, plot_formats=['png', 'pdf'])

if __name__ == "__main__":
    main()
