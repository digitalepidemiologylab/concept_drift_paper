import sys
import logging
import seaborn as sns
from utils.helpers import save_fig, paper_plot, curly_brace
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import glob
import os
from pandas.plotting import register_matplotlib_converters
import matplotlib.ticker
import matplotlib as mpl

mpl.rcParams['hatch.linewidth'] = 0.5

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

    # plot
    fig, ax = plt.subplots(1, 1, figsize=(3, 1.8))

    # compute baselines
    df['b1_train'] = 0
    df['b1_test'] = 0
    df['train'] = 0
    df['test'] = 0
    # b1 train
    df.loc[df.iloc[1:5].index, 'b1_train'] = num_train_samples_per_bin
    # b1 test
    df.loc[df.iloc[4:].index, 'b1_test'] = num_test_samples_per_bin
    # train
    df.loc[df.iloc[:1].index, 'train'] = num_train_samples_per_bin
    df.loc[df.iloc[5:].index, 'train'] = num_train_samples_per_bin
    # test
    df.loc[df.iloc[3:4].index, 'test'] = num_test_samples_per_bin
    df['other'] = df['b1_train'] + df['b1_test'] + df['train'] + df['test']
    df['total'] = df['all'].copy()
    df['all'] -= df['other']

    # plot
    width = 40
    ax.bar(df.index.to_pydatetime(), df['train'].values.tolist(), width=width, color='C0', label=f'Train')
    ax.bar(df.index.to_pydatetime(), df['test'].values.tolist(), bottom=df['train']+df['b1_train'], color='C3', width=width, label=f'Eval')
    ax.bar(df.index.to_pydatetime(), df['b1_train'].values.tolist(), width=width, color='C0', hatch=6*'/',  label='Training data $b_1$', ec='white')
    ax.bar(df.index.to_pydatetime(), df['b1_test'].values.tolist(), bottom=df['b1_train'] + df['train'], width=width, color='C3', hatch=6*'/',  label='Eval datasets $b_1$', ec='white')
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
    ax.set_xlim((datetime(2017, 10, 1), datetime(2020, 9, 1)))
    ax.grid(True)

    # annotations
    ts = ax.transAxes
    coords = ts.transform([0, .5])
    tr = mpl.transforms.Affine2D().rotate_deg_around(*coords, 90)
    t = ts + tr
    brace = curly_brace(x=.18, y=-.85, width=.03, height=.54, lw=.5, pointing='right', transform=t, color='.15')
    ax.add_artist(brace)
    ax.text(.25, .85, 'training window\nfor $b_1$', ha='center', va='bottom', transform=ts, fontsize=7)

    # labels
    ax.set_ylabel('Number of annotations')

    # cosmetics
    sns.despine()

    # save
    save_fig(fig, 'fig1', version=1, plot_formats=['png', 'pdf'])

if __name__ == "__main__":
    main()
