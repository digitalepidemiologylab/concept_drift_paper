import pandas as pd
import os
from utils.helpers import save_fig, paper_plot
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from tqdm import tqdm
import logging
import joblib
import multiprocessing
import pytz
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

sent_dict = {'positive': 1, 'neutral': 0, 'negative': -1}
num_cores = max(multiprocessing.cpu_count() - 1, 1)
# num_cores = 1

DATA_DIR = 'data'

def select_intervals(df, s_date, e_date, interval=timedelta(days=90)):
    df = df.sort_values('trained_at')
    df_all = pd.DataFrame()
    last_index = df.trained_at.nunique() - 1
    for i, (centroid_str, grp) in enumerate(df.groupby(df.trained_at)):
        centroid = datetime.strptime(centroid_str, '%Y-%m-%d')
        centroid = pytz.utc.localize(centroid)
        if i == 0:
            grp = grp[(grp.created_at > s_date) & (grp.created_at < e_date)]
            last_point = grp[(grp.created_at > centroid - interval/2) & (grp.created_at < centroid + interval/2)].sort_values('created_at').iloc[-1]
        else:
            if i == last_index:
                grp = grp[(grp.created_at > centroid - interval/2)].sort_values('created_at')
            else:
                grp = grp[(grp.created_at > centroid - interval/2) & (grp.created_at < centroid + interval/2)].sort_values('created_at')
            new_point = last_point.copy()
            new_point['trained_at'] = centroid_str
            last_point = grp.iloc[-1]
            grp = grp.append(new_point)
        df_all = df_all.append(grp)
    return df_all

def read_bert_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'fig4_bert.csv'), index_col='created_at', parse_dates=['created_at'])
    return df

def read_fasttext_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'fig4_fasttext.csv'), index_col='created_at', parse_dates=['created_at'])
    return df

@paper_plot
def main():
    df_bert = read_bert_data()
    df_fasttext = read_fasttext_data()

    s_date = datetime(2017, 7, 1, tzinfo=pytz.utc)
    e_date = datetime(2020, 10, 24, tzinfo=pytz.utc)

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.5), sharex=True, sharey=True)
    for ax, df, title in zip(axes, [df_fasttext, df_bert], ['FastText', 'BERT']):
        df = df[s_date:e_date]
        df = df.reset_index().melt(id_vars=['created_at'], var_name='trained_at', value_name='sentiment')
        df = select_intervals(df, s_date, e_date)

        palette = sns.color_palette('inferno', n_colors=df.trained_at.nunique())

        # plot
        sns.lineplot(x='created_at', y='sentiment', hue='trained_at', palette=palette, data=df, hue_order=sorted(df.trained_at.unique()), solid_capstyle='round', legend=ax==axes[0], ax=ax)
        
        # legend
        if ax == axes[0]:
            handles, labels = ax.get_legend_handles_labels()
            leg = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.05, 0), borderaxespad=0., frameon=False, title='Trained at')
            leg._legend_box.align = "left"

        # formatting
        ax.grid(True)
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 7]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.set_title(title, pad=4)

        # labels
        ax.xaxis.label.set_visible(False)
        ax.yaxis.label.set_visible(False)
        ax.set_ylim((0, .83))
    
    fig.subplots_adjust(hspace=.15)
    fig.text(.02, .5, 'Sentiment index $s$', rotation=90, ha='left', va='center')

    # save
    save_fig(plt.gcf(), 'fig4', version=1, plot_formats=['png', 'pdf'])


if __name__ == "__main__":
    main()
