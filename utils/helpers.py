import matplotlib.pyplot as plt
import os
from functools import wraps
from matplotlib.ticker import FormatStrFormatter
import matplotlib.dates as mdates
import pandas as pd
import pickle
import shelve


def paper_plot(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        style_sheet = os.path.join(find_project_root(), 'stylesheets', 'figure_small.mplstyle')
        plt.style.use(style_sheet)
        return func(*args, **kwargs)
    return wrapper

def use_stylesheet(name='paper'):
    style_sheet = os.path.join(find_project_root(), 'stylesheets', '{}.mplstyle'.format(name))
    plt.style.use(style_sheet)

def label_subplots(axes, upper_case=True, offset_points=(-40, 0)):
    start_ord = 65 if upper_case else 97
    for ax, lab in zip(axes, ['{}'.format(chr(j)) for j in range(start_ord, start_ord + len(axes))]):
        ax.annotate(lab, (0, 1), xytext=offset_points, xycoords='axes fraction', textcoords='offset points',
                ha='right', va='top', weight='bold')

def save_fig(fig, name, plot_formats=['png', 'eps'], version=1, dpi=600):
    def f_name(fmt):
        f_name = '{}.{}'.format(name, fmt)
        return os.path.join(folder_path, f_name)
    folder_path = os.path.join(find_project_root(), 'plots', name, 'v{}'.format(version))
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    for fmt in plot_formats:
        if not fmt == 'tiff':
            f_path = f_name(fmt)
            print('Writing figure file {}'.format(os.path.abspath(f_path)))
            fig.savefig(f_name(fmt), bbox_inches='tight', dpi=dpi)
    if 'tiff' in plot_formats:
        os.system("convert {} {}".format(f_name('png'), f_name('tiff')))

def format_time_axis(ax, ticks='weekly', major_ticks_out=True, minor_ticks_out=True):
    if ticks == 'daily':
        ax.xaxis.set_minor_locator(mdates.DayLocator())
        ax.xaxis.set_major_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif ticks == 'weekly':
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    elif ticks == 'monthly':
        ax.xaxis.set_minor_locator(mdates.MonthLocator())
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    if minor_ticks_out:
        ax.tick_params(axis='x', direction='out', which='minor', zorder=2, size=2)
    if major_ticks_out:
        ax.tick_params(axis='x', direction='out', which='major', zorder=2, size=4)
    return ax

def cache_folder(use_data_folder=False, subfolder=None):
    current_folder = os.path.dirname(os.path.realpath(__file__))
    if use_data_folder:
        f_path = os.path.join(current_folder, '..', 'data')
    else:
        if subfolder is None:
            f_path = os.path.join(current_folder, '..', 'data', 'cache')
        else:
            f_path = os.path.join(current_folder, '..', 'data', 'cache', subfolder)
    if not os.path.isdir(f_path):
        os.makedirs(f_path)
    return os.path.abspath(f_path)

def get_cache_path(f_name, use_data_folder=False, subfolder=None):
    f_path = os.path.join(cache_folder(subfolder=subfolder, use_data_folder=use_data_folder), f_name)
    return os.path.abspath(f_path)

def find_project_root(num_par_dirs=8):
    for i in range(num_par_dirs):
        par_dirs = i*['..']
        current_dir = os.path.join(*par_dirs, '.git')
        if os.path.isdir(current_dir):
            break
    else:
        raise FileNotFoundError('Could not find project root folder.')
    return os.path.join(*os.path.split(current_dir)[:-1])

def cached(f_name):
    """Uses a shelve to pickle return values of function calls"""
    cache_path = get_cache_path(f_name)
    def cacheondisk(fn):
        db = shelve.open(cache_path)
        @wraps(fn)
        def usingcache(*args, **kwargs):
            __cached = kwargs.pop('__cached', True)
            key = repr((args, kwargs))
            if not __cached or key not in db:
                ret = db[key] = fn(*args, **kwargs)
            else:
                print(f'Using cache')
                ret = db[key]
            return ret
        return usingcache
        db.close()
    return cacheondisk

def cached_parquet(f_name):
    """Uses a parquet to cache return values of function calls"""
    cache_path = get_cache_path(f_name, use_data_folder=True)
    def cacheondisk(fn):
        @wraps(fn)
        def usingcache(*args, **kwargs):
            __cached = kwargs.pop('__cached', True)
            key = repr((args, kwargs))
            if not __cached or not os.path.isfile(cache_path):
                ret = fn(*args, **kwargs)
                ret.to_parquet(cache_path)
            else:
                print(f'Using cache')
                ret = pd.read_parquet(cache_path)
            return ret
        return usingcache
    return cacheondisk
