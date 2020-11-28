import logging
import pandas as pd
import os
from tqdm import tqdm
from collections import Counter
from datetime import datetime, timedelta
import pytz
import numpy as np
import shutil
import json
import joblib
import multiprocessing
import sys
sys.path.append('text_classification')
from text_classification.models.fasttextmodel import FastTextModel
from text_classification.utils.config_reader import ConfigReader
from munch import DefaultMunch


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)

dataset_version = 1
save_model = False
DATA_DIR = 'data'


params =  {
    "write_test_output": True,
    "overwrite": True,
    "replace_user_with": "user",
    "replace_url_with": "url",
    "save_model": save_model,
    "lemmatize": False,
    "remove_punct": True,
    "lower_case": True,
    "remove_accents": True,
    "dim": 10,
    "num_epochs": 500,
    "n_grams": 1,
    "learning_rate": 0.01,
}

def train_test(run_name, train_path, test_paths, args):
    config = {}
    project_root = 'modeling'
    run = {
        'model': 'fasttext',
        'train_data': train_path,
        'test_data': test_paths[0],
        'name': run_name,
        'tmp_path': os.path.join(project_root, 'tmp'),
        'data_path': os.path.join(project_root, 'data'),
        'other_path': os.path.join(project_root, 'other', 'models'),
        'output_path': os.path.join(project_root, 'output', run_name),
        **params
        }
    if os.path.isdir(run['output_path']):
        shutil.rmtree(run['output_path'])
    os.makedirs(run['output_path'])
    f_path = os.path.join(run['output_path'], 'run_config.json')
    with open(f_path, 'w') as f:
        json.dump(run, f, indent=4)
    run = DefaultMunch.fromDict(run, None)
    model = FastTextModel()
    logger.info(f'Starting train/test for {run_name}...')
    logger.info(f'Training model on {train_path}...')
    model.train(run)
    results = []
    for test_path, centroid_day in zip(test_paths, args['centroid_days_test']):
        run['test_data'] = test_path
        logger.info(f'Testing model on {test_path}...')
        result = model.test(run)
        logger.info(f"... F1-score: {result['f1_macro']:.3f}")
        result['train_path'] = train_path
        result['test_path'] = test_path
        result['name'] = run_name
        result['centroid_day'] = centroid_day
        result = {**args, **result}
        results.append(result)
    # cleanup
    if not save_model:
        shutil.rmtree(run['output_path'])
    return results

def get_data():
    df = pd.read_csv(os.path.join(DATA_DIR, 'annotations_merged.csv'), parse_dates=['created_at', 'annotation_created_at'], dtype={'id': str})
    return df

def get_num_examples_in_smallest_bin(df, centroid_days, interval):
    num_examples = []
    for centroid_day in centroid_days:
        num_examples.append(len(df[(df.created_at > centroid_day - interval/2) & (df.created_at < centroid_day + interval/2)]))
    return min(num_examples)

def run_concept_drift(seed=42):
    # read annotation data
    df_all = get_data()

    if 'text' not in df_all:
        logger.warning(f'Text column is not present in raw data. Running experiments with synthetitc data instead!')
        df_all['text'] = [f'Example text {i}' for i in df_all.index.tolist()]

    logger.info(f'Loaded total of {len(df_all):,} labels')

    # contants
    interval_days = 90
    repeats = 50
    num_train_bins = 4
    include_test_samples_to_pool = False
    num_train_samples_per_bin = int(800/num_train_bins)
    num_test_samples_per_bin = 150

    # setup
    step = f'{interval_days}D'
    interval = timedelta(days=interval_days)
    start_date_exp = df_all.created_at.min()
    end_date_exp = df_all.created_at.max()
    centroid_days = pd.date_range(start_date_exp+interval/2, end_date_exp-interval/2, freq=step)
    min_examples_per_bin = get_num_examples_in_smallest_bin(df_all, centroid_days, interval)
    logger.info(f'Min examples per bin: {min_examples_per_bin}')
    if num_train_samples_per_bin + num_test_samples_per_bin > min_examples_per_bin:
        raise ValueError('Insufficient samples per bin')
    number_of_recent_samples = num_train_bins*num_train_samples_per_bin

    # output folder
    now = datetime.now(pytz.utc)
    f_out_folder = os.path.join('data', 'concept_drift', f'concept_drift_v{dataset_version}')
    if os.path.isdir(f_out_folder):
        shutil.rmtree(f_out_folder)
    os.makedirs(f_out_folder)

    # create datasets
    logger.info('Creating datasets...')
    cols = ['id', 'text', 'label']
    repeat_datasets = []
    for k in range(repeats):
        datasets = []
        # populate base sample
        df_prev = pd.DataFrame()
        for v, centroid in enumerate(centroid_days):
            start_date = centroid - interval/2
            end_date = centroid + interval/2
            centroid_str = centroid.strftime('%Y-%m-%d')
            # select data
            df = df_all[(df_all.created_at > start_date) & (df_all.created_at <= end_date)].copy()
            # sample
            df_original_len = len(df)
            if v < num_train_bins:
                # generate base training sample
                num_samples_required = num_train_samples_per_bin
                if df_original_len < num_samples_required:
                    raise Exception(f'Bin {centroid_str} has only {df_original_len:,} samples, but {num_samples_required:,} samples are required!')
                df = df.sample(num_samples_required)
                logger.info(f'Populating base training data for bin {v} with {len(df):,} samples...')
                df_prev = pd.concat([df_prev, df])
                continue
            if v == num_train_bins:
                logger.info(f'Populated base training data with {len(df_prev):,} samples')
                assert len(df_prev) == number_of_recent_samples
            # generate normal train/test bin
            num_samples_required = num_train_samples_per_bin + num_test_samples_per_bin
            if df_original_len < num_samples_required:
                raise Exception(f'Bin {centroid_str} has only {df_original_len:,} samples, but {num_samples_required:,} samples are required!')
            df = df.sample(num_samples_required)
            # create folder
            output_folder = os.path.join(f_out_folder, centroid_str, str(k))
            if not os.path.isdir(output_folder):
                os.makedirs(output_folder)
            # write file containing all samples in bin
            f_path_all = os.path.join(output_folder, 'all.csv')
            df[cols].to_csv(f_path_all, index=False)
            # write test data
            logger.info(f'Dates {start_date}-{end_date}: {len(df):,} labels... (originally {df_original_len:,})')
            df_train, df_test = np.split(df, [num_train_samples_per_bin])
            logger.info(f'... test set: {len(df_test):,}')
            f_path_test = os.path.join(output_folder, 'test.csv')
            df_test[cols].to_csv(f_path_test, index=False)
            assert len(df_test) == num_test_samples_per_bin
            assert len(df_train) == num_train_samples_per_bin
            # include prev. data to train
            df_prev = pd.concat([df_prev, df_train])
            if number_of_recent_samples is not None:
                if len(df_prev) < number_of_recent_samples:
                    raise Exception(f'Too little training data for centroid {centroid_str} ({number_of_recent_samples:,} required, but only found {len(df_prev):,} samples)')
                # exclude old previous data in order to always have `number_of_recent_samples` samples in df_prev
                df_prev = df_prev.sort_values('created_at')
                df_prev = df_prev.iloc[-number_of_recent_samples:]
            # write train data
            f_path_train = os.path.join(output_folder, 'train.csv')
            df_prev = df_prev.sample(frac=1)
            df_prev[cols].to_csv(f_path_train, index=False)
            logger.info(f'... train set: {len(df_prev):,}')
            # add to datasets
            datasets.append({
                'start_date': start_date,
                'end_date': end_date,
                'centroid': centroid,
                'centroid_str': centroid_str,
                'f_path_train': f_path_train,
                'f_path_test': f_path_test,
                'repeat': k
                })
            if include_test_samples_to_pool:
                # add test back to previous
                df_prev = pd.concat([df_prev, df_test])
        logger.info(f'... compiled {len(datasets):,} datasets for repeat {k}...')
        repeat_datasets.append(datasets)

    # generate runs
    logger.info('Compiling runs...')
    runs = []
    for k, datasets in enumerate(repeat_datasets):
        for i, dataset in enumerate(datasets):
            test_paths = []
            centroid_days_test = []
            for j in range(i, len(datasets)):
                # compile all test datasets for this training run
                test_paths.append(datasets[j]['f_path_test'])
                centroid_days_test.append(datasets[j]['centroid'])
            runs.append(dict(
                args=(f'concept_drift_v{dataset_version}-{dataset["centroid_str"]}-test-{i}-{k}', dataset['f_path_train'], test_paths),
                other=dict(
                    centroid_day_train=dataset['centroid'],
                    centroid_days_test=centroid_days_test,
                    repeat=k
                    )
                ))

    # check if run
    yes_no = input(f'Run {len(runs):,} runs? (y/n)\n>>> ')
    if yes_no != 'y':
        sys.exit()

    # run in parallel
    logger.info('Run in parallel...')
    num_cores = max(multiprocessing.cpu_count() - 1, 1)
    parallel = joblib.Parallel(n_jobs=num_cores)
    train_test_delayed = joblib.delayed(train_test)
    res = parallel(train_test_delayed(*run['args'], run['other']) for run in tqdm(runs))
    df_results = pd.concat(pd.DataFrame(r) for r in res)
    df_results = df_results.reset_index(drop=True)

    # save results
    f_path_results = os.path.join(f_out_folder, 'results.parquet')
    logger.info(f'Writing results to {f_path_results}...')
    df_results.to_parquet(f_path_results)

    # save config
    config_path = os.path.join(f_out_folder, 'config.json')
    logger.info(f'Writing config to {config_path}...')
    config = dict(
        start_date_exp=start_date_exp.strftime('%Y-%m-%d'),
        end_date_exp=end_date_exp.strftime('%Y-%m-%d'),
        interval=interval.days,
        step=step,
        repeats=repeats,
        number_of_recent_samples=number_of_recent_samples,
        num_train_samples_per_bin=num_train_samples_per_bin,
        num_test_samples_per_bin=num_test_samples_per_bin,
        num_train_bins=num_train_bins,
        include_test_samples_to_pool=include_test_samples_to_pool,
        dataset_folder=f'concept_drift_v{dataset_version}',
        seed=seed,
        **params
    )
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)


if __name__ == "__main__":
    run_concept_drift()
