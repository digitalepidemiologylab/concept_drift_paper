import pandas as pd
import os
import sys
import json
from utils.helpers import save_fig, paper_plot, cached
import matplotlib.pyplot as plt
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from tqdm import tqdm
import logging
from transformers import pipeline
import numpy as np
import matplotlib.transforms
import matplotlib.ticker
import shutil


logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)-5.5s] [%(name)-12.12s]: %(message)s')
logger = logging.getLogger(__name__)


# dataset to compute things for. First run experiments for fig2 and reference same version number here
dataset_version = 1
DATA_DIR = 'data'

# pick datasets
# train/test = what model sees (official fig3)
# all = only what was added in bin (figure in SI)
datasets = ('train', 'test')
# datasets = ('all',)

def compute_embeddings(recompute=False):
    # create output folder
    datasets_str = '_'.join(list(datasets))
    output_folder = os.path.join('data', 'text_similarity', f'model_drift_v{dataset_version}_{datasets_str}')
    if os.path.isdir(output_folder) and not recompute:
        logger.info(f'Embeddings for dataset version {dataset_version} already computed.')
        return
    if os.path.isdir(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    # find relevant files
    dataset_folder = os.path.join('data', 'model_drift', f'model_drift_v{dataset_version}')
    f_names = []
    for ds in list(datasets):
        fn = glob.glob(os.path.join(dataset_folder, '*', '0',  f'{ds}.csv'))
        f_names.extend(fn)

    # compute embeddings
    pipe = pipeline(task='feature-extraction', model='bert-large-uncased', framework='pt', device=0)
    for f_name in tqdm(f_names):
        date_str = f_name.split('/')[-3]
        type_str = f_name.split('/')[-1].split('.')[0]
        date = datetime.strptime(date_str, '%Y-%m-%d')
        df = pd.read_csv(f_name)

        batch_size = 8
        logger.info(f'Computing embeddings on {len(df):,} texts...')
        all_embeddings = []
        for text_chunk in tqdm(np.array_split(df.text, len(df) // batch_size), unit='batch'):
            embeddings = pipe(text_chunk.tolist())
            embeddings = np.array(embeddings)
            all_embeddings.append(embeddings[:, 0, :])
        all_embeddings = np.concatenate(all_embeddings)

        # output_folder
        f_out_folder = os.path.join(output_folder, date_str, type_str)
        os.makedirs(f_out_folder)

        # write embeddings
        f_out = os.path.join(f_out_folder, 'embeddings.npy')
        logger.info(f'Writing embeddings to {f_out}...')
        np.save(f_out, all_embeddings)

        # write df
        f_out = os.path.join(f_out_folder, 'text.csv')
        logger.info(f'Writing text data to {f_out}...')
        df.to_csv(f_out, index=False)

        logger.info('... done')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def compute_fig3d_similarity_scores():
    # compute embeddings
    compute_embeddings(recompute=False)

    # load data
    datasets_str = '_'.join(list(datasets))
    output_folder = os.path.join('data', 'text_similarity', f'model_drift_v{dataset_version}_{datasets_str}')
    f_names = glob.glob(os.path.join(output_folder, '**', '**', 'embeddings.npy'))
    data = {}
    for f_name in f_names:
        emb = np.load(f_name)
        emb = [x for x in emb]
        centroid_day_str = f_name.split('/')[-3]
        f_in = os.path.join(os.path.dirname(f_name), 'text.csv')
        df = pd.read_csv(f_in)
        labels = df.label.tolist()
        assert len(df) == len(emb)
        if centroid_day_str in data:
            data[centroid_day_str]['embeddings'].extend(emb)
            data[centroid_day_str]['labels'].extend(labels)
        else:
            data[centroid_day_str] = {'embeddings': emb, 'labels': labels}
    data = dict(sorted(data.items(), key=lambda x: x[0]))
    
    # compute cosine similarity
    centroid_days = list(data.keys())
    num_bins = len(centroid_days)
    sim_matrix = np.zeros((num_bins, num_bins))
    for i in np.arange(num_bins):
        for j in np.arange(i, num_bins):
            data_i = pd.DataFrame(data[centroid_days[i]])
            data_j = pd.DataFrame(data[centroid_days[j]])
            emb_i = np.stack(data_i['embeddings'])
            emb_j = np.stack(data_j['embeddings'])
            assert emb_i.shape == emb_j.shape
            sim_matrix[i][j] = cosine_similarity(emb_i.mean(axis=0), emb_j.mean(axis=0))
    sim_matrix = pd.DataFrame(values, index=centroid_days, columns=centroid_days)
    return sim_matrix

def compute_fig3ac_label_imbalance_embedding_variance(recompute=False):
    # check if already computed
    datasets_str = '_'.join(list(datasets))
    output_folder = os.path.join('data', 'text_similarity', f'model_drift_v{dataset_version}_{datasets_str}')
    f_out = os.path.join(output_folder, 'embedding_variance.parquet')
    if os.path.isfile(f_out) and not recompute:
        return pd.read_parquet(f_out)

    # compute embeddings
    compute_embeddings(recompute=False)

    # load data
    f_names = glob.glob(os.path.join(output_folder, '**', '**',  'embeddings.npy'))
    data = {}
    for f_name in f_names:
        emb = np.load(f_name)
        emb = [x for x in emb]
        centroid_day_str = f_name.split('/')[-3]
        f_in = os.path.join(os.path.dirname(f_name), 'text.csv')
        df = pd.read_csv(f_in)
        assert len(df) == len(emb)
        if centroid_day_str in data:
            data[centroid_day_str]['embeddings'].extend(emb)
            data[centroid_day_str]['labels'].extend(df.label.tolist())
        else:
            data[centroid_day_str] = {'embeddings': emb, 'labels': df.label.tolist()}

    # compute embedding variance
    df = []
    for date, row in data.items():
        _df = pd.DataFrame(row)
        for label, grp in _df.groupby('labels'):
            df.append({'date': date, 'variance': np.stack(grp.embeddings.values).var(), 'label': label, 'size': len(grp)})
        df.append({'date': date, 'variance': np.stack(_df.embeddings.values).var(), 'label': 'all', 'size': len(_df)})
    df = pd.DataFrame(df)
    df = df.sort_values('date')

    # store similarity scores
    logger.info(f'Writing variance scores to {f_out}')
    df.to_parquet(f_out)
    return df

def compute_fleiss_kappa(df):
    """Compute Fleiss' Kappa for a question."""
    # compute table (Nxk), N being the number of tweets, k being the number of possible answers for this question
    N = len(df['id'].unique())
    k = len(df['answer_tag'].unique())
    table = pd.DataFrame(0, index=df['id'].unique(), columns=df['answer_tag'].unique())
    for i, row in df.iterrows():
        table.loc[row.id][row.answer_tag] += 1
    # number of annotations given by tweet
    num_annotations = table.sum(axis=1)
    # num agreements (how many annotator-annotator pairs are in agreement): sum(n_ij * (n_ij - 1)) = sum(n_ij ** 2) - sum(n_ij)
    num_agreements = table.pow(2).sum(axis=1) - num_annotations
    # num possible agreements (how many possible annotator-annotator pairs there are)
    num_possible_agreements = num_annotations * (num_annotations - 1)
    # agreement
    table['agreement'] = num_agreements/num_possible_agreements
    # compute chance agreement
    num_annotations = table.sum().sum()
    answer_fractions = table.sum(axis=0)/num_annotations
    chance_agreeement = answer_fractions.pow(2).sum()
    # Fleiss' Kappa: (Mean observed agreement - chance agreement)/(1 - chance agreement)
    fleiss_kappa = (table['agreement'].mean() - chance_agreeement)/(1 - chance_agreeement)
    return fleiss_kappa

def get_annotations():
    df = pd.read_csv(os.path.join(DATA_DIR, 'annotations_raw.csv'), parse_dates=['created_at', 'annotation_created_at'])
    return df

def get_datasets(datasets=('all',)):
    datasets_str = '_'.join(list(datasets))
    dataset_folder = os.path.join('data', 'concept_drift', f'concept_drift_v{dataset_version}')
    f_names = []
    df = pd.DataFrame()
    for ds in list(datasets):
        fn = glob.glob(os.path.join(dataset_folder, '*', '0',  f'{ds}.csv'))
        for f_name in tqdm(fn):
            date_str = f_name.split('/')[-3]
            type_str = f_name.split('/')[-1].split('.')[0]
            date = datetime.strptime(date_str, '%Y-%m-%d')
            _df = pd.read_csv(f_name, dtype={'id': str})
            _df['dataset'] = ds
            _df['centroid_date'] = date
            _df['centroid_date_str'] = date_str
            df = pd.concat([df, _df])
    return df

def compute_fig3b_annotator_agreement():
    df = get_annotations()
    df_ds = get_datasets(datasets=datasets)
    data = []
    for centroid_day_str, grp in df_ds.groupby('centroid_date_str'):
        _df = df[df.id.isin(grp.id)]
        fleiss_kappa = compute_fleiss_kappa(_df)
        fleiss_kappa_sub = {}
        for mode, grp in _df.groupby('mode'):
            fleiss_kappa_sub['fleiss_kappa_mode_' + mode] = compute_fleiss_kappa(grp)
        data.append({
            'centroid_day': centroid_day_str,
            'fleiss_kappa': fleiss_kappa,
            'num_annotations': len(_df),
            **dict(_df['mode'].value_counts()),
            **fleiss_kappa_sub
            })
    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    compute_fig3d_similarity_scores()
    compute_fig3ac_label_imbalance_embedding_variance()
    compute_fig3b_annotator_agreement()
