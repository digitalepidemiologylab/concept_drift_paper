This repository contains data &amp; code necessary to reproduce the paper "How machine learning concept drift can negatively affect social media analysis"

# Install
Create a new environment (e.g. using conda) with Python 3.8 installed.

Clone the repo and install the dependencies:
```bash
pip install -r requirements.txt
```

# Data
All data can be found under `./data/`.

## `annotations_raw.csv`
Raw annotations (before consensus)
| Column | Description                                    |
|---------------|--------------------------------------------------------|
| `id`         | Tweet ID                                        |
| `answer_tag`         | label                                        |
| `created_at`         | Tweet creation at                                        |
| `annotation_created_at`         | Annotation time                                       |
| `annotator_id`         | worker/user ID                                       |

## `annotations_merged.csv`
Annotations after consensus
| Column | Description                                    |
|---------------|--------------------------------------------------------|
| `id`         | Tweet ID                                        |
| `label`         | label                                        |
| `created_at`         | Tweet creation at                                        |
| `annotation_created_at`         | Annotation time                                       |

## Figure data
All data to reproduce the figures are named as `fig_{n}_{description}.csv` and can also be found in the `data` folder.

# Code
There are two scripts which include most relevant code to reproduce the anlaysis:
* `run_fig2_experiments.py`: Runs the drift experiments for FastText
* `run_fig3_experiments.py`: Runs code to compute properties in Figure 3.

If you have any more specific requests, please contact the corresponding author.

