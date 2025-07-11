# MageSQL: Enhancing In-context Learning for Text-to-SQL Applications with Large Language Models  

**Official Implementation — VLDB@AIDB 2025**

---

## Overview

**MageSQL** is a lightweight, secure and practical text-to-SQL pipeline developed by **Megagon Labs**. It takes a natural language question and a table schema as input, and outputs an executable SQL query with high accuracy and low latency.

**Key Features:**  
✅ **Flexible Demonstration Selection:** Offers a rich pool of demonstration selection methods — from simple Jaccard similarity to advanced SQL graph similarity using contrastive learning.

✅ **Low latency & Cost:** While CoT prompting and voting are common in text-to-SQL research, MageSQL needs only 1–3 prompts to generate the final SQL query, ensuring low latency and reduced cost.

✅ **Plug-and-play:** Requires only the question, table schema, and optional question evidence — no heavy database content or manually drafted database descriptions needed.

✅ **No execution dependency:** Unlike pipelines that rely on repeated SQL execution for correction (e.g., CHESS), MageSQL achieves high performance without execution-based fixes, making it robust for production on large databases.

✅ **Data security for LLM:** Database SQLite files are used solely to generate the table schema in SQL clause format for LLM prompting. The actual data content is never sent to the LLM, which guarantees that sensitive information remains private.

## Repository Structure

This section provides a overview of the repository's structure.

```
└── dataset_classes                         # Classes for datasets
    ├── base_dataset.py                     # Abstract base dataset for NL2SQL
    └── <dataset_class>.py                  # Implementation class of dataset
└── demonstration_selector                  # Classes for demonstration selectors
    ├── base_demonstration_selector.py      # Abstract base demonstration selector
    └── <demonstration selector>.py         # Implementation class of demonstration selector
└── utils                                   # Utility functions
    ├── dataset_utils.py                    # Utility functions for I/O of dataset files 
    ├── openai_utils.py                     # Utility functions to call OpenAI API 
    ├── sql_utils.py                        # Utility functions to execute SQL excuations  
    ├── template_utils.py                   # Utility functions to get and fill prompt template
    ├── correction_utils.py                 # Utility functions about error correction
    └── other utils
└── gnn_contrastive_learning                # Train Struct-graph demonstration selector by graph contrastive learning
    ├── gcl_dataset.py                      # Dataset class
    ├── gcl_train.py                        # Model training
    ├── model.py                            # Graph encoder model and losses
    ├── negative_sampling.py                # Generate negative samples for anchor SQL graph
    ├── sql_graph_augmenter.py              # Operators to generate postive graphs for anchor SQL graph
    ├── sql_to_graph.py                     # Build DAG graph from a SQL
├── construct_prompt.py                     # Generate prompts by demonstraion selection from a databset
├── prompt_llm.py                           # Send prompts to LLM and get responses
├── error_correction.py                     # Prompt based error correction
├── rule_based_error_correction.py          # Functions for rule based error correction
├── example_pipeline_run.sh                 # Example script to run the repo
├── requirements.txt                        # Necessary Python packages
└── README.md                               # The file you are reading now.
```

## How to run the scripts

### Environment setup

Install packages

```python_env
conda create -n aidb_submission python=3.10
conda activate aidb_submission
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Notes: we test the enviroment on a Linux machine with GPU (Driver Version: 535.54.03, CUDA Version: 12.2), please update the packages accordingly according to your specifc enviroment.

Download punkt toenizer

```sh
python -m nltk.downloader punkt
python -m nltk.downloader punkt_tab
```

Set the Python path with the root folder of this repo

```sh
export PYTHONPATH=<path_of_this_repo>
```

If you are already in the root folder of this repo, you can use the following command to set the Python path

```sh
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

### Download Dataset

You need to download the dataset from external repos, for example [Spider](https://yale-lily.github.io/spider), [BIRD](https://bird-bench.github.io/). The default paths of datasets are `<repo_path>/datasets/<dataset_name>` directory but it's also fine if you put the datasets on other locations and specify the path in input arguments.

Example steps to download the Spider dataset

```sh
mkdir datasets/
cd datasets/
gdown 1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J
unzip spider_data.zip
mv spider_data/ spider/
```

For Spider dataset, you will need to merge the `train_spider.json` and `train_others.json` into a `train_spider_and_others.json` file, and store in the same folder. We provide an example Python script for your reference in `combine_json.py`.

### Run text2sql Pipeline

We provided a sample example script `example_pipeline_run.sh` to run the whole cycle of the text2sql pipeline. Please update the placeholder and run it at the root dir of this repo.

### Evaluations

Please use the evaluation scripts in repos ([test-suite-sql-eval](https://github.com/taoyds/test-suite-sql-eval.git), [spider](https://github.com/taoyds/spider.git)), [BIRD](https://bird-bench.github.io/) to evaluate the generated SQLs in the corresponding datasets.

## Citation

If you use this code or find **MageSQL** helpful for your research, we kindly ask that you cite our paper:

```bibtex
@article{shen2025magesql,
  title={MageSQL: Enhancing In-context Learning for Text-to-SQL Applications with Large Language Models},
  author={Shen, Chen and Wang, Jin and Rahman, Sajjadur and Kandogan, Eser},
  journal={arXiv preprint arXiv:2504.02055},
  year={2025}
}
```

## Disclosure

Embedded in, or bundled with, this product are open source software (OSS) components, datasets and other third party components identified below. The license terms respectively governing the datasets and third-party components continue to govern those portions, and you agree to those license terms, which, when applicable, specifically limit any distribution. You may receive a copy of, distribute and/or modify any open source code for the OSS component under the terms of their respective licenses, which may be CC license and Apache 2.0 license. In the event of conflicts between Megagon Labs, Inc., license conditions and the Open Source Software license conditions, the Open Source Software conditions shall prevail with respect to the Open Source Software portions of the software. You agree not to, and are not permitted to, distribute actual datasets used with the OSS components listed below. You agree and are limited to distribute only links to datasets from known sources by listing them in the datasets overview table below. You are permitted to distribute derived datasets of data sets from known sources by including links to original dataset source in the datasets overview table below. You agree that any right to modify datasets originating from parties other than Megagon Labs, Inc. are governed by the respective third party’s license conditions. All OSS components and datasets are distributed WITHOUT ANY WARRANTY, without even implied warranty such as for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE, and without any liability to or claim against any Megagon Labs, Inc. entity other than as explicitly documented in this README document. You agree to cease using any part of the provided materials if you do not agree with the terms or the lack of any warranty herein. While Megagon Labs, Inc., makes commercially reasonable efforts to ensure that citations in this document are complete and accurate, errors may occur. If you see any error or omission, please help us improve this document by sending information to <contact_oss@megagon.ai>.
