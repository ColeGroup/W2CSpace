# W2CSpase for interpretable language modeling

Official code for "Constructing Word-Context-Coupled Space Aligned with Associative Knowledge Relations for Interpretable Language Modeling", Findings in ACL2023.

## You can find our paper on [Link](https://aclanthology.org/2023.findings-acl.532/). Fanyu Wang [personal page](https://fanyuuwang.github.io/)

Cite as: Fanyu Wang and Zhenping Xie. 2023. Constructing Word-Context-Coupled Space Aligned with Associative Knowledge Relations for Interpretable Language Modeling. In Findings of the Association for Computational Linguistics: ACL 2023, pages 8414–8427, Toronto, Canada. Association for Computational Linguistics.

## Lauchment

### Our default config are stored in ./config/config.py
You can customize the settings in config.py, where "mode" refer to sentiment classification and spelling correction tasks.

### 0. initial_akn.py
You can use your personal dataset for AKN initialization or you can find our AKN weight in ./akn/akn_download.txt.

### 1. model_initial.py
Initialization py file for finetuning BERT model and training for mapping network.

### 2. context_cluster.py
Clustering py file for abstraction of context-level semantics.

### 3. senti_classify.py / correction.py
Tasks completion py files for sentiment classification and spelling correction tasks.

## Datasets

## CHNST
CHNST dataset for sentiment classification task. We do preprocessing operation.

## SIGHAN15
SIGHAN15 dataset for spelling correction task. Evaluation only.

## Weibo
Weibo dataset for sentiment classification task. We do preprocessing operation.

## trainset_download.txt
We upload our preprocessed training dataset on Google Drive. The trainset are constructed based on HyBird and SIGHAN15-Trainset.
