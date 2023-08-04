# W2CSpase for interpretable language modeling

Official code for "Constructing Word-Context-Coupled Space Aligned with Associative Knowledge Relations for Interpretable Language Modeling", Findings in ACL2023.

## You can find our [paper](https://aclanthology.org/2023.findings-acl.532/). Fanyu Wang [personal page](https://fanyuuwang.github.io/)

Cite as:

@inproceedings{wang-xie-2023-constructing,
    title = "Constructing Word-Context-Coupled Space Aligned with Associative Knowledge Relations for Interpretable Language Modeling",
    author = "Wang, Fanyu  and
      Xie, Zhenping",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.532",
    pages = "8414--8427",
    abstract = "As the foundation of current natural language processing methods, pre-trained language model has achieved excellent performance. However, the black-box structure of the deep neural network in pre-trained language models seriously limits the interpretability of the language modeling process. After revisiting the coupled requirement of deep neural representation and semantics logic of language modeling, a Word-Context-Coupled Space (W2CSpace) is proposed by introducing the alignment processing between uninterpretable neural representation and interpretable statistical logic. Moreover, a clustering process is also designed to connect the word- and context-level semantics. Specifically, an associative knowledge network (AKN), considered interpretable statistical logic, is introduced in the alignment process for word-level semantics. Furthermore, the context-relative distance is employed as the semantic feature for the downstream classifier, which is greatly different from the current uninterpretable semantic representations of pre-trained models. Our experiments for performance evaluation and interpretable analysis are executed on several types of datasets, including SIGHAN, Weibo, and ChnSenti. Wherein a novel evaluation strategy for the interpretability of machine learning models is first proposed. According to the experimental results, our language model can achieve better performance and highly credible interpretable ability compared to related state-of-the-art methods.",
}

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
