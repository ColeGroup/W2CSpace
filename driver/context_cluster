import os
import sys

Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
# print(os.path.split(rootPath)[0])
sys.path.append(os.path.split(rootPath)[0])
import argparse
import pandas as pd
import tqdm
from datasets import Dataset
import torch.nn as nn
from config.config import config
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from transformers.models.bert.modeling_bert import BertConfig
from fast_pytorch_kmeans import KMeans


class Trainer:
    def __init__(self, config):
        self.config = config
        self.pretrain_config = BertConfig.from_pretrained(self.config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.bert_path)
        self.model = torch.load(self.config.normal_encoder_path).to(self.config.device)
        self.graph = torch.load(self.config.mapper_path).to(self.config.device)
        self.akn = torch.load(self.config.akn_path).to(self.config.device)
        self.device = self.config.device
        self.log_freq = 100

    def target_process(self, tgt_ids, tgt_mask, tgt_label):
        act_tgt_ids = tgt_ids * (tgt_label == 0) * tgt_mask
        act_tgt_mask = tgt_mask * (tgt_label == 0) * tgt_mask

        return act_tgt_ids, act_tgt_mask

    def get_var(self, a, b):
        func = nn.L1Loss()
        var = func(a, b)

        return var

    def shift(self, matrix, shift_value):
        new_matrix = torch.zeros(size=matrix.shape, device=matrix.device, dtype=matrix.dtype, requires_grad=False)
        new_matrix[shift_value:] = matrix[:-shift_value]
        new_matrix[:shift_value] = matrix[-shift_value:]

        return new_matrix

    def save_sent_kozmo(self, kozmo):
        torch.save(kozmo, self.config.sent_kozmo_path)

    def save_word_kozmo(self, kozmo):
        torch.save(kozmo, self.config.word_kozmo_path)

    def get_vocab_kozmo(self, train_data):
        self.graph.eval()
        self.model.eval()
        data_loader = tqdm.tqdm(enumerate(train_data),
                                desc="EP_%s" % "construct",
                                total=len(train_data),
                                bar_format="{l_bar}{r_bar}")
        current_sema = torch.zeros(size=[1, self.config.kozmo_size], device=self.config.device,
                                   requires_grad=False)
        total_sema = torch.zeros(size=[self.config.semaspace, self.config.kozmo_size], device=self.config.device,
                                 requires_grad=False)
        sema_count = torch.zeros(size=[self.config.semaspace], device=self.config.device, dtype=torch.short,
                                 requires_grad=False)
        kmeans = KMeans(n_clusters=self.config.semaspace, mode='cosine', verbose=1)
        for step, data in data_loader:
            input_ids = data["input_text"]
            input_mask = (input_ids > 0).int()
            encoder_out, _ = self.model(input_ids, input_mask)
            temp_semantic_kozmo, _ = self.graph.mapping(encoder_out.detach())
            sent_sema = self.calculate(temp_semantic_kozmo, input_mask, div=False, dim=1)
            current_sema = torch.cat((current_sema, sent_sema), dim=0)
            if step == 0:
                current_sema = current_sema[1:]
        labels = kmeans.fit_predict(current_sema)
        for index, _ in enumerate(current_sema):
            total_sema[labels[index]] += current_sema[index]
            sema_count[labels[index]] += 1
        self.save_sent_kozmo(torch.div(total_sema, (sema_count + 1e-8).unsqueeze(-1)))

    def calculate(self, semantic_kozmo, input_mask, div=False, dim=0):
        semantic_kozmo = semantic_kozmo.detach()
        input_mask = input_mask.detach()
        if div:
            cosine = torch.sum(torch.mul(input_mask.unsqueeze(-1), torch.div(semantic_kozmo,
                                                                             (torch.sqrt(
                                                                                 torch.sum(torch.pow(semantic_kozmo, 2),
                                                                                           dim=-1)) + 1e-8).unsqueeze(
                                                                                 -1))) / semantic_kozmo.shape[0],
                               dim=dim)
        else:
            cosine = torch.div(torch.sum(torch.mul(input_mask.unsqueeze(-1), torch.div(semantic_kozmo,
                                                                                       (torch.sqrt(torch.sum(
                                                                                           torch.pow(semantic_kozmo, 2),
                                                                                           dim=-1)) + 1e-8).unsqueeze(
                                                                                           -1))), dim=dim),
                               torch.sum(input_mask.unsqueeze(-1), dim=dim))
        # cosine = torch.zeros(size=[self.config.kozmo_size], device=self.config.device)
        # for k_num, k in enumerate(semantic_kozmo[1:]):
        #     cosine += k / (torch.sqrt(torch.sum(torch.pow(k, 2))) + 1e-8)

        return cosine.detach()

class ClassifyDataset(Dataset):
    def __init__(self, dataset, max_length, device):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.device = torch.device(device)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
        self.max_size = max_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        label = item['label']
        input_text = self.tokenizer(item['text'], padding='max_length', max_length=512)["input_ids"]

        output = {
            'label': torch.tensor(label, device=self.device),
            'input_text': torch.tensor(input_text, device=self.device),
        }
        # if torch.sum(output["input_text"] > 0) != torch.sum(output["target_text"] > 0):
        #     print("!!!!")
        return {key: value for key, value in output.items()}

class CorrectDataset(Dataset):
    def __init__(self, dataset, max_length, device):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
        self.max_size = max_length

    def __len__(self):
        return self.data_size

    def __getitem__(self, item):
        item = self.dataset.iloc[item]
        input_text = self.tokenizer(item['source'], padding='max_length', max_length=self.max_size)["input_ids"]
        target_text = self.tokenizer(item['target'], padding='max_length', max_length=self.max_size)["input_ids"]

        output = {
            'input_text': torch.tensor(input_text, device=self.device),
            'target_text': torch.tensor(target_text, device=self.device),
        }
        return {key: value for key, value in output.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--dataset", type=str, default=rootPath + "/data",
                        help="train dataset")
    args = parser.parse_args()
    torch.manual_seed(1)
    config = config()
    # train = pd.read_csv(args.dataset + "/Weibo/train2.csv")
    train = pd.read_csv(args.dataset + "/train_dataset.csv")
    if config.mode == "classify_out2":
        train = ClassifyDataset(train, max_length=config.max_length, device=config.device)
    else:
        train = CorrectDataset(train, max_length=config.max_length, device=config.device)
    train = DataLoader(train, batch_size=8, num_workers=0)
    trainer = Trainer(config=config)
    best_f1 = 0

    trainer.get_vocab_kozmo(train)

    print("Complete!")
    print("Actual SemaSpace = " + str(config.semaspace))
