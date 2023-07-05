
import os
import sys

Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(os.path.split(rootPath)[0])
import argparse
import pandas as pd
import tqdm
from datasets import Dataset
import torch.nn as nn
from config.config import config
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW
from transformers.models.bert.modeling_bert import BertConfig
from model.KozmoClassifier import GraphEncoder, BertOnlyMLMHead
from model.SingleGraph import SingleGraph


class Trainer:
    def __init__(self, config, epoch, lr=(2e-5, 1e-5), gama=0.8, weight_decay=0.01,
                 save_path="",
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        self.normal_num_warmup = epoch[0] / 10
        self.normal_num_training = epoch[0]
        self.mapping_num_warmup = epoch[1] / 10
        self.mapping_num_training = epoch[1]
        self.max_grad_norm = max_grad_norm
        self.config = config
        self.pretrain_config = BertConfig.from_pretrained(self.config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(
            self.config.bert_path, local_files_only=True)
        self.graph = SingleGraph(self.config).to(self.config.device)
        self.model = GraphEncoder(self.config).to(self.config.device)
        self.classifier = BertOnlyMLMHead(self.pretrain_config).to(self.config.device)
        self.gama = gama
        self.save_path = save_path
        self.akn = torch.load(Base_DIR + "/akn/akn.pt").to(self.config.device)
        # mainframe optimizing parameter setting
        self.lm_param = self.get_param(self.model, weight_decay)
        self.cl_param = self.get_param(self.classifier, weight_decay)
        self.lm_optimizer = AdamW(params=self.lm_param + self.cl_param, lr=lr[0], correct_bias=True)
        self.lm_scheduler = get_linear_schedule_with_warmup(self.lm_optimizer,
                                                            num_warmup_steps=self.normal_num_warmup,
                                                            num_training_steps=self.normal_num_training)
        self.graph_param = self.get_param(self.graph, weight_decay)
        self.graph_optimizer = torch.optim.Adam(params=self.graph_param, lr=lr[1])
        self.graph_scheduler = get_linear_schedule_with_warmup(self.graph_optimizer,
                                                               num_warmup_steps=self.mapping_num_warmup,
                                                               num_training_steps=self.mapping_num_training)
        self.device = self.config.device
        self.log_freq = 100
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.vocab_kozmo = torch.zeros(size=[self.config.vocab_size, self.config.hidden_size],
                                       device=self.config.device)
        self.vocab_count = torch.zeros(size=[self.config.vocab_size, 1], device=self.config.device)

    def get_param(self, model, weight_decay):
        no_decay = ['bias', 'LayerNorm.Weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters

    def loc_loss(self, input_ids, input_mask, dis_matrix, act_dis_tags):
        local_akn = self.get_context_akn(input_ids, input_mask)
        mse_loss_fun = torch.nn.MSELoss()
        avg_ids = torch.zeros(size=[act_dis_tags.shape[0]], device=self.config.device)
        avg_akn = torch.zeros(size=[act_dis_tags.shape[0]], device=self.config.device)
        for batch_num in range(act_dis_tags.shape[0]):
            avg_ids[batch_num] = torch.div(torch.sum(dis_matrix[batch_num][act_dis_tags[batch_num] == 1]),
                                           (torch.sum(act_dis_tags[batch_num] == 1) + 1e-8))
            avg_akn[batch_num] = torch.div(torch.sum(local_akn[batch_num][act_dis_tags[batch_num] == 1]),
                                           (torch.sum(act_dis_tags[batch_num] == 1) + 1e-8))
        loss = mse_loss_fun(
            torch.div(dis_matrix, (avg_ids.unsqueeze(-1).unsqueeze(-1) + 1e-8))[act_dis_tags == 1].unsqueeze(0),
            torch.div(local_akn, (avg_akn.unsqueeze(-1).unsqueeze(-1) + 1e-8))[act_dis_tags == 1].unsqueeze(0))
        return loss

    def reconstruction_loss(self, encoder_out, decoder_out):
        loss_fun = nn.L1Loss()
        loss = loss_fun(encoder_out, decoder_out)

        return loss

    def get_vocab_kozmo(self, train_data):
        data_loader = tqdm.tqdm(enumerate(train_data),
                                desc="EP_%s:" % ("construct"),
                                total=len(train_data),
                                bar_format="{l_bar}{r_bar}")
        for step, data in data_loader:
            input_ids = data["input_text"]
            target_ids = data["target_text"]
            input_mask = (input_ids > 0).int()
            wrong_label = torch.zeros(size=input_ids.shape, device=target_ids.device, dtype=target_ids.dtype)
            wrong_label[input_ids != target_ids] = target_ids[input_ids != target_ids]
            encoder_out = self.model(input_ids, input_mask)
            for index, ids in enumerate(input_ids[input_mask == 1]):
                self.vocab_kozmo[ids] += encoder_out.detach()[input_mask == 1][index]
                self.vocab_count[ids] += 1
        self.save_vocabs()
        print("updated")

    def save_vocabs(self):
        torch.save(torch.div(self.vocab_kozmo, self.vocab_count + 1e-8), self.config.vocab_path)

    def save_classifier(self):
        torch.save(self.classifier, self.config.normal_classifier_path)

    def save_model(self):
        torch.save(self.model, self.config.normal_encoder_path)

    def save_graph(self):
        torch.save(self.graph, self.config.mapper_path)

    def mapping_trainer(self, train_data, epoch, train=True):
        str_code = "train" if train else "val"
        self.model.to(self.device)
        self.model.eval()
        data_loader = tqdm.tqdm(enumerate(train_data),
                                desc="EP_%s:%d" % (str_code, epoch),
                                total=len(train_data),
                                bar_format="{l_bar}{r_bar}")
        avg_loss_mapping = 0
        iter_loss_mapping = 0
        iter_loss_recons = 0
        current_iter = 1
        vocab_semantic = torch.load(self.config.vocab_path).to(self.config.device)
        for step, data in data_loader:
            # 1. Data Preprocess
            input_ids = data["input_text"]
            target_ids = data["target_text"]
            input_mask = (input_ids > 0).int()
            wrong_label = torch.zeros(size=input_ids.shape, device=target_ids.device, dtype=target_ids.dtype)
            wrong_label[input_ids != target_ids] = target_ids[input_ids != target_ids]
            encoder_out, _ = self.model(input_ids, input_mask)
            locations, dis_matrix, act_dis_tags, autoencoder_out = self.graph(
                encoder_out.detach(),
                input_mask)
            mp_loss = self.loc_loss(input_ids, input_mask, dis_matrix, act_dis_tags)
            rc_loss = self.reconstruction_loss(
                self.config.lamb * encoder_out.detach() + (1 - self.config.lamb) * vocab_semantic[input_ids].detach(),
                autoencoder_out)
            mapping_loss = mp_loss + rc_loss
            # global_mapping_loss = self.loc_loss(src_ids, src_mask, global_dis_matrix, global_act_dis_tags)
            # mapping_loss = mapping_loss + global_mapping_loss
            self.graph_optimizer.zero_grad()
            mapping_loss.backward()
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.graph_optimizer.step()
                self.graph_scheduler.step()
            avg_loss_mapping += mapping_loss.item()
            iter_loss_mapping += mp_loss.item()
            iter_loss_recons += rc_loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": step,
                "avg_loss_m": "%.2f" % (avg_loss_mapping / (step + 1)),
                "iter_loss_mp": "%.2f" % (iter_loss_mapping / current_iter),
                "iter_loss_rc": "%.2f" % (iter_loss_recons / current_iter),
            }

            if step % self.log_freq == 0:
                iter_loss_mapping = 0
                iter_loss_recons = 0
                current_iter = 100
                data_loader.write(str(post_fix))

        print("EP%d_%s, mapping_avg_loss=" % (epoch, str_code), avg_loss_mapping / (len(data_loader) + 1e-8))
        return avg_loss_mapping / (len(data_loader) + 1e-8)

    def get_context_akn(self, input_ids, input_mask):
        max_act_len = torch.max(torch.sum(input_mask, dim=-1))
        act_dis_tags = torch.zeros(size=[input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]],
                                   device=input_ids.device)
        dis_matrix = torch.zeros(size=[input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]],
                                 device=input_ids.device, dtype=torch.float64)

        for shift in range(max_act_len):
            shift_tags = torch.zeros(size=[input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]],
                                     device=input_ids.device)  #
            if shift != 0:

                shift_tags[:, :-(shift), shift:] = torch.eye(input_ids.shape[1] - shift,
                                                             device=input_ids.device).unsqueeze(0).repeat(
                    input_ids.shape[0],
                    1, 1)  #
                act_dis_tags += shift_tags  #
                temp_akn = self.get_akn(input_ids[:, :-(shift)], input_ids[:, shift:])
                dis_matrix[shift_tags == 1] = torch.flatten(temp_akn, start_dim=0, end_dim=-1)  #

                shift_tags = torch.zeros(size=[input_ids.shape[0], input_ids.shape[1], input_ids.shape[1]],
                                         device=input_ids.device)  #
                shift_tags[:, shift:, :-(shift)] = torch.eye(input_ids.shape[1] - shift,
                                                             device=input_ids.device).unsqueeze(0).repeat(
                    input_ids.shape[0],
                    1, 1)  #
                temp_akn = self.get_akn(input_ids[:, shift:], input_ids[:, :-(shift)])
                dis_matrix[shift_tags == 1] = torch.flatten(temp_akn, start_dim=0, end_dim=-1)  #
            else:
                shift_tags = torch.eye(input_ids.shape[1],
                                       device=input_ids.device).unsqueeze(0).repeat(
                    input_ids.shape[0],
                    1, 1)  #
                act_dis_tags += shift_tags  #
                temp_akn = self.get_akn(input_ids, input_ids)
                dis_matrix[shift_tags == 1] = torch.flatten(temp_akn, start_dim=0, end_dim=-1)  #
        mean_dis = torch.div(torch.sum(torch.sum(dis_matrix, dim=-1), dim=-1),
                             torch.mul(torch.sum(input_mask, dim=-1), torch.sum(input_mask, dim=-1)))
        associative_score = torch.sigmoid(torch.div(dis_matrix, (mean_dis.unsqueeze(-1).unsqueeze(-1) + 1e-8))) - 0.5

        mean_as = torch.div(torch.sum(associative_score, dim=-1),
                            torch.sum(input_mask, dim=-1).unsqueeze(-1).repeat(1, input_mask.shape[1]))
        mean_as = mean_as.unsqueeze(-1)
        associative_score = associative_score / (mean_as + 1e-8)
        associative_score = associative_score * input_mask.unsqueeze(1)

        return associative_score

    def get_akn(self, pre, post):
        input_shape = pre.shape
        pre = pre.flatten()
        post = post.flatten()
        local_akn = self.akn[pre, post].reshape(input_shape)

        return local_akn

    def loc_loss(self, input_ids, input_mask, dis_matrix, act_dis_tags):
        local_akn = self.get_context_akn(input_ids, input_mask)
        similarities = torch.cosine_similarity(dis_matrix[act_dis_tags == 1], local_akn[act_dis_tags == 1], dim=-1)
        return 1 - torch.mean(similarities)

    def target_process(self, tgt_ids, tgt_mask, tgt_label):
        act_tgt_ids = tgt_ids * (tgt_label == 0) * tgt_mask
        act_tgt_mask = tgt_mask * (tgt_label == 0) * tgt_mask

        return act_tgt_ids, act_tgt_mask

    def ar_correction_loss(self, post_states, tgt_ids, input_mask):
        labels = tgt_ids[input_mask == 1]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(post_states[input_mask > 0].view(-1, self.config.vocab_size),
                        labels.long())
        decoder_ids = torch.argmax(post_states, dim=-1)
        total = torch.sum(input_mask > 0)
        correct = torch.sum(torch.eq(decoder_ids[input_mask > 0], labels))
        return loss, correct.item(), total.item()

    def correction_loss(self, decoder_out, prd_label):
        labels = prd_label[prd_label > 0]
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(decoder_out[prd_label > 0].view(-1, self.config.vocab_size),
                        labels.long())

        decoder_ids = torch.argmax(decoder_out, dim=-1)
        total = torch.sum(prd_label > 0)
        correct = torch.sum(torch.eq(decoder_ids[prd_label > 0], labels))
        return loss, correct.item(), total.item()

    def train(self, train_data, epoch, train=True):
        str_code = "train" if train else "val"
        self.model.to(self.device)
        self.model.train()
        self.classifier.train()
        data_loader = tqdm.tqdm(enumerate(train_data),
                                desc="EP_%s:%d" % (str_code, epoch),
                                total=len(train_data),
                                bar_format="{l_bar}{r_bar}")
        current_iter = 1
        c_correct = 0
        w_correct = 0
        c_total = 0
        w_total = 0
        avg_loss_m = 0
        iter_loss_m = 0

        for step, data in data_loader:
            # 1. Data Preprocess
            input_ids = data["input_text"]
            target_ids = data["target_text"]
            input_mask = (input_ids > 0).int()
            wrong_label = torch.zeros(size=input_ids.shape, device=target_ids.device, dtype=target_ids.dtype)
            wrong_label[input_ids != target_ids] = target_ids[input_ids != target_ids]
            encoder_out = self.model(input_ids, input_mask)
            classifier_out = self.classifier(encoder_out)
            ce_loss_fun = nn.CrossEntropyLoss()
            correction_loss = ce_loss_fun(classifier_out[input_mask > 0].view(-1, self.config.vocab_size),
                                          target_ids[input_mask > 0].view(-1).long())
            decoder_out = torch.argmax(classifier_out, dim=-1)
            c_correct += torch.sum(
                torch.eq(decoder_out[input_mask == 1][wrong_label[input_mask == 1] == 0],
                         target_ids[input_mask == 1][wrong_label[input_mask == 1] == 0])).item()  #
            w_correct += torch.sum(
                torch.eq(decoder_out[input_mask == 1][wrong_label[input_mask == 1] != 0],
                         target_ids[input_mask == 1][wrong_label[input_mask == 1] != 0])).item()  #
            c_total += torch.sum(input_mask[wrong_label == 0]).item()
            w_total += torch.sum(input_mask[wrong_label != 0]).item()
            avg_loss_m += correction_loss.item()
            iter_loss_m += correction_loss.item()
            self.lm_optimizer.zero_grad()
            correction_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)  #
            torch.nn.utils.clip_grad_norm_(self.classifier.parameters(), max_norm=self.max_grad_norm)  #

            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lm_optimizer.step()
                self.lm_scheduler.step()
            avg_loss_m += correction_loss.item()
            iter_loss_m += correction_loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": step,
                "avg_loss_c": "%.2f" % (avg_loss_m / (step + 1)),
                "iter_loss_m": "%.2f" % (iter_loss_m / current_iter),
                "iter_c": str(c_correct) + "/" + str(c_total),
                "iter_w": str(w_correct) + "/" + str(w_total),
            }

            if step % self.log_freq == 0:
                iter_loss_m = 0
                current_iter = 100
                c_total = 0
                w_total = 0
                c_correct = 0
                w_correct = 0
                data_loader.write(str(post_fix))

        print("EP%d_%s, correct_avg_loss=" % (epoch, str_code), avg_loss_m / len(data_loader))
        return avg_loss_m / len(data_loader)


class CorrectDataset(Dataset):
    def __init__(self, dataset, max_length):
        self.dataset = dataset
        self.data_size = len(dataset)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("-i", "--normal_epoch", type=int,
                        default=20,
                        help="training_epoch")
    parser.add_argument("-l1", "--normal_learning_rate", type=float,
                        default=2e-5,
                        help="learning_rate")
    parser.add_argument("-l2", "--mapping_learning_rate", type=float,
                        default=1e-5,
                        help="learning_rate")
    parser.add_argument("-e", "--mapping_epoch", type=int,
                        default=4,
                        help="training_epoch")
    args = parser.parse_args()
    torch.manual_seed(1)
    config = config()
    train = pd.read_csv(args.dataset + "/train_dataset.csv")

    train = CorrectDataset(train, max_length=config.max_length)
    train = DataLoader(train, batch_size=32, num_workers=0)
    trainer = Trainer(config=config, epoch=(int(train.dataset.data_size / train.batch_size) * args.normal_epoch,
                                            int(train.dataset.data_size / train.batch_size) * args.mapping_epoch),
                      lr=(args.normal_learning_rate, args.mapping_learning_rate))
    best_f1 = 0
    for e in range(args.normal_epoch):
        trainer.train(train, e)
        trainer.save_model()
        trainer.save_classifier()
    trainer.get_vocab_kozmo(train)

    for e in range(args.mapping_epoch):
        trainer.mapping_trainer(train, e)
        trainer.save_graph()
