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
from transformers.optimization import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from torch.optim import AdamW
from transformers.models.bert.modeling_bert import BertConfig
from model.KozmoClassifier import KozmoClassifier


class Trainer:
    def __init__(self, config, epoch, lr=4e-4, gama=0.8, weight_decay=0.01,
                 save_path="",
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        self.num_warmup = epoch / 10
        self.num_training = epoch
        self.max_grad_norm = max_grad_norm
        self.config = config
        self.pretrain_config = BertConfig.from_pretrained(self.config.bert_path)
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, local_files_only=True)
        self.model = torch.load(self.config.normal_encoder_path).to(self.config.device)
        self.kozmo_decoder = KozmoClassifier(config).to(self.config.device)
        self.akn = torch.load(self.config.akn_path).to(self.config.device)
        self.gama = gama
        self.save_path = save_path
        self.device = self.config.device
        self.log_freq = 100
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.kc_param = self.get_param(self.kozmo_decoder, weight_decay)
        self.lm_param = self.get_param(self.model, weight_decay)
        self.lm_optimizer = AdamW(params=self.lm_param + self.kc_param, lr=lr)
        self.lm_scheduler = get_linear_schedule_with_warmup(self.lm_optimizer,
                                                            num_warmup_steps=self.num_warmup,
                                                            num_training_steps=self.num_training)

    def get_param(self, model, weight_decay):
        no_decay = ['bias', 'LayerNorm.Weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]

        return optimizer_grouped_parameters

    def save_model(self):
        torch.save(self.model, self.config.encoder_path)

    def save_semaspace(self):
        torch.save(self.kozmo_decoder, self.config.semaspace_path)

    def save(self):
        self.save_model()
        self.save_semaspace()

    def get_akn(self, pre, post):
        input_shape = pre.shape
        pre = pre.flatten()
        post = post.flatten()
        local_akn = self.akn[pre, post].reshape(input_shape)

        return local_akn

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
        # for batch_num, batch in enumerate(dis_matrix):
        #     dis_matrix[batch_num] = dis_matrix[input_mask[batch_num].usqueeze(0).repeat(input_mask.shape[1], 1)]
        mean_dis = torch.div(torch.sum(torch.sum(dis_matrix, dim=-1), dim=-1), torch.mul(torch.sum(input_mask, dim=-1), torch.sum(input_mask, dim=-1)))
        associative_score = torch.sigmoid(torch.div(dis_matrix, (mean_dis.unsqueeze(-1).unsqueeze(-1) + 1e-8))) - 0.5

        mean_as = torch.div(torch.sum(associative_score, dim=-1), torch.sum(input_mask, dim=-1).unsqueeze(-1).repeat(1, input_mask.shape[1]))
        mean_as = mean_as.unsqueeze(-1)
        associative_score = associative_score / (mean_as + 1e-8)
        associative_score = associative_score * input_mask.unsqueeze(1)

        return associative_score

    def loc_loss(self, input_ids, input_mask, dis_matrix, act_dis_tags):
        local_akn = self.get_context_akn(input_ids, input_mask)
        similarities = torch.cosine_similarity(dis_matrix[act_dis_tags == 1], local_akn[act_dis_tags == 1], dim=-1)
        return 1 - torch.mean(similarities)

    def reconstruction_loss(self, encoder_out, decoder_out):
        loss_fun = nn.L1Loss()
        loss = loss_fun(encoder_out, decoder_out)

        return loss

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

    def do_correct(self, sentence):
        predict_label = self.evaluate(sentence)

        return predict_label

    def preprocess(self, input_sentence):
        tokenized_sentence = self.tokenizer(input_sentence, padding='max_length', max_length=128)
        input_text = torch.tensor(tokenized_sentence["input_ids"], device=self.device).unsqueeze(0)
        input_mask = torch.tensor(tokenized_sentence["attention_mask"], device=self.device).unsqueeze(0)
        return input_text, input_mask

    def evaluate(self, input_sentence):
        self.model.to(self.device)
        self.model.eval()
        self.kozmo_decoder.to(self.device)
        self.kozmo_decoder.eval()
        input_ids, input_mask = self.preprocess(input_sentence)
        encoder_out, _ = self.model(input_ids, input_mask)
        classifier_out, kozmo_dis = self.kozmo_decoder(encoder_out)
        decode_out = torch.argmax(classifier_out, dim=-1)
        return decode_out.item()

    def merge(self, kozmo_dis, input_mask):
        act_dis_tags = torch.zeros(size=[kozmo_dis.shape[0], kozmo_dis.shape[1], kozmo_dis.shape[1]],
                                   device=kozmo_dis.device)
        dis_matrix = torch.zeros(size=[kozmo_dis.shape[0], kozmo_dis.shape[1], kozmo_dis.shape[1]],
                                 device=kozmo_dis.device)
        max_act_len = torch.max(torch.sum(input_mask, dim=-1))
        for shift in range(max_act_len):
            shift_tags = torch.zeros(size=[kozmo_dis.shape[0], kozmo_dis.shape[1], kozmo_dis.shape[1]],
                                     device=kozmo_dis.device)  #

            shift_tags[:, :-(shift + 1), shift + 1:] = torch.eye(kozmo_dis.shape[1] - shift - 1,
                                                                 device=kozmo_dis.device).unsqueeze(0).repeat(
                kozmo_dis.shape[0],
                1, 1)  #
            act_dis_tags += shift_tags  #
            shift_dis = torch.cosine_similarity(kozmo_dis[:, shift + 1:], kozmo_dis[:, :-(shift + 1)], dim=-1)
            # shift_location = location[:, shift + 1:] - location[:, :-(shift + 1)]  #
            # shift_dis = self.get_euclidean_dis(shift_location)  #
            dis_matrix[shift_tags == 1] = torch.flatten(shift_dis, start_dim=0, end_dim=-1)  #

        return dis_matrix, act_dis_tags

    def sighan_eval(self):
        current_count = 0
        correct_count = 0
        test_file = pd.read_csv("/home/wangfanyu/project/GraphDecoder/data/Weibo/test2.csv")
        labels = test_file["label"]
        texts = test_file["text"]
        for label, text in zip(labels, texts):
            strip_line = text.strip()
            predict_label = self.do_correct(strip_line)
            current_count += 1
            if predict_label == label:
                correct_count += 1

        print(correct_count / current_count)

    def train(self, train_data, epoch, train=True):
        str_code = "train" if train else "val"
        self.model.to(self.device)
        self.model.train()
        self.kozmo_decoder.to(self.device)
        self.kozmo_decoder.train()
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
            label = data["label"]
            input_ids = data["input_text"]
            input_mask = (input_ids > 0).int()
            encoder_out, _ = self.model(input_ids, input_mask)
            classifier_out, kozmo_dis = self.kozmo_decoder(encoder_out)
            ce_loss_fun = nn.CrossEntropyLoss()
            correction_loss = ce_loss_fun(classifier_out,
                                          label.view(-1).long())
            loss = correction_loss
            decoder_out = torch.argmax(classifier_out, dim=-1)
            c_correct += torch.sum(
                torch.eq(decoder_out[label == 1],
                         label[label == 1])).item()  #
            w_correct += torch.sum(
                torch.eq(decoder_out[label == 0],
                         label[label == 0])).item()  #
            c_total += torch.sum(label == 1).item()
            w_total += torch.sum(label == 0).item()
            avg_loss_m += correction_loss.item()
            iter_loss_m += correction_loss.item()
            self.lm_optimizer.zero_grad()
            correction_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)  #
            torch.nn.utils.clip_grad_norm_(self.kozmo_decoder.parameters(), max_norm=self.max_grad_norm)  #

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

        return {key: value for key, value in output.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--dataset", type=str, default=rootPath + "/data",
                        help="train dataset")
    parser.add_argument("-m", "--model_path", type=str,
                        default="",
                        help="model save path")
    parser.add_argument("-e", "--epoch", type=int,
                        default=3,
                        help="training_epoch")
    parser.add_argument("-l", "--learning_rate", type=float,
                        default=8e-6,
                        help="learning rate for classification task")
    args = parser.parse_args()
    torch.manual_seed(1)
    config = config()
    train = pd.read_csv(args.dataset + "/Weibo/train2.csv")
    train = ClassifyDataset(train, max_length=config.max_length, device=config.device)

    train = DataLoader(train, batch_size=32, num_workers=0)
    print(args.epoch)
    print(args.learning_rate)
    trainer = Trainer(config=config, save_path=rootPath + args.model_path, epoch=int(train.dataset.data_size / train.batch_size) * args.epoch, lr=args.learning_rate)
    for e in range(args.epoch):
        trainer.train(train, e)
        trainer.save()
        trainer.sighan_eval()
