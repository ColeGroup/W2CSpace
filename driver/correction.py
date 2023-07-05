
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
from model.KozmoClassifier import KozmoDecoder


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
        self.kozmo_decoder = KozmoDecoder(config).to(self.config.device)
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
        predict_sentence, decode_ss = self.evaluate(sentence)

        return predict_sentence

    def preprocess(self, input_sentence):
        tokenized_sentence = self.tokenizer(input_sentence, padding='max_length', max_length=128)
        input_text = torch.tensor(tokenized_sentence["input_ids"], device=self.device).unsqueeze(0)
        input_mask = torch.tensor(tokenized_sentence["attention_mask"], device=self.device).unsqueeze(0)
        return input_text, input_mask

    def character_check(self, ori_sent, pred_sent, gold_sent):
        ctp, dtp = 0, 0
        ctn, dtn = 0, 0
        cfp, dfp = 0, 0
        cfn, dfn = 0, 0
        for ori_tags, prd_tags, god_tags in zip(ori_sent, pred_sent, gold_sent):
            pred_pos, pred_char = self.compare_pos(ori_tags, prd_tags)
            gold_pos, gold_char = self.compare_pos(ori_tags, god_tags)
            if len(gold_pos) == 0:
                continue
            for char_num, _ in enumerate(ori_tags):
                if char_num in pred_pos or char_num in gold_pos:
                    if ori_tags[char_num] != god_tags[char_num]:
                        if god_tags[char_num] == prd_tags[char_num]:
                            ctp += 1
                        else:
                            cfn += 1
                        if prd_tags[char_num] != ori_tags[char_num]:
                            dtp += 1
                        else:
                            dfn += 1
                    else:
                        cfp += 1
                        dfp += 1
        dp = dtp / (dtp + dfp + 1e-8)
        dr = dtp / (dtp + dfn + 1e-8)
        df1 = 2 * dp * dr / (dp + dr + 1e-8)
        print("character detection result")
        print("P=" + str(dp) + "|R=" + str(dr) + "|F1=" + str(df1))
        cp = ctp / (dtp + 1e-8)
        cr = ctp / (ctp + cfn + 1e-8)
        cf1 = 2 * cp * cr / (cp + cr + 1e-8)
        print("character correction result")
        print("P=" + str(cp) + "|R=" + str(cr) + "|F1=" + str(cf1))

    def compare_pos(self, sent1, sent2):
        dif_pos = []
        dif_char = []
        for char_num, _ in enumerate(sent1):
            if sent1[char_num] != sent2[char_num]:
                dif_pos.append(char_num)
                dif_char.append([sent1[char_num], sent2[char_num]])
        return dif_pos, dif_char

    def sent_check(self, ori_sent, pred_sent, gold_sent):
        ctp, dtp = 0, 0
        ctn, dtn = 0, 0
        cfp, dfp = 0, 0
        cfn, dfn = 0, 0
        for ori_tags, prd_tags, god_tags in zip(ori_sent, pred_sent, gold_sent):
            pred_pos, _ = self.compare_pos(ori_tags, prd_tags)
            gold_pos, _ = self.compare_pos(ori_tags, god_tags)
            if len(gold_pos) > 0:
                if prd_tags == god_tags:
                    ctp += 1
                else:
                    cfn += 1
                if gold_pos == pred_pos:
                    dtp += 1
                else:
                    dfn += 1
            else:
                if prd_tags == god_tags:
                    ctn += 1
                else:
                    cfp += 1
                if gold_pos == pred_pos:
                    dtn += 1
                else:
                    dfp += 1
        dp = dtp / (dtp + dfp + 1e-8)
        dr = dtp / (dtp + dfn + 1e-8)
        df1 = 2 * dp * dr / (dp + dr + 1e-8)
        print("sent detection result")
        print("P=" + str(dp) + "|R=" + str(dr) + "|F1=" + str(df1))
        cp = ctp / (ctp + cfp + 1e-8)
        cr = ctp / (ctp + cfn + 1e-8)
        cf1 = 2 * cp * cr / (cp + cr + 1e-8)
        print("sent correction result")
        print("P=" + str(cp) + "|R=" + str(cr) + "|F1=" + str(cf1))
        return cf1

    def evaluate(self, input_sentence):
        self.model.to(self.device)
        self.model.eval()
        self.kozmo_decoder.to(self.device)
        self.kozmo_decoder.eval()
        input_ids, input_mask = self.preprocess(input_sentence)
        encoder_out, _ = self.model(input_ids, input_mask)
        classifier_out, _ = self.kozmo_decoder(encoder_out)
        if classifier_out.shape[-1] != self.config.vocab_size or len(classifier_out.shape) != 3:
            classifier_out = classifier_out.unsqueeze(0)
        decode_out = torch.argmax(classifier_out, dim=-1)
        decode_sentence = self.tokenizer.convert_ids_to_tokens(decode_out[0][1:torch.sum(input_mask, dim=-1) - 1])
        predict_sentence = "".join(decode_sentence)
        return predict_sentence, decode_out[1:torch.sum(input_mask, dim=-1) - 1]

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

    def sighan_eval(self, test_dir):
        write_lines = []
        current_count = 0
        with open(test_dir + "15test.txt", "r",
                  encoding="utf-8") as f:
            lines = f.readlines()
            # for line in lines:
            #     input_lines.append(line.strip().split("\t")[1])
            for line in lines:
                strip_line = line.strip()
                wrong_sentence = strip_line
                predict_sentence = self.do_correct(wrong_sentence)
                line = wrong_sentence + "\t" + predict_sentence + "\n"
                write_lines.append(line)
                current_count += 1
                # print(current_count)
                with open(test_dir + "my_predict.txt",
                          "w",
                          encoding="utf-8") as f:
                    f.writelines(write_lines)
        print('zi result:')
        all_inputs_sent = []
        all_golds_sent = []
        all_preds_sent = []
        with open(test_dir + "15test.txt", "r") as input:
            for line in input.readlines():
                all_inputs_sent.append(line.strip("\n"))
        with open(test_dir + "15truth.txt", "r") as input:
            for line in input.readlines():
                all_golds_sent.append(line.strip("\n"))
        with open(test_dir + "my_predict.txt",
                  "r") as input:
            for line in input.readlines():
                all_preds_sent.append(line.strip("\n").split("\t")[1])
        self.character_check(all_inputs_sent, all_preds_sent, all_golds_sent)
        f_sent = self.sent_check(all_inputs_sent, all_preds_sent, all_golds_sent)
        return f_sent

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
        iter_loss_x = 0
        for step, data in data_loader:
            # 1. Data Preprocess
            input_ids = data["input_text"]
            target_ids = data["target_text"]
            input_mask = (input_ids > 0).int()
            wrong_label = torch.zeros(size=input_ids.shape, device=target_ids.device, dtype=target_ids.dtype)
            wrong_label[input_ids != target_ids] = target_ids[input_ids != target_ids]
            encoder_out, _ = self.model(input_ids, input_mask)
            classifier_out, kozmo_dis = self.kozmo_decoder(encoder_out)
            if classifier_out.shape[-1] != self.config.vocab_size or len(classifier_out.shape) != 3:
                classifier_out = classifier_out.unsqueeze(0)
                kozmo_dis = kozmo_dis.unsqueeze(0)
            # merged_kozmo_dis, act_dis_tags = self.merge(kozmo_dis, input_mask)
            # loss_xie = 1 - torch.cosine_similarity(merged_kozmo_dis[act_dis_tags == 1], local_akn[act_dis_tags == 1],
            #                                        dim=0)
            ce_loss_fun = nn.CrossEntropyLoss()
            correction_loss = ce_loss_fun(classifier_out[input_mask > 0].view(-1, self.config.vocab_size),
                                          target_ids[input_mask > 0].view(-1).long())
            loss = correction_loss
            decoder_out = torch.argmax(classifier_out, dim=-1)
            c_correct += torch.sum(
                torch.eq(decoder_out[input_mask == 1][wrong_label[input_mask == 1] == 0],
                         target_ids[input_mask == 1][wrong_label[input_mask == 1] == 0])).item()  #
            w_correct += torch.sum(
                torch.eq(decoder_out[input_mask == 1][wrong_label[input_mask == 1] != 0],
                         target_ids[input_mask == 1][wrong_label[input_mask == 1] != 0])).item()  #
            c_total += torch.sum(input_mask[wrong_label == 0]).item()
            w_total += torch.sum(input_mask[wrong_label != 0]).item()
            avg_loss_m += loss.item()
            # iter_loss_x += loss_xie.item()
            iter_loss_m += correction_loss.item()
            self.lm_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm)  #
            if (step + 1) % self.gradient_accumulation_steps == 0:
                self.lm_optimizer.step()
                self.lm_scheduler.step()
            avg_loss_m += correction_loss.item()
            iter_loss_m += correction_loss.item()
            post_fix = {
                "epoch": epoch,
                "iter": step,
                # "avg_loss_m": "%.2f" % (avg_loss_m / (step + 1)),
                "iter_loss_c": "%.2f" % (iter_loss_m / current_iter),
                # "iter_loss_x": "%.2f" % (iter_loss_x / current_iter),
                "iter_c": str(c_correct) + "/" + str(c_total),
                "iter_w": str(w_correct) + "/" + str(w_total),
            }

            if step % self.log_freq == 0:
                iter_loss_m = 0
                current_iter = 100
                # iter_loss_x = 0
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
        # self.device = torch.device("cpu")

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
    parser.add_argument("-m", "--model_path", type=str,
                        default="/py_out27",
                        help="model save path")
    parser.add_argument("-e", "--epoch", type=int,
                        default=20,
                        help="training_epoch")
    parser.add_argument("-l", "--learning_rate", type=float,
                        default=2e-5,
                        help="training_epoch")
    parser.add_argument("-t", "--test_dir", type=str,
                        default=2e-5,
                        help="test file direction")
    args = parser.parse_args()
    torch.manual_seed(1)
    config = config()
    train = pd.read_csv(args.dataset + "/train_dataset.csv")
    train = CorrectDataset(train, max_length=config.max_length)
    train = DataLoader(train, batch_size=32, num_workers=0)
    test_dir = args.test_dir
    print(args.epoch)
    print(args.learning_rate)
    trainer = Trainer(config=config, save_path=rootPath + args.model_path, epoch=int(train.dataset.data_size / train.batch_size) * args.epoch, lr=args.learning_rate)
    for e in range(args.epoch):
        trainer.train(train, e)
        trainer.save()
        trainer.sighan_eval(test_dir)
