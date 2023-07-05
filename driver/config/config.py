import torch
from transformers import AutoConfig
import copy
import os
import random
import re
import sys
from transformers.models.bert.modeling_bert import BertConfig

Base_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(Base_DIR)
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]


class config():
    def __init__(self):
        self.hidden_size = 768
        self.kozmo_size = 100
        self.akn_path = rootPath + "/akn/akn.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.bert_path = "bert-base-chinese"
        self.bert_config = BertConfig.from_pretrained("bert-base-chinese")
        self.decoder_config = BertConfig.from_pretrained("bert-base-chinese")
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.max_length = 128
        self.dropout = 0.2
        self.eps = 1e-12
        self.head_num = 12
        self.position_embedding_type = "absolute"  # "relative_key"
        self.vocab_size = 21128
        self.num_decoder_layer = 6
        self.lamb = 0.5
        self.k = 9
        self.filters = [5, 13, 23]  # CNN卷积核宽度
        self.filter_num = 256  # CNN卷积核个数
        self.kozmo_num = 768
        self.kozmo_head = 12
        self.kozmo_layer = 12
        self.semaspace = 500
        self.num_labels = 2
        self.mode = "correction_out"
        self.normal_classifier_path = rootPath + self.mode + "/normal_classifier/classifier.pt"
        self.normal_encoder_path = rootPath + self.mode + "/normal_encoder/encoder.pt"
        self.mapper_path = rootPath + self.mode + "/mapper/" + "mapping" + str(self.kozmo_size) + ".pt"
        self.sent_kozmo_path = rootPath + self.mode + "/kozmo/" + "sent_kozmo" + str(self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.word_kozmo_path = rootPath + self.mode + "/kozmo/" + "word_kozmo" + str(self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.vocab_path = rootPath + self.mode + "/vocabs/vocabs.pt"
        self.encoder_path = rootPath + self.mode + "/encoder/encoder" + str(self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.semaspace_path = rootPath + self.mode + "/semaspace/" + "semaspace" + str(self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
    def reload(self):
        self.hidden_size = 768
        self.kozmo_size = 50
        self.akn_path = rootPath + "/akn/akn.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")
        self.bert_path = "bert-base-chinese"
        self.bert_config = BertConfig.from_pretrained("bert-base-chinese")
        self.decoder_config = BertConfig.from_pretrained("bert-base-chinese")
        self.decoder_config.is_decoder = True
        self.decoder_config.add_cross_attention = True
        self.max_length = 128
        self.dropout = 0.2
        self.eps = 1e-12
        self.head_num = 12
        self.position_embedding_type = "absolute"  # "relative_key"
        self.vocab_size = 21128
        self.num_decoder_layer = 6
        self.lamb = 0.5
        self.k = 9
        self.filters = [5, 13, 23]  # CNN卷积核宽度
        self.filter_num = 256  # CNN卷积核个数
        self.kozmo_num = 768
        self.kozmo_head = 12
        self.kozmo_layer = 12
        # self.semaspace = 100
        self.num_labels = 2
        self.mode = "classify_out2"
        self.normal_classifier_path = rootPath + self.mode + "/normal_classifier/classifier.pt"
        self.normal_encoder_path = rootPath + self.mode + "/normal_encoder/encoder.pt"
        self.mapper_path = rootPath + self.mode + "/mapper/" + "mapping" + str(
            self.kozmo_size) + ".pt"
        self.sent_kozmo_path = rootPath + self.mode + "/kozmo/" + "sent_kozmo" + str(
            self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.word_kozmo_path = rootPath + self.mode + "/kozmo/" + "word_kozmo" + str(
            self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.vocab_path = rootPath + self.mode + "/vocabs/vocabs.pt"
        self.encoder_path = rootPath + self.mode + "/encoder/encoder" + str(
            self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
        self.semaspace_path = rootPath + self.mode + "/semaspace/" + "semaspace" + str(
            self.kozmo_size) + "_" + str(self.semaspace) + ".pt"
