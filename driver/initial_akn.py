from transformers import BertTokenizer
import numpy as np
import re


def calculate_sentence(sentence_ids, network, shrink=0.9):
    for pivot_char_idx, pivot_char_ids in enumerate(sentence_ids):
        for query_char_idx, query_char_ids in enumerate(sentence_ids[pivot_char_idx + 1:]):
            if query_char_idx == 0 or query_char_idx == (len(sentence_ids) - 1):
                network[pivot_char_ids][query_char_ids] = (network[pivot_char_ids][query_char_ids] + 1 / (
                            query_char_idx + 1))
                # network[pivot_char_ids][query_char_ids] = (network[pivot_char_ids][query_char_ids] + 1 / (query_char_idx + 1)) * shrink
    return network


def cut_sent(para):
    para = re.sub('([。！？\?])([^”’])', r"\1\n\2", para)
    para = re.sub('(\.{6})([^”’])', r"\1\n\2", para)
    para = re.sub('(\…{2})([^”’])', r"\1\n\2", para)
    # para = re.sub('([。！？\?][”’])([^，。！？\?])', r'\1\n\2', para)
    para = re.sub('([。！？\?][”’])([^。！？\?])', r'\1\n\2', para)
    para = para.rstrip()  # \n
    return para.split("\n")


def normalization(network):
    for left_ids, left_ids_intensity in enumerate(network):
        left_ids_intensity_sum = np.sum(left_ids_intensity)
        if left_ids_intensity_sum != 0:
            for left_right_ids, left_right_ids_intensity in enumerate(left_ids_intensity):
                network[left_ids][left_right_ids] = left_right_ids_intensity / left_ids_intensity_sum
    return network


if __name__ == '__main__':
    total_sentence = []
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)
    my_net = np.zeros(shape=[tokenizer.vocab_size, tokenizer.vocab_size])
    for file in range(0, 2400):
        print(file, flush=True)
        with open("./data/txts/news_" + str(file) + ".txt", "r",
                  encoding="utf-8") as article:
            article_content = article.read()
            sentences = cut_sent(article_content)
            total_sentence = total_sentence + sentences
    for sent_num, sentence in enumerate(total_sentence):
        if sent_num % 1000000 == 0:
            print(str(sent_num)+"|"+str(len(total_sentence)-sent_num) ,flush=True)
        if len(sentence) > 510 or len(sentence) == 0:
            continue
        sentence_tokenize_result = tokenizer.encode_plus(text=sentence,
                                                         # max_length=64,
                                                         pad_to_max_length=False,
                                                         return_token_type_ids=True,
                                                         return_attention_mask=True,
                                                         return_tensors="np")
        sentence_ids = sentence_tokenize_result["input_ids"][0]
        for pivot_char_idx, pivot_char_ids in enumerate(sentence_ids):
            for query_char_idx, query_char_ids in enumerate(sentence_ids[pivot_char_idx + 1:]):
                my_net[pivot_char_ids][query_char_ids] = (my_net[pivot_char_ids][query_char_ids] + 1 / (
                        query_char_idx + 1))
    # my_net = normalization(my_net)
    np.save('./akn/new_akn.npy', my_net)

