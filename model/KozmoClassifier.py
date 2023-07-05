import torch
import torch.nn as nn
from model.LocalBERT import BertLayer, BertModel, BertEmbeddings
from transformers.activations import ACT2FN
import copy

class GraphEncoder(nn.Module):
    def __init__(self, config):
        super(GraphEncoder, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)

    def forward(self, input_ids, input_mask):
        hidden_state = self.bert(input_ids, input_mask)[0]

        return hidden_state, None


class KozmoClassifier(nn.Module):
    def __init__(self, config):
        super(KozmoClassifier, self).__init__()
        self.config = config
        self.decode_layer1 = torch.load(self.config.sent_kozmo_path).to(
            self.config.device)
        self.linear = nn.Linear(self.config.semaspace, self.config.num_labels)
        self.dropout = nn.Dropout(0.3)
        self.dense = nn.Linear(self.config.semaspace, self.config.semaspace)
        self.activation = nn.Tanh()
        self.layer_norm1 = nn.LayerNorm(self.decode_layer1.shape[0], elementwise_affine=False)
        self.linear1 = nn.Linear(self.decode_layer1.shape[0], self.decode_layer1.shape[0], bias=False)
        self.graph = torch.load(self.config.mapper_path).to(self.config.device)
        self.graph.eval()

    def forward(self, hidden_states, ana=False, set_dis=None):
        if set_dis is None:
            mapped_hidden_states, _ = self.graph.mapping(hidden_states)
            kozmo_hidden1 = torch.matmul(self.decode_layer1.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1),
                                         mapped_hidden_states.transpose(-1, -2)).squeeze(0).transpose(-1, -2)
            release_kozmo_hidden1 = copy.copy(kozmo_hidden1.detach())
        else:
            kozmo_hidden1 = set_dis
        if len(kozmo_hidden1.shape) == 2:
            kozmo_hidden1 = kozmo_hidden1.unsqueeze(0)
        kozmo_hidden1 = self.layer_norm1(kozmo_hidden1)
        matrix_kozmo_hidden1 = self.linear1(kozmo_hidden1)
        first_token_tensor = matrix_kozmo_hidden1[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        pooled_output = self.linear(pooled_output)
        if ana:
            return release_kozmo_hidden1

        return pooled_output, matrix_kozmo_hidden1



class KozmoDecoder(nn.Module):
    def __init__(self, config):
        super(KozmoDecoder, self).__init__()
        self.config = config
        self.decode_layer1 = torch.load(self.config.sent_kozmo_path).to(
            self.config.device)
        self.linear1 = nn.Linear(self.decode_layer1.shape[0], self.decode_layer1.shape[0], bias=False)
        self.decoder_layer_size = self.decode_layer1.shape[0]
        self.transform = nn.Linear(self.decoder_layer_size,
                                   self.decoder_layer_size)
        self.transform_act_fn = ACT2FN["relu"]
        self.layer_norm = nn.LayerNorm(self.decoder_layer_size)
        self.graph = torch.load(self.config.mapper_path).to(self.config.device)
        self.graph.eval()
        self.to_vocab = nn.Linear(self.decoder_layer_size, self.config.vocab_size)
        self.layer_norm1 = nn.LayerNorm(self.decode_layer1.shape[0], elementwise_affine=False)

    def forward(self, hidden_states):
        mapped_hidden_states, _ = self.graph.mapping(hidden_states)
        kozmo_hidden1 = torch.matmul(self.decode_layer1.unsqueeze(0).repeat(hidden_states.shape[0], 1, 1),
                                     mapped_hidden_states.transpose(-1, -2)).squeeze(0).transpose(-1, -2)
        kozmo_hidden1 = self.layer_norm1(kozmo_hidden1)
        matrix_kozmo_hidden1 = self.linear1(kozmo_hidden1)
        kozmo_hidden = self.transform(matrix_kozmo_hidden1)

        kozmo_hidden = self.transform_act_fn(kozmo_hidden)
        kozmo_hidden = self.layer_norm(kozmo_hidden)
        kozmo_hidden = self.to_vocab(kozmo_hidden)

        return kozmo_hidden, matrix_kozmo_hidden1


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class SqGraphEncoder(nn.Module):
    def __init__(self, config):
        super(SqGraphEncoder, self).__init__()
        self.config = config
        self.bert = BertModel.from_pretrained(self.config.bert_path)

    def forward(self, input_ids, input_mask):
        out = self.bert(input_ids, input_mask)
        hidden_state = out[0]
        pooled_out = out[1]
        return hidden_state, pooled_out

class SequencePrediction(nn.Module):
    def __init__(self, config):
        super(SequencePrediction, self).__init__()
        self.config = config
        self.num_labels = self.config.num_labels
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.config.hidden_size, self.num_labels)

    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.classifier(hidden_states)

        return hidden_states
