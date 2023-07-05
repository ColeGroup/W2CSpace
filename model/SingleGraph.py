# The semantic kozmo consists of multiple merged semantic nodes. Specifically, we additionally expand 2 dimensions to the semantic nodes.
import torch
import torch.nn as nn


class SingleGraph(nn.Module):
    def __init__(self, config, mapping_matrix=None, graph=None, expand_akn=None, corres_ids=None):
        super(SingleGraph, self).__init__()
        self.config = config
        self.total_index = 0
        self.center_index = 0
        if graph is not None:
            self.graph = torch.load(graph).to(self.config.device)
        else:
            self.graph = torch.zeros(size=[1, 1, self.config.kozmo_size], device=self.config.device).unsqueeze(0)
        self.akn = torch.load(self.config.akn_path).to(self.config.device)
        if expand_akn is not None:
            self.expand_akn = torch.load(expand_akn).to(self.config.device)
        else:
            self.expand_akn = torch.zeros(size=[1, 1, self.config.vocab_size], device=self.config.device).unsqueeze(0)
        if corres_ids is not None:
            self.corres_ids = torch.load(corres_ids).to(self.config.device)
        else:
            self.corres_ids = torch.zeros(size=[1, 1, 1], device=self.config.device).unsqueeze(0)
        self.mapping = AutoEncoder(in_feature=self.config.hidden_size, out_feature=self.config.kozmo_size,
                                   config=self.config).to(
            self.config.device)

    def forward(self, representation, src_mask):
        src_dis_mask = torch.mul(src_mask.unsqueeze(-1).repeat(1, 1, src_mask.shape[1]), src_mask.unsqueeze(1))
        act_dis_tags = torch.zeros(size=[representation.shape[0], representation.shape[1], representation.shape[1]],
                                   device=representation.device)
        dis_matrix = torch.zeros(size=[representation.shape[0], representation.shape[1], representation.shape[1]],
                                 device=representation.device)
        location, decoder_out = self.mapping(representation)
        max_act_len = torch.max(torch.sum(src_mask, dim=-1))
        for shift in range(max_act_len):
            shift_tags = torch.zeros(size=[representation.shape[0], representation.shape[1], representation.shape[1]],
                                     device=representation.device)  #

            shift_tags[:, :-(shift + 1), shift + 1:] = torch.eye(representation.shape[1] - shift - 1,
                                                                 device=representation.device).unsqueeze(0).repeat(
                representation.shape[0],
                1, 1)  #
            act_dis_tags += shift_tags  #
            shift_dis = self.get_cosine_distance(location[:, shift + 1:], location[:, :-(shift + 1)])
            dis_matrix[shift_tags == 1] = torch.flatten(shift_dis, start_dim=0, end_dim=-1)  #
        return location, dis_matrix, act_dis_tags * src_dis_mask, decoder_out

    def get_cosine_distance(self, a, b):
        cose = torch.cosine_similarity(a, b, dim=-1)
        return cose

    def get_euclidean_dis(self, shift_location):
        dis = torch.sqrt(torch.sum(torch.pow(shift_location, 2) + 1e-8, dim=-1) + 1e-8)

        return dis

    def update(self, ids, locations):
        # SK update
        self.graph[self.total_index] = torch.cat((self.graph[self.total_index], locations.unsqueeze(0)), dim=0)
        # EA update
        self.corres_ids[self.total_index] = torch.cat((self.corres_ids[self.total_index], ids.unsqueeze(0)), dim=0)

    def merge(self, gravity=1):
        total_graph = self.graph[self.total_index]
        center_dis = torch.cos((torch.sum(
            torch.pow(self.graph[self.total_index] - self.graph[self.total_index][self.center_index].unsqueeze(0), 2),
            dim=-1)))
        center_dis_mark = center_dis // gravity
        for gravity_index in range(torch.max(center_dis_mark)):
            index_center_mark = center_dis_mark == gravity_index

    def expand2akn(self):
        pass


class AutoEncoder(nn.Module):
    def __init__(self, in_feature, out_feature, config):
        super(AutoEncoder, self).__init__()
        self.config = config
        self.dropout1 = nn.Dropout2d(p=0.2)
        self.dropout2 = nn.Dropout2d(p=0.2)
        self.dropout3 = nn.Dropout2d(p=0.2)
        self.dropout4 = nn.Dropout2d(p=0.2)

        self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.eps)
        self.encoder_convs = nn.ModuleList(
            [nn.Conv2d(1, int(in_feature / len(self.config.filters)), (k, in_feature), padding=(int(k / 2), 0), bias=False) for k in
             self.config.filters])
        self.encoder_linear = nn.Linear(self.config.hidden_size, self.config.hidden_size, bias=False)
        self.encoder_activation = nn.Tanh()
        self.encoder_transform = nn.Linear(self.config.hidden_size, out_feature, bias=False)

        self.decoder_transform = nn.Linear(out_feature, in_feature, bias=False)

        self.decoder_convs = nn.ModuleList(
            [nn.Conv2d(1, int(in_feature / len(self.config.filters)), (k, in_feature), padding=(int(k / 2), 0),
                       bias=False) for k in
             self.config.filters])
        self.decoder_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.decoder_activation = nn.Tanh()
        self.output_linear = nn.Linear(in_feature, in_feature, bias=False)
        self.init_model_weight()


    def init_model_weight(self):
        for conv in self.encoder_convs:
            nn.init.xavier_uniform_(conv.weight)
        for conv in self.decoder_convs:
            nn.init.xavier_uniform_(conv.weight)

    def forward(self, hidden_states, autoencode=True):
        input_hidden = hidden_states
        hidden_states = torch.cat([conv(hidden_states.unsqueeze(1)).squeeze(3).transpose(-1, -2) for conv in self.encoder_convs], dim=-1)
        hidden_states = self.dropout1(hidden_states)
        hidden_states = self.layernorm(hidden_states + input_hidden)
        hidden_states = self.encoder_linear(hidden_states)
        hidden_states = self.dropout2(hidden_states)
        hidden_states = self.encoder_activation(hidden_states)
        encode_out = self.encoder_transform(hidden_states)

        if autoencode:
            hidden_states = self.decoder_transform(encode_out)
            hidden_states = torch.cat(
                [conv(hidden_states.unsqueeze(1)).squeeze(3).transpose(-1, -2) for conv in self.decoder_convs], dim=-1)
            hidden_states = self.dropout3(hidden_states)
            hidden_states = self.decoder_linear(hidden_states)
            hidden_states = self.dropout4(hidden_states)

            hidden_states = self.decoder_activation(hidden_states)
            decode_out = self.output_linear(hidden_states)
            return encode_out, decode_out
        return encode_out, None
