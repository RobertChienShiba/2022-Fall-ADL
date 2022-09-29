from typing import Dict
import logging

import torch
import torch.nn as nn
from torch.nn import Embedding


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        pad_id: int,
        bidirectional: bool,
        num_class: int,
        model_name: str,
        init_method: str,
        device: torch.device
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.pad_id = pad_id
        self.bidirectional = bidirectional
        self.num_class = num_class
        self.init_method = init_method
        self.embed = nn.Sequential(
            Embedding.from_pretrained(embeddings, freeze=False, padding_idx=self.pad_id),
            nn.LayerNorm(embeddings.shape[-1])
        )
        # TODO: model architecture
        if model_name == 'rnn':
            self.model = torch.nn.RNN(input_size=input_size, 
                                      hidden_size=hidden_size, 
                                      num_layers=num_layers,
                                      dropout=dropout_rate,
                                      bidirectional=bidirectional,
                                      batch_first=True)
        elif model_name == 'gru':
            self.model = torch.nn.GRU(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout_rate,
                                      bidirectional=bidirectional,
                                      batch_first=True)
        elif model_name == 'lstm':
            self.model = torch.nn.LSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout=dropout_rate,
                                       bidirectional=bidirectional,
                                       batch_first=True)
        else :
            raise NameError('please choose one of ["rnn", "gru", "lstm"]')

        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, num_class)
        )
        self.device = device
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None :
        def init_weights_method(module, init_method: str, gain: float = 0.02, std: float = 1.0, mean: float = 0.0):
            weights = module.weight.data
            if init_method == 'uniform':
                return nn.init.uniform_(weights)
            elif init_method == 'normal':
                return nn.init.normal_(weights, std=gain)
            elif init_method == 'xavier_uniform':
                return nn.init.xavier_uniform_(weights, gain=1.0)
            elif init_method == 'xavier_normal':
                return nn.init.xavier_normal_(weights, gain=gain)
            elif init_method == 'kaiming_uniform':
                return nn.init.kaiming_uniform_(weights)
            elif init_method == 'kaiming_normal':
                return torch.nn.init.kaiming_normal_(weights)
            elif init_method == 'orthogonal':
                return nn.init.orthogonal_(weights, gain=gain)
            elif init_method == 'identity':
                return nn.init.eye_(weights)

        if isinstance(module, nn.Embedding):
            logging.warning("It's Embedding ! Don't initial weight of embedding")
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Linear) and hasattr(module, 'weight'):
            init_weights_method(module, self.init_method)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        logging.debug(f'input shape = {x.shape}')
        # x -> (Batch, Seq_len)
        embeddings = self.embed(x)
        logging.debug(f'embeddings shape = {embeddings.shape}')
        # embed -> (Seq_len, Batch, embed_dim)
        if isinstance(self.model, nn.LSTM):
            out, (h_n, _) = self.model(embeddings)
        else:
            out, h_n = self.model(embeddings)
        # out -> (Batch, Seq_len, D*hidden_size)
        # h_n -> (D*num_layer, Batch, hidden)
        logging.debug(f'output shape = {out.shape}')
        logging.debug(f'hidden shape = {h_n.shape}')
        last_layer_bi_h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]
        # last_layer_bi_hn -> (2, Batch, hidden)
        logging.debug(f'last_layer_bi_h_n shape = {last_layer_bi_h_n.shape}')
        output_feature = last_layer_bi_h_n.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)
        # output_feature -> (Batch, hidden*2)
        logging.debug(f'output feature shape = {output_feature.shape}')
        prob = self.classifier(output_feature)
        logging.debug(f'prob size = {prob.shape}')
        return prob

class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
