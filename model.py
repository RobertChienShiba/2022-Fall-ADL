from typing import Dict
import logging

import torch
import torch.nn as nn
from torch.nn import Embedding

class SeqClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        embeddings,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        pad_id: int,
        bidirectional: bool,
        num_class,
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
        self.device = device

        if isinstance(embeddings, torch.Tensor):
            self.embed = nn.Sequential(
                Embedding.from_pretrained(embeddings, freeze=False, padding_idx=self.pad_id),
                # embedding size = 300
                nn.LayerNorm(embeddings.shape[-1])
            ).to(self.device)
        else:
            self.embed = None

        # TODO: model architecture
        if model_name == 'rnn':
            self.model = nn.RNN(input_size=input_size, 
                                      hidden_size=hidden_size, 
                                      num_layers=num_layers,
                                      dropout=dropout_rate,
                                      bidirectional=bidirectional,
                                      batch_first=True)
        elif model_name == 'gru':
            self.model = nn.GRU(input_size=input_size,
                                      hidden_size=hidden_size,
                                      num_layers=num_layers,
                                      dropout=dropout_rate,
                                      bidirectional=bidirectional,
                                      batch_first=True)
        elif model_name == 'lstm':
            self.model = nn.LSTM(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout=dropout_rate,
                                       bidirectional=bidirectional,
                                       batch_first=True)
        else :
            raise NameError('please choose one of ["rnn", "gru", "lstm"]')

        if isinstance(num_class, int):
            self.classifier = nn.Sequential(
                nn.LayerNorm(hidden_size * 2),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 2, num_class)
            )
        else:
            self.classifier = None
            
        self.apply(self._init_weights)

    def _init_weights(self, module) -> None :
        def init_weights_method(module, init_method: str, gain: float = 0.02):
            weights = module.weight.data
            if init_method == 'normal':
                return nn.init.normal_(weights, std=gain)
            elif init_method == 'xavier_normal':
                return nn.init.xavier_normal_(weights, gain=gain)
            elif init_method == 'kaiming_normal':
                return torch.nn.init.kaiming_normal_(weights)
            elif init_method == 'orthogonal':
                return nn.init.orthogonal_(weights, gain=gain)
            elif init_method == 'identity':
                return nn.init.eye_(weights)

        if isinstance(module, nn.Embedding):
            logging.warning("It's Pretrained Embedding ! Don't initial weight of Embedding")
        elif isinstance(module, nn.Linear):
            init_weights_method(module, self.init_method)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        # TODO: implement model forward
        logging.debug(f'input shape = {x.shape}')
        # x -> (Batch, Seq_len)
        embeddings = self.embed(x)
        logging.debug(f'embeddings shape = {embeddings.shape}')
        # embed -> (Batch, Seq_len, embed_dim)
        if isinstance(self.model, nn.LSTM):
            output, (h_n, _)= self.model(embeddings)
        else:
            output, h_n = self.model(embeddings)
        # out -> (Batch, Seq_len, D*hidden_size)
        # h_n -> (D*num_layer, Batch, hidden)
        logging.debug(f'output shape = {output.shape}')
        logging.debug(f'hidden shape = {h_n.shape}')
        last_layer_h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]
        # last_layer_hn -> (2, Batch, hidden)
        logging.debug(f'last_layer_h_n shape = {last_layer_h_n.shape}')
        output_feature = last_layer_h_n.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)
        # output_feature -> (Batch, hidden*2)
        logging.debug(f'output feature shape = {output_feature.shape}')
        logits = self.classifier(output_feature)
        # logits -> (Batch, num_class)
        logging.debug(f'logits size = {logits.shape}')
        return logits

class SeqTagger(SeqClassifier):
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
        super(SeqTagger, self).__init__(input_size, embeddings, hidden_size, num_layers, dropout_rate, 
                                pad_id, bidirectional, num_class, model_name, init_method, device)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size * (bidirectional + 1)),
            nn.Linear(hidden_size * (bidirectional + 1), num_class)
        )
        self.apply(self._init_weights)

    def forward(self, batch) :
        # TODO: implement model forward
        logging.debug(f'input shape = {batch.shape}')
        # batch -> (Batch, Seq_len)
        embeddings = self.embed(batch)
        logging.debug(f'embeddings shape = {embeddings.shape}')
        # embed -> (Seq_len, Batch, embed_dim)
        if isinstance(self.model, nn.LSTM):
            output, (h_n, _) = self.model(embeddings)
        else:
            output, h_n = self.model(embeddings)
        # out -> (Batch, Seq_len, D*hidden_size)
        # h_n -> (D*num_layer, Batch, hidden)
        logging.debug(f'output shape = {output.shape}')
        logging.debug(f'hidden shape = {h_n.shape}')
        logits = self.classifier(output)
        # logits -> (Batch, Seq_len, num_class)
        logging.debug(f'logits size = {logits.shape}')
        # Match CrossEntropy Loss input dimension
        logits = logits.transpose(1, 2)
        # logits -> (Batch, num_class, Seq_len)
        logging.debug(f'Match CrossEntropy Loss input dimension size = {logits.shape}')

        return logits

class MultitaskNet(SeqClassifier):
    def __init__(
        self,
        input_size: int,
        embeddings: Dict,
        hidden_size: int,
        num_layers: int,
        dropout_rate: float,
        pad_id: int,
        bidirectional: bool,
        num_class: Dict,
        model_name: str,
        init_method: str,
        device: torch.device,
        crf = None
    ) -> None:
        super(MultitaskNet, self).__init__(input_size, embeddings, hidden_size, num_layers, dropout_rate, 
                                pad_id, bidirectional, num_class, model_name, init_method, device)

        self.embed = {task: nn.Sequential(Embedding.from_pretrained(embedding, freeze=False, padding_idx=self.pad_id),
            # embedding size = 300 
            nn.LayerNorm(embedding.shape[-1])).to(self.device) 
            for task, embedding in embeddings.items()
        }
        
        self.icf_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.LayerNorm(hidden_size * (bidirectional + 1)),
            nn.Linear(hidden_size * (bidirectional + 1), num_class['intent'])
        )
        
        self.slt_classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 2),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, num_class['slot'])
        )

        if crf :
            from torchcrf import CRF
            self.crf = CRF(num_tags=num_class['slot'], batch_first=True)
        else:
            self.crf = None

        self.apply(self._init_weights)

    def forward_intent(self, h_n):
        last_layer_h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]
        # last_layer_hn -> (2, Batch, hidden)
        logging.debug(f'last_layer_h_n shape = {last_layer_h_n.shape}')
        output_feature = last_layer_h_n.permute(1, 0, 2).reshape(-1, 2 * self.hidden_size)
        # output_feature -> (Batch, hidden*2)
        logging.debug(f'output feature shape = {output_feature.shape}')
        logits = self.icf_classifier(output_feature)
        # logits -> (Batch, num_class)
        logging.debug(f'logits size = {logits.shape}')
        return logits

    def forward_slot(self, output):
        logits = self.slt_classifier(output)
        # logits -> (Batch, Seq_len, num_class)
        logging.debug(f'logits size = {logits.shape}')
        # Match CrossEntropy Loss input dimension
        logits = logits.transpose(1, 2)
        # logits -> (Batch, num_class, Seq_len)
        logging.debug(f'Match CrossEntropy Loss input dimension size = {logits.shape}')

        return logits

    def forward(self, x, task: str):
        logging.debug(f'input shape = {x.shape}')
        # x -> (Batch, Seq_len)
        embeddings = self.embed[task](x)
        logging.debug(f'embeddings shape = {embeddings.shape}')
        # embed -> (Seq_len, Batch, embed_dim)
        if isinstance(self.model, nn.LSTM):
            output, (h_n, _) = self.model(embeddings)
        else:
            output, h_n = self.model(embeddings)
        # out -> (Batch, Seq_len, D*hidden_size)
        # h_n -> (D*num_layer, Batch, hidden)
        logging.debug(f'output shape = {output.shape}')
        logging.debug(f'hidden shape = {h_n.shape}')
        if task == 'intent':
            return self.forward_intent(h_n)
        elif task == 'slot':
            return self.forward_slot(output)
        else:
            raise NameError(f'YOU CHOOSE THE WRONG {task} !!! please choose one of ["intent", "slot"] task')