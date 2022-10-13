from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Embedding, LSTM, Sequential, Dropout, Linear
from torch.nn import Conv1d, ReLU
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.embed_dim = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidrect = bidirectional
        self.num_class = num_class
        self.rnn = LSTM(
            self.embed_dim,
            hidden_size,
            num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True,
        )
        self.classifier = Sequential(
            Dropout(dropout),
            Linear(self.encoder_output_size, num_class)
        )


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        if self.bidrect: return 2*self.hidden_size
        return self.hidden_size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output = {}
        X, y = batch["text"], batch["intent"]
        mask = (X.gt(0)).float()

        X = self.embed(X)
        pack_X = pack_padded_sequence(X, batch["len"], batch_first=True)
        self.rnn.flatten_parameters()
        X, (h_, c_) = self.rnn(pack_X)
        X, _ = pad_packed_sequence(X, batch_first=True)

        if self.bidrect: h_ = torch.cat((h_[-1], h_[-2]), axis=-1)
        else: h_ = h_[-1]

        pred_logits = [self.classifier(h_)]

        output["pred_logists"] = pred_logits
        output["pred_labels"] = pred_logits[-1].max(1, keepdim=True)[1].reshape(-1)
        output["loss"] = F.cross_entropy(pred_logits[-1], y) # ? y.long()

        return output


class SeqTagger(SeqClassifier):
    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
    ) -> None:
        super(SeqTagger, self).__init__(embeddings, hidden_size, num_layers, dropout, bidirectional, num_class)
        # self.embed = Embedding.from_pretrained(embeddings, freeze=False)
        # self.embed_dim = embeddings.size(1)
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.dropout = dropout
        # self.bidrect = bidirectional
        # self.num_class = num_class

        cnn = []
        for i in range(num_layers):
            conv = Sequential(
                Conv1d(self.embed_dim, self.embed_dim, 5, 1, 2),
                ReLU(),
                Dropout(dropout)
            )
            cnn.append(conv)
        self.cnn = nn.ModuleList(cnn)

        self.rnn = LSTM(
            self.embed_dim,
            self.hidden_size,
            self.num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
            )
        
        self.tag_classifier = Sequential(
            Dropout(dropout),
            Linear(self.encoder_output_size, self.num_class)
        )

    # @property
    # def encoder_output_size(self) -> int:
        # if self.num_layers <= 0: return self.embed_dim
        # if self.bidrect: return 2*self.hidden_size
        # return self.hidden_size
    
    def _get_idx(self, tokens_len):
        batch_idx = torch.cat([torch.full((len, ), i) for i, len in enumerate(tokens_len)])
        tokens_idx = torch.cat([torch.arange(0, len) for len in tokens_len])
        return batch_idx, tokens_idx

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        output = {}

        X, y = batch["tokens"], batch["tags"]
        X = self.embed(X)

        # For CNN
        X = X.permute(0, 2, 1)
        for conv in self.cnn:
            X = conv(X)
        X = X.permute(0, 2, 1)

        # For RNN
        pack_X = pack_padded_sequence(X, batch["len"], batch_first=True)
        self.rnn.flatten_parameters()
        X, _ = self.rnn(pack_X)
        X, _ = pad_packed_sequence(X, batch_first=True)

        batch["mask"] = batch["mask"][:, :X.size(1)]
        batch["tags"] = batch["tags"][:, :X.size(1)]

        pred_logits = self.tag_classifier(X)
        idx = self._get_idx(batch["len"])
        output["loss"] = F.cross_entropy(pred_logits[idx], y[idx]) # ? y.long()

        output["pred_logists"] = pred_logits
        output["pred_labels"] = pred_logits.max(-1, keepdim=True)[1].squeeze(2)

        return output