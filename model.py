from typing import Dict

import torch
from torch.nn import Embedding, LSTM


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
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidrect = bidirectional
        self.num_class = num_class
        self.rnn = LSTM(
            embedding_dim,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=True,
        )


    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        num_direct = 2 if self.bidirectional else 1
        self.output_dim = num_direct * self.hidden_size
        # self.hidden_state_dim = self.num_layers * num_direct * self.hidden_size
        return self.output_dim

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        embedded = self.embed
        output, hidden_state = self.rnn(embedded)
        hidden_state = hidden_state.permute(1, 0, 2) ## ?
        # raise NotImplementedError
        return {output: hidden_state}


class SeqTagger(SeqClassifier):
    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward
        raise NotImplementedError
