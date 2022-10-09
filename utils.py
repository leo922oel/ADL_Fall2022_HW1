from typing import Iterable, List


class Vocab:
    PAD = "[PAD]"
    UNK = "[UNK]"

    def __init__(self, vocab: Iterable[str]) -> None:
        self.token2idx = {
            Vocab.PAD: 0,
            Vocab.UNK: 1,
            **{token: i for i, token in enumerate(vocab, 2)},
        }

    @property
    def pad_id(self) -> int:
        return self.token2idx[Vocab.PAD]

    @property
    def unk_id(self) -> int:
        return self.token2idx[Vocab.UNK]

    @property
    def tokens(self) -> List[str]:
        return list(self.token2idx.keys())

    def token_to_id(self, token: str) -> int:
        return self.token2idx.get(token, self.unk_id)

    def encode(self, tokens: List[str]) -> List[int]:
        return [self.token_to_id(token) for token in tokens]

    def encode_batch(
        self, batch_tokens: List[List[str]], to_len: int = None
    ) -> List[List[int]]:
        batch_ids = [self.encode(tokens) for tokens in batch_tokens]
        to_len = max(len(ids) for ids in batch_ids) if to_len is None else to_len
        padded_ids = pad_to_len(batch_ids, to_len, self.pad_id)
        return padded_ids


def pad_to_len(seqs: List[List[int]], to_len: int, padding: int) -> List[List[int]]:
    paddeds = [seq[:to_len] + [padding] * max(0, to_len - len(seq)) for seq in seqs]
    return paddeds

class CEMetrics(object):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.loss = 0
        self.sum = 0
        self.n = 0
        self.avg = 0
    
    def udpate(self, loss, n):
        self.loss = loss
        self.sum += loss * n
        self.n += n
        self.avg = self.sum / self.n + self.eps

class IntentMetrics(object):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.correct = 0
        self.n = 0
    
    def update(self, ground, pred):
        n = ground.size(0)
        self.correct += pred.eq(ground.view_as(pred)).sum().item()
        self.n += n

    def eval(self):
        self.acc = self.correct / (self.n + self.eps)

class SlotMetrics(object):
    def __init__(self, eps=1e-8):
        self.eps = eps
        self.tok_cor = 0
        self.joi_cor = 0
        self.tok_n = 0
        self.joi_n = 0
    
    def update(self, ground, pred, mask):
        mask = mask[:, :ground.size(1)]
        batch_cor = (pred.eq(ground.view_as(pred)) * mask).sum(-1)
        len = mask.sum(-1)

        self.tok_cor += batch_cor.sum().long().item()
        self.joi_cor += batch_cor.eq(len).sum().item()
        self.tok_n += mask.sum().long().item()
        self.joi_n += len(ground)

    def eval(self):
        self.tok_acc = self.tok_cor / (self.tok_n + self.eps)
        self.joi_acc = self.joi_cor / (self.joi_n + self.eps)