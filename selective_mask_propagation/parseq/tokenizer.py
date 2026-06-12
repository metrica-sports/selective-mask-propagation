"""PARSeq tokenizer — character-level encoder/decoder.

Inference-only port of strhub.data.utils.Tokenizer. Builds a flat
character vocabulary `[EOS, charset..., BOS, PAD]` and provides:

    - `bos_id` / `eos_id` / `pad_id` for sequence boundary tokens
    - `decode(probs)` — greedy argmax decode of a softmax tensor with
      truncation at the first EOS, returning string labels and the
      per-character probabilities for confidence aggregation

Encode-side functionality (label → token ids) is intentionally omitted:
inference never encodes labels, only decodes model output.
"""

from typing import List, Tuple

from torch import Tensor


class Tokenizer:
    BOS = "[B]"
    EOS = "[E]"
    PAD = "[P]"

    def __init__(self, charset: str) -> None:
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        self._itos: Tuple[str, ...] = specials_first + tuple(charset) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}
        self.eos_id = self._stoi[self.EOS]
        self.bos_id = self._stoi[self.BOS]
        self.pad_id = self._stoi[self.PAD]

    def __len__(self) -> int:
        return len(self._itos)

    def decode(self, token_dists: Tensor) -> Tuple[List[str], List[Tensor]]:
        """Greedy decode, truncate at EOS, return (labels, per-char probs).

        Algorithm:
            For each row of the softmax tensor:
                1. Take argmax per position to get token ids and their
                   probabilities.
                2. Find the first EOS index; truncate the id list to
                   exclude it but keep its probability (so confidence
                   reflects EOS prediction quality too).
                3. Map ids back to characters and join.
        """
        batch_tokens: List[str] = []
        batch_probs: List[Tensor] = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            ids_list = ids.tolist()
            try:
                eos_idx = ids_list.index(self.eos_id)
            except ValueError:
                eos_idx = len(ids_list)
            ids_list = ids_list[:eos_idx]
            probs = probs[: eos_idx + 1]
            batch_tokens.append("".join(self._itos[i] for i in ids_list))
            batch_probs.append(probs)
        return batch_tokens, batch_probs
