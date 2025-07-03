from .base import AbstractWatermarkKey, AbstractContextCodeExtractor
import torch


class NGramHashing(AbstractWatermarkKey):
    def __init__(
        self, context_code_extractor: AbstractContextCodeExtractor, ignore_history: bool
    ) -> None:
        self.context_code_extractor = context_code_extractor
        self.ignore_history = ignore_history
        self.cc_history = []

    def __repr__(self) -> str:
        return f"NGramHashing(context_code_extractor={repr(self.context_code_extractor)},ignore_history={self.ignore_history})"

    def reset(self, batch_size):
        self.cc_history = [set() for _ in range(batch_size)]

    def generate_key_and_mask(self, input_id, batch_idx):
        context_code = self.context_code_extractor.extract(input_id)

        mask = context_code in self.cc_history[batch_idx]
        if not self.ignore_history:
            self.cc_history[batch_idx].add(context_code)
        return mask, context_code
