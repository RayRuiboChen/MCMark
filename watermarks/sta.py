##!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.nn import functional as F
import time
from typing import Union

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore
import json


class STA_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, list[torch.Generator]],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return cls(shuffle)


class STA_Reweight(AbstractReweight):
    watermark_code_type = STA_WatermarkCode
    # raise NotImplementedError # a little tricky here, because the process includes sampling

    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def __repr__(self):

        return f"STA_Reweight(gamma={self.gamma})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        """
        sample then accept -- one
        """

        # s_ means shuffled
        s_p_logits = torch.gather(p_logits, -1, code.shuffle)

        vocab_size = s_p_logits.shape[-1]

        green_list_size = round(vocab_size * self.gamma)

        dist = torch.distributions.Categorical(probs=torch.softmax(s_p_logits, dim=-1))

        idx1 = dist.sample()
        idx2 = dist.sample()

        final_idx = torch.where(idx1 < green_list_size, idx1, idx2)

        s_modified_logits = torch.where(
            torch.arange(s_p_logits.shape[-1], device=s_p_logits.device)
            == final_idx.unsqueeze(-1),
            torch.full_like(s_p_logits, 0),
            torch.full_like(s_p_logits, float("-inf")),
        )

        modified_logits = torch.gather(s_modified_logits, -1, code.unshuffle)
        return modified_logits
