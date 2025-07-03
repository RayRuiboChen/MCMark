# TODO: implement unigram

##!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor, BoolTensor
from torch.nn import functional as F
import time
from typing import Union

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore
import json


class Unigram_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: Union[torch.Generator, list[torch.Generator]],
        vocab_size: int,
    ):
        # Note: the rng in the parameter is not used
        if isinstance(rng, list):
            batch_size = len(rng)

            new_rng = []
            for i in range(batch_size):
                fixed_rng = torch.Generator(device=rng[i].device)
                fixed_rng.manual_seed(42)
                new_rng.append(fixed_rng)
            shuffle = torch.stack(
                [
                    torch.randperm(
                        vocab_size, generator=new_rng[i], device=new_rng[i].device
                    )
                    for i in range(batch_size)
                ]
            )
        else:
            fixed_rng = torch.Generator(device=rng.device)
            fixed_rng.manual_seed(42)
            shuffle = torch.randperm(
                vocab_size, generator=fixed_rng, device=fixed_rng.device
            )
        return cls(shuffle)


class Unigram_Reweight(AbstractReweight):
    watermark_code_type = Unigram_WatermarkCode
    # raise NotImplementedError # a little tricky here, because the process includes sampling

    def __init__(self, delta, gamma=0.5):
        self.gamma = gamma
        self.delta = delta

    def __repr__(self):

        return f"Unigram_Reweight(delta={self.delta},gamma={self.gamma})"

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:

        # s_ means shuffled
        s_p_logits = torch.gather(p_logits, -1, code.shuffle)

        vocab_size = s_p_logits.shape[-1]

        green_list_size = round(vocab_size * self.gamma)

        s_p_logits[:, :green_list_size] += self.delta

        modified_logits = torch.gather(s_p_logits, -1, code.unshuffle)
        return modified_logits
