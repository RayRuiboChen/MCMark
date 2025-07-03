#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from .base import (
    AbstractReweight,
    AbstractContextCodeExtractor,
    AbstractScore,
    AbstractWatermarkKey,
)
from typing import List
from .dipmark import Dip_Reweight
from .mcmark import MC_Reweight
from .sta import STA_Reweight
from .unigram import Unigram_Reweight
import json


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight,  # sample strategy
        watermark_key_list: List[AbstractWatermarkKey],
    ):
        self.watermark_key_list = watermark_key_list
        self.private_key = private_key
        self.reweight = reweight

    def __repr__(self):
        watermark_str = ", ".join(
            [repr(watermark_key) for watermark_key in self.watermark_key_list]
        )

        res_str = f"WatermarkLogitsProcessor(private_key={repr(self.private_key)}, reweight={repr(self.reweight)}, watermark_key_list=[{watermark_str}])"

        return res_str

    def get_rng_seed(self, key_list) -> any:
        import hashlib

        m = hashlib.sha256()
        # m.update(self.private_key)
        for key in key_list:
            m.update(key)
        full_hash = m.digest()
        seed = int.from_bytes(full_hash, "big") % (2**32 - 1)
        return seed

    def reset_watermark_key(self, batch_size):
        for watermark_key in self.watermark_key_list:
            watermark_key.reset(batch_size)

    def _get_codes(self, input_ids: LongTensor):
        batch_size = input_ids.size(0)

        mask = []
        seeds = []
        for batch_idx in range(batch_size):
            cur_mask = 0
            key_list = [self.private_key]
            for watermark_key in self.watermark_key_list:
                cur_wm_mask, cur_wm_key = watermark_key.generate_key_and_mask(
                    input_ids[batch_idx], batch_idx
                )
                if cur_wm_key is not None:
                    key_list.append(cur_wm_key)
                cur_mask = cur_mask or cur_wm_mask
            mask.append(cur_mask)
            seeds.append(self.get_rng_seed(key_list))

        return mask, seeds

    def _core(self, input_ids: LongTensor, scores: FloatTensor):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=scores.device).manual_seed(seed) for seed in seeds
        ]
        mask = torch.tensor(mask, device=scores.device, dtype=torch.bool)

        if isinstance(self.reweight, MC_Reweight):
            watermark_code = self.reweight.watermark_code_type.from_random(
                rng, scores.size(1), self.reweight.n
            )
        else:
            watermark_code = self.reweight.watermark_code_type.from_random(
                rng, scores.size(1)
            )
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores)
        return mask, reweighted_scores

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        mask, reweighted_scores = self._core(input_ids, scores)
        return torch.where(mask[:, None], scores, reweighted_scores)

    def get_green_token_quantile(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, Dip_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        # calculate the score here
        token_quantile = [
            (torch.where(watermark_code.shuffle[i] == current_token[i])[0] + 1)
            / vocab_size
            for i in range(input_ids.shape[0])
        ]

        return token_quantile

    def get_sta_score(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, STA_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        green_list_size = round(self.reweight.gamma * vocab_size)
        scores = [
            torch.tensor(
                current_token[i] in watermark_code.shuffle[i][:green_list_size]
            ).float()
            for i in range(input_ids.shape[0])
        ]

        return scores

    def get_unigram_score(
        self, input_ids: LongTensor, vocab_size, current_token, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, Unigram_Reweight)
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(rng, vocab_size)

        green_list_size = round(self.reweight.gamma * vocab_size)
        scores = [
            torch.tensor(
                current_token[i] in watermark_code.shuffle[i][:green_list_size]
            ).float()
            for i in range(input_ids.shape[0])
        ]

        return scores

    def get_n_res(
        self, input_ids: LongTensor, vocab_size, current_token, cur_n, debug=False
    ):
        mask, seeds = self._get_codes(input_ids)
        rng = [
            torch.Generator(device=input_ids.device).manual_seed(seed) for seed in seeds
        ]
        assert isinstance(self.reweight, MC_Reweight)
        assert self.reweight.n == cur_n
        mask = torch.tensor(mask, device=input_ids.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, vocab_size, self.reweight.n
        )

        # cur_n=32000
        splits = []
        if vocab_size % cur_n == 0:
            splits = (
                torch.arange(start=0, end=vocab_size)
                .reshape(cur_n, vocab_size // cur_n)
                .to(input_ids.device)
            )
        else:
            for n_idx in range(cur_n):
                splits.append(
                    list(
                        range(
                            round(vocab_size * n_idx / cur_n),
                            round(vocab_size * (n_idx + 1) / cur_n),
                        )
                    )
                )

        scores = []
        for bsz_idx in range(input_ids.shape[0]):
            cur_k = watermark_code.split_k[bsz_idx]
            if current_token[bsz_idx] in watermark_code.shuffle[bsz_idx][splits[cur_k]]:
                scores.append(1)
            else:
                scores.append(0)
        return scores


class WatermarkLogitsProcessor_Baseline(LogitsProcessor):
    def __repr__(self):
        return f"WatermarkLogitsProcessor_Baseline()"

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        return scores
