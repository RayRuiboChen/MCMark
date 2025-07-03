def get_wps(reweight_type, model_str):
    from watermarks import (
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
        WatermarkLogitsProcessor_Baseline,
        NGramHashing,
        Dip_Reweight,
        MC_Reweight,
        STA_Reweight,
        Unigram_Reweight,
    )

    from ..lm_watermarking.watermark_processor import (
        WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
    )

    import random
    import copy

    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")

    watermark_key_list = [
        NGramHashing(PrevN_ContextCodeExtractor(2), ignore_history=False)
    ]

    reweight_list = []

    if reweight_type == "main_exp":
        reweight_list = [
            Unigram_Reweight(delta=0.5, gamma=0.5),
            Unigram_Reweight(delta=1.0, gamma=0.5),
            Unigram_Reweight(delta=1.5, gamma=0.5),
            Unigram_Reweight(delta=2.0, gamma=0.5),
            Dip_Reweight(alpha=0.5),
            Dip_Reweight(alpha=0.4),
            Dip_Reweight(alpha=0.3),
            STA_Reweight(gamma=0.5),
            MC_Reweight(20),
        ]

    elif reweight_type == "mcmark_ablation":
        n_list = {
            "meta-llama/Llama-2-7b-chat-hf": [
                2,
                3,
                4,
                5,
                10,
                20,
                50,
                100,
                200,
                500,
                1000,
                2000,
                4000,
                8000,
                16000,
                32000,
            ],
            "meta-llama/Llama-3.2-3B-Instruct": [
                2,
                3,
                4,
                5,
                10,
                20,
                50,
                100,
                167,
                501,
                1002,
                2004,
                4008,
                8016,
                16032,
                32064,
                64128,
                128256,
            ],
            "mistralai/Mistral-7B-Instruct-v0.3": [
                2,
                3,
                4,
                5,
                8,
                16,
                20,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
            ],
            "microsoft/Phi-3.5-mini-instruct": [
                2,
                3,
                4,
                5,
                10,
                20,
                50,
                167,
                501,
                1002,
                2004,
                4008,
                8016,
                16032,
                32064,
            ],
        }
        reweight_list = [MC_Reweight(n) for n in n_list[model_str]]
    elif reweight_type == "mcmark":
        reweight_list = [MC_Reweight(20)]
    else:
        raise ValueError(f"Unknown reweight_type: {reweight_type}")

    wm_wps = []

    for wm_key in watermark_key_list:
        for reweight in reweight_list:
            wm_wps.append(
                WatermarkLogitsProcessor(
                    private_key,
                    reweight=copy.deepcopy(reweight),
                    watermark_key_list=[copy.deepcopy(wm_key)],
                )
            )

    if reweight_type == "main_exp":
        john_wps = [
            WatermarkLogitsProcessor_John(
                vocab_size=0,  # placeholder
                gamma=0.5,
                delta=delta,
                seeding_scheme="simple_1",
            )
            for delta in [0.5, 1.0, 1.5, 2.0]
        ]
        return [*wm_wps, *john_wps]
    else:
        return [*wm_wps]


def get_num_gpus():
    import torch

    num_gpus = torch.cuda.device_count()
    return num_gpus


def batched_wp_task_worker(
    tq, get_in_ds, reweight_type, dataset_name, model_str, batch_size=8
):
    ds = get_in_ds(dataset_name=dataset_name)

    from .common import get_wps

    wps = get_wps(reweight_type=reweight_type, model_str=model_str)

    from tqdm import tqdm

    for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
        for wp in wps:
            tq.put({"batch": batch, "watermark_processor": wp})


def merged_task_worker(get_in_ds, output_filepath, tq, batch_size=8, dataset_name=None):
    in_ds = get_in_ds(dataset_name=dataset_name)

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": output_filepath})["test"]
    out_ds = out_ds.sort("id")

    dss, wps = add_reference(in_ds, out_ds)

    from tqdm import tqdm

    for ds, wp_str in zip(dss, wps):
        for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
            tq.put(batch)


def log(line: dict, f):
    import json

    f.write(json.dumps(line))
    f.write("\n")
    f.flush()


def simple_store_worker(path, rq, rqe):
    import os

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    from queue import Empty

    with open(path, "w") as f:
        while not (rqe.is_set() and rq.empty()):
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            assert isinstance(result, dict)
            if result == {}:
                continue
            if isinstance(next(iter(result.values())), list):
                assert all([isinstance(v, list) for v in result.values()])
                lens = [len(v) for v in result.values()]
                assert all([l == lens[0] for l in lens])
                for i in range(lens[0]):
                    log({k: v[i] for k, v in result.items()}, f)
            else:
                log(result, f)


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}


from typing import Union


def tokenize_batch(
    example,
    tokenizer,
    fields=["input"],
    max_length: Union[int, dict] = 512,
    task_template: Union[str, dict] = {},
    padding_side={},
):
    # FIXME: repair the task template
    if isinstance(task_template, str):
        #  like "{input}"
        task_template = {"input": task_template}
    result = {}
    tokenizer.pad_token = tokenizer.eos_token

    for field in fields:
        if field in example:
            kwargs = {}
            if isinstance(max_length, dict):
                kwargs["max_length"] = max_length[field]
            else:
                kwargs["max_length"] = max_length
            if field in task_template:
                texts = [
                    task_template[field].format(**{field: s}) for s in example[field]
                ]
            else:
                texts = example[field]
            if field in ["output", "reference"]:
                kwargs["text_target"] = texts
            else:
                kwargs["text"] = texts
            if field == "output":
                kwargs["add_special_tokens"] = False
            if field in padding_side:
                kwargs["padding_side"] = padding_side[field]

            if field == "input":
                new_input = []
                for cur_input in kwargs["text"]:
                    messages = [{"role": "user", "content": cur_input}]
                    warpped_input = tokenizer.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    new_input.append(warpped_input)
                kwargs["text"] = new_input

            result[field] = tokenizer(
                **kwargs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

    return result


def set_spawn():
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def remove_tailing_pad_s(s: str):
    while s.endswith("<pad>"):
        s = s[:-5]
    return s


def remove_tailing_pad(strs: list[str]):
    return [remove_tailing_pad_s(s) for s in strs]


def transformer_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    model_str,
    generation_kwargs={},
    decoder_only=False,
    tokenization_kwargs={},
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import AutoModelForCausalLM
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from watermarks import patch_model

    print("start loading model...", flush=True)
    if decoder_only:
        model = AutoModelForCausalLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    from queue import Empty

    model.eval()
    with torch.no_grad():
        while not (tqe.is_set() and tq.empty()):
            try:
                task = tq.get(timeout=1)
            except Empty as e:
                continue
            batch = task["batch"]
            tbatch = tokenize_batch(batch, tokenizer, **tokenization_kwargs)
            wp = task["watermark_processor"]
            lps = []
            if wp is not None:
                if "reset_watermark_key" in dir(wp):
                    batch_size = len(batch["id"])
                    wp.reset_watermark_key(batch_size)
                if "vocab_size" in dir(wp):
                    wp.vocab_size = model.config.vocab_size

                lps.append(wp)

            # for reproducibility and sufficient randomness
            import hashlib

            hash = hashlib.sha256()
            hash.update((str(batch["id"]) + repr(wp)).encode("utf-8"))
            seed = hash.digest()
            seed = int.from_bytes(seed, "big") % (2**32 - 1)

            set_seed(seed)

            outputs_ids = model.generate(
                tbatch["input"]["input_ids"].to(device=model.device),
                attention_mask=tbatch["input"]["attention_mask"].to(
                    device=model.device
                ),
                # temperature=1,
                do_sample=True,
                num_beams=1,
                top_k=0,
                top_p=1,
                logits_warper=LogitsProcessorList(lps),
                **generation_kwargs,
            )

            if decoder_only:
                outputs_ids = outputs_ids[:, tbatch["input"]["input_ids"].shape[1] :]
            outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=False)
            outputs = remove_tailing_pad(outputs)
            display_outputs = tokenizer.batch_decode(
                outputs_ids, skip_special_tokens=True
            )
            wp_str = repr(wp)
            rq.put(
                {
                    "output": outputs,
                    "display_output": display_outputs,
                    "output_ids": outputs_ids.tolist(),
                    "id": batch["id"],
                    "watermark_processor": [wp_str] * len(outputs),
                }
            )


def add_reference(in_ds, out_ds):
    """assuming ordered by ids"""
    wp_types = set(out_ds["watermark_processor"])

    s_out_dss = []
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        s_out_ds = s_out_ds.add_column("input", in_ds["input"])
        s_out_ds = s_out_ds.add_column("reference", in_ds["reference"])
        s_out_dss.append(s_out_ds)

    return s_out_dss, wp_types


def remove_text_worker(tq, tqe, rq):
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        for f in ["input", "output", "reference", "display_output"]:
            if f in batch:
                del batch[f]
        rq.put(batch)


import torch


def random_paraphrase(input_ids, vocab_size, device, eps):
    import hashlib

    hash = hashlib.sha256()
    # hash.update(str(batch["id"]).encode("utf-8"))
    hash.update(str(input_ids).encode("utf-8"))
    seed = hash.digest()
    seed = int.from_bytes(seed, "big") % (2**32 - 1)
    torch.manual_seed(seed)

    modified_input_ids = torch.where(
        torch.rand(input_ids.shape).to(device) > eps,
        input_ids,
        torch.randint_like(input_ids, low=0, high=vocab_size).to(device),
    )

    return modified_input_ids


@torch.no_grad()
def get_sta_score_id(vocab_size, output_ids, wp, device, eps=0):
    assert eps <= 1
    assert eps >= 0

    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(output_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(
            decoder_input_ids, vocab_size, device, eps
        )

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_sta_score(pre, vocab_size, cur_token)
        # try:
        scores[:, i + 1] = torch.stack(out).reshape(-1)
    return scores, label_attention_mask


@torch.no_grad()
def get_unigram_score_id(vocab_size, output_ids, wp, device, eps=0):
    assert eps <= 1
    assert eps >= 0

    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(
            decoder_input_ids, vocab_size, device, eps
        )

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_unigram_score(pre, vocab_size, cur_token)
        scores[:, i + 1] = torch.stack(out).reshape(-1)

    return scores, label_attention_mask


# for Dipmark and gamma-reweight
@torch.no_grad()
def get_quantile_id(vocab_size, output_ids, wp, device, eps=0):
    assert eps <= 1
    assert eps >= 0

    decoder_input_ids = output_ids.to(device)

    label_attention_mask = torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(
            decoder_input_ids, vocab_size, device, eps
        )

    quantile = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_green_token_quantile(pre, vocab_size, cur_token)
        quantile[:, i + 1] = torch.stack(out).reshape(-1)

    return quantile, label_attention_mask


# for Splitmark
@torch.no_grad()
def get_split_res_id(vocab_size, output_ids, wp, device, eps=0, split_num=None):
    assert eps <= 1
    assert eps >= 0
    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(decoder_input_ids).to(device)

    if eps > 0:
        decoder_input_ids = random_paraphrase(
            decoder_input_ids, vocab_size, device, eps
        )

    scores = torch.zeros(decoder_input_ids.shape, device=device)
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[:, : i + 1]
        cur_token = decoder_input_ids[:, i + 1]
        out = wp.get_n_res(pre, vocab_size, cur_token, cur_n=split_num)
        scores[:, i + 1] = torch.tensor(out).reshape(-1)
    return scores, label_attention_mask


from ..lm_watermarking.watermark_processor import (
    WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
)


@torch.no_grad()
def get_green_token_scores_id(vocab_size, output_ids, wp, device, eps=0):
    assert eps <= 1
    assert eps >= 0

    decoder_input_ids = output_ids.to(device)
    label_attention_mask = torch.ones_like(decoder_input_ids)

    if eps > 0:
        decoder_input_ids = random_paraphrase(
            decoder_input_ids, vocab_size, device, eps
        )

    scores = torch.zeros(decoder_input_ids.shape, device=device)

    assert decoder_input_ids.shape[0] == 1
    for i in range(decoder_input_ids.size(1) - 1):
        pre = decoder_input_ids[0, : i + 1]
        cur_token = decoder_input_ids[0, i + 1]
        assert wp.select_green_tokens
        green_token_ids = wp._get_greenlist_ids(pre)
        if cur_token in green_token_ids:
            scores[0, i + 1] = 1
    return scores, label_attention_mask


def watermark_score_worker(
    tq,
    tqe,
    rq,
    gpu_id,
    oracle_model_str,
    decoder_only=False,
    eps=0,
    tokenization_kwargs={},
):
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        set_seed,
        AutoModelForCausalLM,
    )
    from transformers import LogitsProcessorList, TemperatureLogitsWarper
    from queue import Empty

    if decoder_only:
        model = AutoModelForCausalLM.from_pretrained(oracle_model_str).to(
            f"cuda:{gpu_id}"
        )
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(oracle_model_str).to(
            f"cuda:{gpu_id}"
        )

    tokenizer = AutoTokenizer.from_pretrained(oracle_model_str)
    vocab_size = model.config.vocab_size
    del model

    device = f"cuda:{gpu_id}"

    def score_func(quantiles, lens):
        """
        params:
            quantiles: [batch_size, max_len]
        """

        return torch.sum(quantiles > 0.5, dim=-1), lens / 2

    from watermarks import (
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
        NGramHashing,
        Dip_Reweight,
        MC_Reweight,
        STA_Reweight,
        Unigram_Reweight,
    )
    from ..lm_watermarking.watermark_processor import (
        WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
    )

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        assert len(set(batch["watermark_processor"])) == 1

        wp_str = batch["watermark_processor"][0]
        tbatch = tokenize_batch(
            batch, tokenizer, ["input", "output"], **tokenization_kwargs
        )

        output_ids = tbatch["output"]["input_ids"]

        if "MC_Reweight" in wp_str:  # Mc-mark
            wp = eval(wp_str)
            wp.reset_watermark_key(len(batch["watermark_processor"]))
            wp.ignore_history = True

            import re

            def extract_n_value(text):
                pattern = re.search(r"MC_Reweight\(n=(\d+)\)", text)
                if pattern:
                    return int(pattern.group(1))
                return None

            cur_n = extract_n_value(wp_str)
            assert cur_n is not None
            scores, label_attention_mask = get_split_res_id(
                vocab_size, output_ids, wp, device, eps=eps, split_num=cur_n
            )

            assert label_attention_mask.shape[0] == 1
            label_attention_mask[0, :2] = 0

            scores = scores * label_attention_mask
            raw_scores = scores.sum(dim=-1)
            seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)

            rq.put(
                {
                    **batch,
                    "lens": seq_len.cpu().tolist(),
                    "raw_scores": raw_scores.cpu().tolist(),
                }
            )

        elif "John" in wp_str:  # KGW
            wp = eval(wp_str)
            scores, label_attention_mask = get_green_token_scores_id(
                vocab_size, output_ids, wp, device, eps=eps
            )

            assert label_attention_mask.shape[0] == 1
            label_attention_mask[0, :2] = 0

            scores = scores * label_attention_mask
            seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
            raw_scores = scores.sum(dim=-1)
            rq.put(
                {
                    **batch,
                    "lens": seq_len.cpu().tolist(),
                    "raw_scores": raw_scores.cpu().tolist(),
                }
            )
        elif "Unigram" in wp_str:  # Unigram
            wp = eval(wp_str)
            wp.reset_watermark_key(len(batch["watermark_processor"]))
            wp.ignore_history = True

            scores, label_attention_mask = get_unigram_score_id(
                vocab_size, output_ids, wp, device, eps=eps
            )

            assert label_attention_mask.shape[0] == 1
            label_attention_mask[0, :2] = 0  # for fair comparision

            scores = scores * label_attention_mask
            seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
            raw_scores = scores.sum(dim=-1)
            rq.put(
                {
                    **batch,
                    "lens": seq_len.cpu().tolist(),
                    "raw_scores": raw_scores.cpu().tolist(),
                }
            )

        elif "STA_Reweight" in wp_str:  # STA-1
            wp = eval(wp_str)
            wp.reset_watermark_key(len(batch["watermark_processor"]))
            wp.ignore_history = True

            scores, label_attention_mask = get_sta_score_id(
                vocab_size, output_ids, wp, device, eps=eps
            )

            assert label_attention_mask.shape[0] == 1
            label_attention_mask[0, :2] = 0  # for fair comparision

            scores = scores * label_attention_mask
            seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)
            raw_scores = scores.sum(dim=-1)
            rq.put(
                {
                    **batch,
                    "lens": seq_len.cpu().tolist(),
                    "raw_scores": raw_scores.cpu().tolist(),
                }
            )

        elif "Dip_" in wp_str:  # Dipmakr

            wp = eval(wp_str)
            wp.reset_watermark_key(len(batch["watermark_processor"]))
            wp.ignore_history = True

            quantiles, label_attention_mask = get_quantile_id(
                vocab_size, output_ids, wp, device, eps=eps
            )
            assert label_attention_mask.shape[0] == 1

            label_attention_mask[0, :2] = 0

            quantiles = quantiles * label_attention_mask
            cum_label_attention_mask = torch.cumsum(label_attention_mask, dim=-1)
            lens = cum_label_attention_mask[:, -1]

            raw_score, expected_value = score_func(quantiles, lens)
            seq_len = torch.sum(label_attention_mask, dim=-1, keepdim=False)

            assert torch.all(seq_len == lens)
            final_score = (raw_score - expected_value) / torch.sqrt(seq_len)

            rq.put(
                {
                    **batch,
                    "lens": lens.cpu().tolist(),
                    "beta_score": final_score.cpu().tolist(),
                }
            )

        else:
            print(f"Unknown Watermark Processor: {wp_str}")
            raise NotImplementedError
