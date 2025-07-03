import tqdm
from datasets import Dataset
import pandas as pd
import json
from .generation_dataset import (
    get_mmw_book_report_prompts,
    get_mmw_fake_news_prompts,
    get_mmw_story_prompts,
    get_dolly_cw,
    get_wb_2_1,
    get_wb_2_2,
)
from . import get_output
from . import evaluate_watermark_score


def get_in_ds_undetectable_exp(dataset_name="c4_subset"):
    def truncate_text(text, word_num):
        assert len(text.split(" ")) > word_num
        return " ".join(text.split(" ")[:word_num]), " ".join(
            text.split(" ")[word_num:]
        )

    print("generating text generation dataset...", flush=True)

    if dataset_name == "c4_subset":
        dataset_path = "dataset/c4_subset.json"
        with open(dataset_path, "r") as f:
            c4_subset = json.load(f)

        import random

        random.seed(43)
        random.shuffle(c4_subset)
        prompt_num = 1000  # randomly sample 1000 prompts
        truncate_num = 100
        id_list = range(prompt_num)
        ds_subset = []
        for prompt_idx in range(prompt_num):
            id_idx = prompt_num + prompt_idx
            new_item = {}
            new_input, new_reference = truncate_text(
                c4_subset[prompt_idx], word_num=truncate_num
            )
            new_item["input"] = (
                "Help me complete the following text with at least 500 words:\n\n"
                + new_input
            )
            new_item["reference"] = new_reference
            new_item["id"] = id_list[id_idx]
            ds_subset.append(new_item)

    else:
        if dataset_name == "mmw_book_report":
            instructions = get_mmw_book_report_prompts()
        elif dataset_name == "mmw_story":
            instructions = get_mmw_story_prompts()
        elif dataset_name == "mmw_fake_news":
            instructions = get_mmw_fake_news_prompts()
        elif dataset_name == "dolly_cw":
            instructions = get_dolly_cw()
        elif dataset_name == "longform_qa":
            instructions = get_wb_2_1()
        elif dataset_name == "finance_qa":
            instructions = get_wb_2_2()
        else:
            print("Unknown dataset_name: ", dataset_name)
            raise NotImplementedError

        ds_subset = []
        prompt_num = len(instructions)
        for prompt_idx in range(prompt_num):
            new_item = {}
            new_item["input"] = instructions[prompt_idx]
            new_item["reference"] = ""
            new_item["id"] = prompt_idx
            ds_subset.append(new_item)

    ds_subset = pd.DataFrame(ds_subset)
    ds_subset = Dataset.from_pandas(ds_subset, preserve_index=False)

    ds_subset = ds_subset.sort("id")
    return ds_subset
