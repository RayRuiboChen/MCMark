import os
import json
import argparse
import multiprocessing
import openai

import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tqdm
import nltk
from nltk.tokenize import sent_tokenize

nltk.download("punkt_tab")


class DipperParaphraser(object):
    def __init__(self, model="kalpeshk2011/dipper-paraphraser-xxl", verbose=True):
        time1 = time.time()
        self.tokenizer = T5Tokenizer.from_pretrained("google/t5-v1_1-xxl")
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        if verbose:
            print(f"{model} model loaded in {time.time() - time1}")
        self.model.cuda()
        self.model.eval()

    def paraphrase(
        self,
        input_text,
        lex_diversity,
        order_diversity,
        prefix="",
        sent_interval=3,
        **kwargs,
    ):
        """Paraphrase a text using the DIPPER model.

        Args:
            input_text (str): The text to paraphrase. Make sure to mark the sentence to be paraphrased between <sent> and </sent> blocks, keeping space on either side.
            lex_diversity (int): The lexical diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            order_diversity (int): The order diversity of the output, choose multiples of 20 from 0 to 100. 0 means no diversity, 100 means maximum diversity.
            **kwargs: Additional keyword arguments like top_p, top_k, max_length.
        """
        assert lex_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Lexical diversity must be one of 0, 20, 40, 60, 80, 100."
        assert order_diversity in [
            0,
            20,
            40,
            60,
            80,
            100,
        ], "Order diversity must be one of 0, 20, 40, 60, 80, 100."

        lex_code = int(100 - lex_diversity)
        order_code = int(100 - order_diversity)

        input_text = " ".join(input_text.split())
        sentences = sent_tokenize(input_text)
        prefix = " ".join(prefix.replace("\n", " ").split())
        output_text = ""

        for sent_idx in range(0, len(sentences), sent_interval):
            curr_sent_window = " ".join(sentences[sent_idx : sent_idx + sent_interval])
            final_input_text = f"lexical = {lex_code}, order = {order_code}"
            if prefix:
                final_input_text += f" {prefix}"
            final_input_text += f" <sent> {curr_sent_window} </sent>"

            final_input = self.tokenizer([final_input_text], return_tensors="pt")
            final_input = {k: v.cuda() for k, v in final_input.items()}

            with torch.inference_mode():
                outputs = self.model.generate(**final_input, **kwargs)
            outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            prefix += " " + outputs[0]
            output_text += " " + outputs[0]

        return output_text


def query_gpt(prompt):
    seed = 42
    model_name = "gpt-3.5-turbo"  # Currently points to gpt-3.5-turbo-0125.
    temperature = 0
    max_tokens = 1000
    response = None
    success = 1
    try:
        output = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            seed=seed,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = output.choices[0].message.content

    except Exception as e:
        print("-" * 80, flush=True)
        # print(f"Caught error querying with unique idx: {unique_idx}", flush=True)
        print("Error info: ", e, flush=True)
        success = 0

    return success, response


def gpt_paraphrase_attack(attack_args):
    prompt = attack_args["prompt"]
    max_try = attack_args["max_try"]

    if len(prompt) == 0:
        return 0, None

    for idx in range(max_try):
        success, response = query_gpt(prompt)
        if success:
            break
    return success, response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--attack_type", type=str)
    parser.add_argument("--openai_api_key", type=str)
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--dataset_name", type=str)
    parser.add_argument("--model_str", type=str)
    parser.add_argument("--reweight_type", type=str)

    args = parser.parse_args()

    sub_dir = os.path.join(
        args.res_dir,
        args.dataset_name,
        args.model_str.split("/")[-1].replace("-", "_"),
        args.reweight_type,
    )
    input_file = os.path.join(sub_dir, "text_generation.txt")
    assert os.path.exists(input_file)
    with open(input_file, "r") as f:
        lines = f.readlines()

    if args.attack_type == "gpt_rephrase":
        save_path = os.path.join(sub_dir, "text_generation_gpt_rephrase.txt")
        openai_api_key = args.openai_api_key
        global client
        client = openai.OpenAI(api_key=openai_api_key)
        input_args = []
        for line in lines:
            res_dict = json.loads(line)

            raw_output = res_dict["display_output"]

            prompt = f"""Please rephrase the following sentences:

{raw_output}
"""
            input_args.append({"prompt": prompt, "max_try": 10})

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(gpt_paraphrase_attack, input_args)

    elif args.attack_type == "back_translation":
        save_path = os.path.join(sub_dir, "text_generation_back_translation.txt")
        input_args = []
        for line in lines:
            res_dict = json.loads(line)

            raw_output = res_dict["display_output"]

            prompt = f"""Please translate the following sentences into French, directly output the results:

{raw_output}
"""
            input_args.append({"prompt": prompt, "max_try": 10})

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(gpt_paraphrase_attack, input_args)

        input_args = []
        for result in results:
            success, output = result
            if success:
                prompt = f"""Please translate the following sentences into English, directly output the results:

{output}
"""
            else:
                prompt = ""
            input_args.append({"prompt": prompt, "max_try": 10})

        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(gpt_paraphrase_attack, input_args)
    elif args.attack_type == "dipper":
        save_path = os.path.join(sub_dir, "text_generation_dipper.txt")
        dp = DipperParaphraser()

        results = []
        print("dipper running...", flush=True)
        for line in tqdm.tqdm(lines):
            res_dict = json.loads(line)

            raw_output = res_dict["display_output"]

            output = dp.paraphrase(
                raw_output,
                lex_diversity=60,
                order_diversity=0,
                prefix="",
                do_sample=True,
                top_p=1,
                top_k=None,
                max_length=1024,
            )
            if len(output) > 0:
                results.append([1, output])
            else:
                results.append([0, output])

    else:
        raise ValueError(f"Unknown attak_type: {args.attack_type}")

    with open(save_path, "w") as f:
        for idx, line in enumerate(lines):
            res_dict = json.loads(line)
            res_dict["output"] = results[idx][1]

            if results[idx][0]:
                f.write(json.dumps(res_dict) + "\n")


if __name__ == "__main__":
    main()
