import json
import os
import numpy as np
from collections import defaultdict
import math
import re
import numpy as np
import argparse


def get_fpr(n, m):
    res = 0
    for i in range(m, n + 1):
        res += math.comb(n, i) * (2 ** (n - i))
    res /= 3**n
    return res


def get_split_fpr(n, m, split_num):
    res = 0
    for i in range(m, n + 1):
        res += math.comb(n, i) * ((split_num - 1) ** (n - i))
    res /= split_num**n
    return res


def extract_n_value(text):
    pattern = re.search(r"MC_Reweight\(n=(\d+)\)", text)
    if pattern:
        return int(pattern.group(1))
    raise NotImplementedError


def get_save_path(score_path, fpr_thres, len_limit):
    save_dir = "./results"
    info_path = "/".join(score_path.split("/")[-4:-1])
    file_name = (
        score_path.split("/")[-1].split(".")[0]
        + "_mcmark_eval_results_"
        + str(fpr_thres).replace(".", "_")
        + "_"
        + str(len_limit)
        + ".txt"
    )

    return os.path.join(save_dir, info_path, file_name)


def generate_result(score_path, save_path, fpr_thres, len_limit):
    with open(score_path, "r") as f:
        lines = f.readlines()

    tot_cnt = defaultdict(int)
    acc_cnt = defaultdict(int)
    fpr_list = defaultdict(list)

    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]

        wp = res_dict["watermark_processor"]

        if cur_len < len_limit:
            continue
        if "MC_Reweight" not in wp:
            continue

        raw_score = res_dict["raw_scores"]
        cur_n = extract_n_value(wp)

        cur_fpr = get_split_fpr(int(cur_len), int(raw_score), split_num=cur_n)
        fpr_list[cur_n].append(cur_fpr)

        if cur_fpr <= fpr_thres:
            acc_cnt[cur_n] += 1
        tot_cnt[cur_n] += 1

    lines = []
    for wp in sorted(tot_cnt.keys()):
        lines.append("-" * 80 + "\n")
        lines.append(f"MC_Reweight(n={wp})\n")
        lines.append(f"Total cnt: {tot_cnt[wp]}\n")
        lines.append(f"Median p-value: {np.median(fpr_list[wp])}\n")
        lines.append(f"TPR@FPR={fpr_thres}: {acc_cnt[wp]/tot_cnt[wp]}\n")
    # return lines

    with open(save_path, "w") as f:
        f.writelines(lines)


def get_lines(score_path, fpr_thres, len_limit):
    save_path = get_save_path(score_path, fpr_thres, len_limit)

    if not os.path.exists(save_path):
        save_dir = "/".join(save_path.split("/")[:-1])
        os.makedirs(save_dir, exist_ok=True)
        generate_result(score_path, save_path, fpr_thres, len_limit)

    with open(save_path, "r") as f:
        lines = f.readlines()
    return lines


def get_result_dict(score_path, fpr_thres, len_limit):
    lines = get_lines(score_path, fpr_thres, len_limit)

    line_num = len(lines)
    assert line_num % 5 == 0
    res_num = line_num // 5
    res_dict = {}
    for res_idx in range(res_num):
        cur_n = extract_n_value(lines[res_idx * 5 + 1].strip())
        median_p = float((lines[res_idx * 5 + 3].strip()).split(":")[-1])
        tpr = float((lines[res_idx * 5 + 4].strip()).split(":")[-1])
        res_dict[f"MCMark(l={cur_n})"] = {"median_p": median_p, "tpr": tpr}
    return res_dict


def print_results(res_dict):
    for key in sorted(res_dict.keys()):
        print(f"{key}: {res_dict[key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--fpr_thres", type=float)
    args = parser.parse_args()
    res_dict = get_result_dict(args.score_path, args.fpr_thres, len_limit=510)
    print_results(res_dict)


if __name__ == "__main__":
    main()
