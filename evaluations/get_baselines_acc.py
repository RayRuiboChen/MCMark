import json
import os
import numpy as np
from collections import defaultdict
import scipy.stats as stats
import re
import math
import argparse


def z_to_fpr(z):
    # return 1 - stats.norm.cdf(z)
    return stats.norm.sf(z)


def fpr2thres(fpr):
    return (-0.5 * math.log(fpr)) ** 0.5


def thres2fpr(thres):
    return math.exp(-2 * thres**2)


def get_KGW_res(score_path, fpr, len_limit):
    with open(score_path, "r") as f:
        lines = f.readlines()
    tot_cnt = defaultdict(int)
    acc_cnt = defaultdict(int)
    z_score_list = defaultdict(list)
    fpr_list = defaultdict(list)

    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]
        wp = res_dict["watermark_processor"]
        if cur_len < len_limit:
            continue
        if ("STA" not in wp) and ("John" not in wp) and ("Unigram" not in wp):
            continue

        raw_score = res_dict["raw_scores"]
        z_score = 2 * (raw_score - 0.5 * cur_len) / cur_len**0.5
        tot_cnt[wp] += 1
        z_score_list[wp].append(z_score)
        cur_fpr = z_to_fpr(z_score)

        fpr_list[wp].append(cur_fpr)
        if cur_fpr <= fpr:
            acc_cnt[wp] += 1

    res_dict = {}
    for k in tot_cnt.keys():
        if "John" in k:
            delta = re.findall(r"delta=(\d+\.?\d*)", k)[0]
            wp_name = f"KGW($\\delta$={delta})"
        elif "Unigram" in k:
            delta = re.findall(r"delta=(\d+\.?\d*)", k)[0]
            wp_name = f"Unigram($\\delta$={delta})"
        elif "STA" in k:
            wp_name = "STA-1"
        else:
            raise NotImplementedError
        res_dict[wp_name] = {
            "median_p": np.median(fpr_list[k]),
            "tpr": acc_cnt[k] / tot_cnt[k],
        }
    return res_dict


def get_dip_res(score_path, fpr, len_limit):
    with open(score_path, "r") as f:
        lines = f.readlines()

    tot_cnt = defaultdict(int)
    acc_cnt = defaultdict(int)
    beta_score_list = defaultdict(list)
    threshold = fpr2thres(fpr)

    for line in lines:
        res_dict = json.loads(line)
        cur_len = res_dict["lens"]
        wp = res_dict["watermark_processor"]
        if cur_len < len_limit:
            continue
        if "Dip" not in wp:
            continue

        beta_score = res_dict["beta_score"]
        beta_score_list[wp].append(beta_score)

        tot_cnt[wp] += 1
        if beta_score > threshold:
            acc_cnt[wp] += 1
    res_dict = {}
    for k in tot_cnt.keys():
        alpha = re.findall(r"alpha=(\d+\.?\d*)", k)[0]
        if abs(float(alpha) - 0.5) < 1e-5:
            wp_name = "$\\gamma$-reweight"
        else:
            wp_name = f"DiPmark($\\alpha$={alpha})"
        median_beta = np.median(beta_score_list[k])

        res_dict[wp_name] = {
            "median_p": thres2fpr(median_beta),
            "tpr": acc_cnt[k] / tot_cnt[k],
        }
    return res_dict


def print_results(res_dict):
    for key in sorted(res_dict.keys()):
        print(f"{key}: {res_dict[key]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score_path", type=str)
    parser.add_argument("--fpr_thres", type=float)
    args = parser.parse_args()

    res_dict = get_KGW_res(args.score_path, fpr=args.fpr_thres, len_limit=510)
    print_results(res_dict)
    res_dict = get_dip_res(args.score_path, fpr=args.fpr_thres, len_limit=510)
    print_results(res_dict)


if __name__ == "__main__":
    main()
