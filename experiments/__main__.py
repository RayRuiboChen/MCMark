import os


def text_generation_undetectable_exp(
    res_dir, eps, model_str, reweight_type, dataset_name, robustness_type
):
    from . import text_generation as tg

    assert eps >= 0
    assert eps <= 1

    sub_dir = os.path.join(
        res_dir, dataset_name, model_str.split("/")[-1].replace("-", "_"), reweight_type
    )
    os.makedirs(sub_dir, exist_ok=True)

    if robustness_type is None:  # not a robustness experiment
        output_path = os.path.join(sub_dir, "text_generation.txt")
        score_save_path = os.path.join(sub_dir, "score.txt")
        if not os.path.exists(output_path):
            print("tg.get_output.undetectable_exp_pipeline()", flush=True)
            tg.get_output.undetectable_exp_pipeline(
                output_path=output_path,
                model_str=model_str,
                reweight_type=reweight_type,
                dataset_name=dataset_name,
            )
        else:
            print(output_path, " exists! Job skipped.")

    elif robustness_type == "random_token_replacement":
        output_path = os.path.join(sub_dir, "text_generation.txt")
        assert os.path.exists(output_path)
        eps_str = str(eps).replace(".", "_")
        score_save_path = os.path.join(
            sub_dir, f"score_random_token_replacement_eps_{eps_str}.txt"
        )
    elif robustness_type == "gpt_rephrase":
        output_path = os.path.join(sub_dir, "text_generation_gpt_rephrase.txt")
        score_save_path = os.path.join(sub_dir, "score_gpt_rephrase.txt")
        assert os.path.exists(output_path)
    elif robustness_type == "back_translation":
        output_path = os.path.join(sub_dir, "text_generation_back_translation.txt")
        score_save_path = os.path.join(sub_dir, "score_back_translation.txt")
        assert os.path.exists(output_path)
    elif robustness_type == "dipper":
        output_path = os.path.join(sub_dir, "text_generation_dipper.txt")
        score_save_path = os.path.join(sub_dir, "score_gpt_dipper.txt")
        assert os.path.exists(output_path)
    else:
        raise ValueError(f"Unknown robustness type: {robustness_type}")

    if os.path.exists(score_save_path):
        print("Found exisiting score_save_path:")
        print(score_save_path)
        print("Job skipped.")
        return

    print("tg.evaluate_watermark_score.pipeline()", flush=True)
    tg.evaluate_watermark_score.pipeline(
        output_path=output_path,
        score_save_path=score_save_path,
        eps=eps,
        model_str=model_str,
        dataset_name=dataset_name,
    )

    print("finish text generation.")


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_str", type=str, help="Model path for text generation.")
    parser.add_argument(
        "--reweight_type", type=str, choices=["mcmark_ablation", "main_exp", "mcmark"]
    )
    parser.add_argument("--res_dir", type=str)
    parser.add_argument("--eps", type=float, default=0)
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name for text generation",
        choices=[
            "mmw_book_report",
            "mmw_story",
            "mmw_fake_news",
            "dolly_cw",
            "longform_qa",
            "finance_qa",
        ],
    )
    parser.add_argument(
        "--robustness_type",
        type=str,
        choices=[
            "random_token_replacement",
            "dipper",
            "gpt_rephrase",
            "back_translation",
        ],
    )
    args = parser.parse_args()

    text_generation_undetectable_exp(
        res_dir=args.res_dir,
        eps=args.eps,
        model_str=args.model_str,
        reweight_type=args.reweight_type,
        dataset_name=args.dataset_name,
        robustness_type=args.robustness_type,
    )


if __name__ == "__main__":
    main()
