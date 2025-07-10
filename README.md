# Improved unbiased watermark for large language models (ACL2025-Main)
Ruibo Chen*, Yihan Wu*, Junfeng Guo, Heng Huang

## Introduction \[[Paper](https://www.arxiv.org/pdf/2502.11268)\]


<p align="center">
<img src=images/mcmark_method.png  width="60%">
</p>


In this paper, we introduce **MCMARK**, a family of unbiased, **M**ulti-**C**hannel-based water**mark**s. **MCMARK** works by partitioning the model’s vocabulary into segments and promoting token probabilities within a selected segment based on a watermark key. We demonstrate that **MCMARK**
not only preserves the original distribution of the language model but also offers significant improvements in detectability and robustness over existing unbiased watermarks.

<p align="center">
<img src=images/mcmark_experiments.png  width="95%">
</p>



## Quick Start

Prepare the environment:

```
conda create -n mcmark python=3.11
conda activate mcmark
pip install -r requirements.txt
```

## Usage

Run the watermarking experiments:

```
bash ./scripts/run_text_generation_exp.sh
```


Run robustness experiments:  
Add your OpenAi API key for GPT rephrase and GPT back translation attacks in `./scripts/run_robustness_exp.sh`.


Then run:

```
bash ./scripts/run_robustness_exp.sh
```

Please note that you could change some parameters in these files for different setting.


For evaluations, use:

```
bash ./scripts/run_evaluations.sh
```



## Citation

If you find our work useful for your research and applications, please consider citing:

```
@article{chen2025improved,
  title={Improved unbiased watermark for large language models},
  author={Chen, Ruibo and Wu, Yihan and Guo, Junfeng and Huang, Heng},
  journal={arXiv preprint arXiv:2502.11268},
  year={2025}
}
```
