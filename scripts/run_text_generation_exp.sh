# models used in our paper: 
# 'mistralai/Mistral-7B-Instruct-v0.3',
# 'meta-llama/Llama-3.2-3B-Instruct',
# 'meta-llama/Llama-2-7b-chat-hf',
# 'microsoft/Phi-3.5-mini-instruct',
model_str="meta-llama/Llama-3.2-3B-Instruct"

# 'mmw_book_report','mmw_story','mmw_fake_news','dolly_cw','longform_qa','finance_qa'
dataset_name="mmw_book_report"

# 'mcmark': run mcmark with l=20
# 'main_exp' run main experiments
# 'mcmark_ablation' run ablation study
reweight_type='mcmark'


python -m experiments \
    --res_dir "./results" \
    --model_str "$model_str" \
    --reweight_type "$reweight_type" \
    --dataset_name "$dataset_name"