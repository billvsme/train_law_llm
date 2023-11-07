train_law_llm
==============

LLM微调上手项目，一步一步使用colab训练法律LLM，基于phi-1_5

| name | Colab |
| --- | --- |
自我认知微调 | [![web ui](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/drive/1in_tXBkewd5FivNTOn-B6Za_WjRyHL4r#scrollTo=h43G1zhf7msU&forceEdit=true&sandboxMode=true)



## 目标
使用colab免费的T4显卡，完成通过法律问答数据微调 [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) 模型  


## 自我认知微调
自我认知数据来源：[self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  

80条数据，使用T4微淘phi-1_5，几分钟就可以微调完毕  

**微调参数**，具体步骤详见colab
```
python src/train_bash.py \
    --stage sft \
    --model_name_or_path microsoft/phi-1_5 \
    --do_train True\
    --finetuning_type lora \
    --template vanilla \
    --flash_attn False \
    --shift_attn False \
    --dataset_dir data \
    --dataset self_cognition \
    --cutoff_len 1024 \
    --learning_rate 2e-04 \
    --num_train_epochs 20.0 \
    --max_samples 1000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 100 \
    --warmup_steps 0 \
    --neft_alpha 0 \
    --train_on_prompt False \
    --upcast_layernorm False \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target Wqkv \
    --resume_lora_training True \
    --output_dir saves/Phi1.5-1.3B/lora/my \
    --fp16 True \
    --plot_loss True
```

## 法律问答微调
法律问答数据来源：[DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)
```
```


