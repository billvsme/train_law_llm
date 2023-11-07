train_law_llm
==============

LLM微调上手项目，一步一步使用colab训练法律LLM，基于phi-1_5

| name | Colab | 数据
| --- | --- | --- 
自我认知微调 | [![web ui](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/drive/1in_tXBkewd5FivNTOn-B6Za_WjRyHL4r#scrollTo=h43G1zhf7msU&forceEdit=true&sandboxMode=true) | [self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  
法律问答微调 | [![web ui](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/drive/1bfUb1HsJOgdzZMrVlk2RCXDynQIa6It3#forceEdit=true&sandboxMode=true) | [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)  

## 目标
使用colab免费的T4显卡，完成通过法律问答数据微调 [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) 模型  


## 自我认知微调
自我认知数据来源：[self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  

80条数据，使用T4微调phi-1_5，几分钟就可以微调完毕  

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

**效果**

<a href="https://sm.ms/image/gXNOy3lHdmeqv5j" target="_blank"><img src="https://s2.loli.net/2023/11/07/gXNOy3lHdmeqv5j.png" width="60%"></a>

## 法律问答微调
法律问答数据来源：[DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)  
为了减省显存，使用deepspeed stage2，cutoff_len可以最多到1664，再多就要爆显存了  

**deepspeed配置**
```
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "gradient_clipping": "auto",
  "zero_allow_untested_optimizer": true,
  "fp16": {
    "enabled": "auto",
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
```

**微调参数**  

1000条数据，T4大概需要60分钟  
```
deepspeed --num_gpus 1 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path microsoft/phi-1_5 \
    --do_train True \
    --finetuning_type lora \
    --template vanilla \
    --flash_attn False \
    --shift_attn False \
    --dataset_dir data \
    --dataset self_cognition,law_sft_triplet \
    --cutoff_len 1664 \
    --learning_rate 2e-04 \
    --num_train_epochs 5.0 \
    --max_samples 1000 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --save_steps 1000 \
    --warmup_steps 0 \
    --neft_alpha 0 \
    --train_on_prompt False \
    --upcast_layernorm False \
    --lora_rank 8 \
    --lora_dropout 0.1 \
    --lora_target Wqkv \
    --resume_lora_training True \
    --output_dir saves/Phi1.5-1.3B/lora/law \
    --fp16 True \
    --plot_loss True
```


