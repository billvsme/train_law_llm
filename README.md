✏️LLM微调上手项目
==============

一步一步使用Colab训练法律LLM，基于[microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) ,[ChatGLM3-6B](https://huggingface.co/THUDM/chatglm3-6b)。通过本项目你可以0成本手动了解微调LLM。如果想要了解LLM微调具体代码实现，**可以参考 [my_finetune](https://github.com/billvsme/my_finetune) 项目**🤓。

| name | Colab | Datasets
| --- | --- | --- 
自我认知 lora-SFT 微调 | [![Colab](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/github/billvsme/train_law_llm/blob/master/colab/train_self_cognition.ipynb) | [self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  
法律问答 lora-SFT 微调 | [![Colab](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/github/billvsme/train_law_llm/blob/master/colab/train_law.ipynb) | [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)  
法律问答 全参数-SFT 微调* | [![Colab](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/github/billvsme/train_law_llm/blob/master/colab/train_law_full.ipynb) | [DISC-LawLLM](https://github.com/FudanDISC/DISC-LawLLM)  
ChatGLM3-6B 自我认知 lora-SFT 微调* | [![Colab](https://img.shields.io/badge/✏️-Colab-important)](https://colab.research.google.com/github/billvsme/train_law_llm/blob/master/colab/train_law_chatglm3.ipynb) | [self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  



*如果是Colab Pro会员用户，可以尝试全参数-SFT微调，使用高RAM+T4，1000条数据大概需要20+小时  
*如果是Colab Pro会员用户，ChatGLM3-6B 自我认知lora-SFT 微调，使用高RAM+T4，只需要几分钟，效果比较好


## 目标
使用colab免费的T4显卡，完成法律问答 指令监督微调(SFT) [microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) 模型  


## 自我认知微调
自我认知数据来源：[self_cognition.json](https://github.com/hiyouga/LLaMA-Factory/blob/main/data/self_cognition.json)  

80条数据，使用T4 lora微调phi-1_5，几分钟就可以微调完毕  

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
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
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
为了减省显存，使用deepspeed stage2，cutoff_len可以最多到1792，再多就要爆显存了  

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
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
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
    --cutoff_len 1792 \
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

## 全参微调

可以通过，estimate_zero3_model_states_mem_needs_all_live查看deepspeed各个ZeRO stage 所需要的内存。  

```
from transformers import AutoModel, AutoModelForCausalLM
from deepspeed.runtime.zero.stage3 import estimate_zero3_model_states_mem_needs_all_live

model_name = "microsoft/phi-1_5"
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
estimate_zero3_model_states_mem_needs_all_live(model, num_gpus_per_node=1, num_nodes=1)
```

如图所适 offload_optimizer -> cpu 后[microsoft/phi-1_5](https://huggingface.co/microsoft/phi-1_5) 需要32G内存，colab高内存有52G可以满足需求。  


<a href="https://sm.ms/image/EvTL7FgRzUj69rf" target="_blank"><img src="https://s2.loli.net/2023/11/07/EvTL7FgRzUj69rf.png" width="60%"></a>

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
    "allgather_bucket_size": 2e8,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "overlap_comm": false,
    "contiguous_gradients": true
  }
}
```
```
deepspeed --num_gpus 1 --master_port=9901 src/train_bash.py \
    --deepspeed ds_config.json \
    --stage sft \
    --model_name_or_path microsoft/phi-1_5 \
    --do_train True \
    --finetuning_type full \
    --template vanilla \
    --flash_attn False \
    --shift_attn False \
    --dataset_dir data \
    --dataset self_cognition,law_sft_triplet \
    --cutoff_len 1024 \
    --learning_rate 2e-04 \
    --num_train_epochs 10.0 \
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
    --output_dir saves/Phi1.5-1.3B/lora/law_full \
    --fp16 True \
    --plot_loss True
```

也可以考虑使用 [kaggle](https://www.kaggle.com/)，可以每周使用30个小时，可以选择2张T4，使用ZeRO stage 3 全参微调  

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
    "stage": 3,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "sub_group_size": 5e7,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 5e7,
    "stage3_max_reuse_distance": 5e7,
    "stage3_gather_16bit_weights_on_model_save": true
  }
}
```



