import torch
from load_data import DataProcessor
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import swanlab
import argparse
import os
import deepspeed
import numpy as np
import json
from zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

def train(args):
    """
    训练主函数
    test=True 表示在训练完成后进行测试
    """
    # 1 使用 Transformers 加载模型权重
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir + args.model_name, 
        trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir + args.model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True)
    
    # 2 加载数据集
    data_processor = DataProcessor(args, tokenizer)
    train_dataloader = data_processor.load_train_data()

    # 3 配置 LoRA
    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )
    peft_model = get_peft_model(model, config)

    # 4 配置Trainer
    # 配置 DeepSpeed 训练参数
    with open("deepspeed_config.json", "r") as f:
        training_args = json.load(f)

    training_args.update({
        "train_micro_batch_size_per_gpu": args.batch_size,
        "num_train_epochs": args.epochs,
    })

    # 初始化 DeepSpeed
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=peft_model,
        model_parameters=peft_model.parameters(),
        config_params=training_args # 配置文件
    )

    # 5 配置 SwanLab
    run = swanlab.init(
        project="DeepSpeed-LoRA",
        # 跟踪超参数与实验元数据
        config=training_args
    )

    # 6 DeepSpeed 训练循环
    model_engine.train()
    epoch_min_loss = 2^32-1
    
    for epoch in range(training_args["num_train_epochs"]):
        epoch_loss = []
        for step, batch in enumerate(train_dataloader):
            # Load data
            input_ids = batch["input_ids"].to(model_engine.device)
            attention_mask = batch["attention_mask"].to(model_engine.device)
            labels = batch["labels"].to(model_engine.device)
            
            outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            epoch_loss.append(loss.item())
            model_engine.backward(loss)
            model_engine.step()

            # SwanLab 记录训练数据(step)
            swanlab.log({"loss(step)": loss.item()})

        # Save checkpoint
        epoch_avg_loss = np.average(epoch_loss)
        if epoch_avg_loss < epoch_min_loss:
            epoch_min_loss = epoch_avg_loss
            model_engine.save_checkpoint(
                os.path.join(args.output_dir, "DS-" + args.model_name),
                args.ckpt_id
                )
            
        # SwanLab 记录训练数据(epoch)    
        swanlab.log({"loss(epoch)": epoch_avg_loss})

    # 在Jupyter Notebook中运行时要停止SwanLab记录，需要调用swanlab.finish()
    swanlab.finish()

def test_infer(args, max_new_tokens=128):
    """
    训练完成后测试推理
    max_new_tokens: 推理输出多少 tokens
    """

    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir + args.model_name, 
        trust_remote_code=True)
    
    # 加载 model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir + args.model_name, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True)

    # 读取测试数据
    data_processor = DataProcessor(args, tokenizer)
    test_data = data_processor.load_test_data()

    # 3 配置 LoRA
    val_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=True,  # 推理模式
        r=64,  # Lora 秩
        lora_alpha=16,  # Lora alaph，具体作用参见 Lora 原理
        lora_dropout=0.05,  # Dropout 比例
        bias="none",
    )
    val_peft_model = get_peft_model(model, val_config)

    # 加载 DeepSpeed 训练好的权重
    ds_state_dict = get_fp32_state_dict_from_zero_checkpoint(os.path.join(args.output_dir, "DS-" + args.model_name)) # already on cpu
    val_peft_model = val_peft_model.cpu() # move to cpu
    val_peft_model.load_state_dict(ds_state_dict)
    val_peft_model.cuda()

    # Test
    # 5 配置 SwanLab
    run = swanlab.init(
        project="DeepSpeed-LoRA",
        # 跟踪超参数与实验元数据
        config={"eval": True}
    )

    r_list = []
    for item in test_data:
        input_prompt = item["conversations"][0]["value"]
        label_response = item["conversations"][1]["value"]

        messages = [
            {"role": "system", "content": "你是一个B站老玩家."},
            {"role": "user", "content": input_prompt}
        ]
        response = predict_deepspeed(messages, tokenizer, model, max_new_tokens)

        messages.append({"role": "assistant", "content": f"{response}", "label": label_response})
        infer_result = {
            "System Prompt": messages[0]["content"],
            "User Prompt": messages[1]["content"],
            "Response": messages[2]['content']
        }
        infer_result_str = json.dumps(infer_result, ensure_ascii=False, indent=4)
        r_list.append(swanlab.Text(infer_result_str))
        print(infer_result_str)

    swanlab.log({"Infer result": r_list})
    swanlab.finish()

def predict_deepspeed(messages, tokenizer, model, max_new_tokens):
    """
    推理函数
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../data/", help="数据集目录")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="名称见 https://huggingface.co/")
    parser.add_argument("--model_dir", type=str, default="../model/", help="所有模型存放目录")
    parser.add_argument("--output_dir", type=str, default="output/", help="训练权重输出目录")
    parser.add_argument("--ckpt_id", type=int, default=0, help="权重id")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=4096, help="用于训练的最大 token 数")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="推理输出的最大 token 数")
    parser.add_argument("--train_mode", type=bool, default=True, help="开启训练模式")
    parser.add_argument("--test_mode", type=bool, default=True, help="开启测试模式")
    
    args = parser.parse_args()
    if args.train_mode:
        train(args)
    if args.test_mode:
        test_infer(args)