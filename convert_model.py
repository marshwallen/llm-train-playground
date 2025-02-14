# 将 DeepSpeed 的模型转换成 Hugging Face 支持的 Transformers 模型
# 参考：https://zhuanlan.zhihu.com/p/21711764788

import argparse
import subprocess
import os
import shutil


def merge_pt_to_bin(args):
    """
    自动合并多个 bf16_zero_pp_rank_*_optim_states.pt 和 zero_pp_rank_*_model_states.pt,
    生成一个完整的 PyTorch.pth 模型, 不再包含优化器状态
    """
    os.makedirs("output_hf/", exist_ok=True)
    os.makedirs(os.path.join("output_hf", args.save_model), exist_ok=True)

    r = subprocess.run([
        "python", os.path.join(args.zero_model, "zero_to_fp32.py"), 
        args.zero_model, os.path.join("output_hf", args.save_model),
        "--tag", args.t,
        ])
    
    copy_files = [
        "README.md",
        "config.json", 
        "configuration.json",
        "generation_config.json",
        "merges.txt",
        "special_tokens_map.json", 
        "tokenizer_config.json", 
        "vocab.json", 
        "vocab.txt",
        ]
    
    for f in copy_files:
        if os.path.exists(os.path.join(args.org_model, f)):
            shutil.copyfile(os.path.join(args.org_model, f), os.path.join("output_hf", args.save_model, f))

def test_load(args):
    """
    验证转换后的模型能否正常加载
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model = AutoModelForCausalLM.from_pretrained(
        os.path.join("output_hf", args.save_model),
        trust_remote_code=True  # 如果有自定义代码
    )
    tokenizer = AutoTokenizer.from_pretrained(os.path.join("output_hf", args.save_model))

    inputs = tokenizer("Hello world!", return_tensors="pt")
    outputs = model(**inputs)

    print("# Model load success.")

if __name__ == "__main__":
    # 主函数
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_model", type=str, required=True)
    parser.add_argument("--save_model", type=str, required=True)
    parser.add_argument("--org_model", type=str, required=True)
    parser.add_argument("--t", type=str, required=True, help="checkpoint tag used as a unique identifier for checkpoint. e.g., global_step1")
    args = parser.parse_args()

    merge_pt_to_bin(args)
    test_load(args)




