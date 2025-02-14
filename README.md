# llm-train-playground
- 本 repo 是 LLM 训练/微调实践
- 通过学习B友的说话风格，实现对 LLM 输入B站视频标题、简介和父评论内容（如有）时给出可能的回复。

## DeepSpeed 简介
- DeepSpeed 是一个开源的深度学习优化库，支持分布式数据并行、混合精度训练和多种加速器
官方教程：https://www.deepspeed.ai/getting-started/
- 本 repo 采用单卡训练（NVIDIA RTX 3060 Ti 8G），使用 DeepSpeed 的 ZeRO-Offload 优化器，将模型参数和优化器状态存储在 GPU 上，将模型权重存储在 CPU 上，从而实现模型参数和优化器状态的分布式存储和通信。

## 数据集
- B站评论区数据。获取方法移步至该 repo：https://github.com/NanmiCoder/MediaCrawler
- 本 repo 使用的部分配置如下：
```python
# 文件目录：MediaCrawler/config/base_config.py
# 爬取一级评论的数量控制(单视频/帖子)
CRAWLER_MAX_COMMENTS_COUNT_SINGLENOTES = 5

# 是否开启爬二级评论模式, 默认不开启爬二级评论
# 老版本项目使用了 db, 则需参考 schema/tables.sql line 287 增加表字段
ENABLE_GET_SUB_COMMENTS = True

# 指定bili创作者ID列表(sec_id)
# 极客湾 + 笔吧测评室
BILI_CREATOR_ID_LIST = [
    "25876945",
    "367877"
]
```
- 执行命令（需要扫码登陆 Bilibili）
```sh
cd MediaCrawler/
python main.py --platform bili --lt qrcode --type creator
```
- 数据拷贝
```MediaCrawler/data/bilibili/json/```目录下的所有json文件拷贝至当前目录下的```data/```即可。具体的数据格式可见根目录下的```examples/```

## Instruction
- 安装环境依赖
```sh
pip install -r requirements.txt
```

## 数据预处理与权重下载
1. **数据预处理**
将抓取的B站评论区的数据格式化到指定文件
```sh
# total 参数可控制需要从中那多少条数据来训练
python prepare_data.py --total 2000
```

2. **从 Hugging Face Hub/Modelscope 下载预训练模型权重到本地**
- 这里的model参数需为 HuggingFace 上的完整名称
```sh
python download_pt.py --model Qwen/Qwen2.5-0.5B
```

## DeepSpeed 微调训练
- LoRA 微调与 QLoRA 微调
- 跳转：https://github.com/marshwallen/llm-train-playground/tree/main/deepspeed_ft

## DeepSpeed 蒸馏训练
- 知识蒸馏将复杂的大型模型（教师模型）的知识迁移到较小的模型（学生模型）
- 例如：DeepSeek-R1-Distill-Qwen、DeepSeek-R1-Distill-Llama 等
- 100B 以上大模型蒸馏到本地 0.5-32B 小模型
- 跳转：https://github.com/marshwallen/llm-train-playground/tree/main/deepspeed_distill

## 模型转换与发布
1. **模型转换**
- 在分布式训练 LLM 时，DeepSpeed 的 ZeRO (Zero Redundancy Optimizer) 技术会将模型权重和优化器状态分片存储，以减少 GPU 显存占用。会产生多个 pt 文件（如下所示），需要将它们合并为一个文件。

```sh
-rw-r--r-- 1 marshwallen marshwallen 1937771600 Feb 13 03:51 bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt
-rw-r--r-- 1 marshwallen marshwallen 5548157566 Feb 13 03:51 zero_pp_rank_0_mp_rank_00_model_states.pt
```

- 这里使用 DeepSpeed 的 ```zero_to_fp32.py``` 去转换微调好的模型，并附上原模型的分词器和其他配置
- 关于该脚本的官方介绍：
```python
# This script extracts fp32 consolidated weights from a zero 1, 2 and 3 DeepSpeed checkpoints. It gets
# copied into the top level checkpoint dir, so the user can easily do the conversion at any point in
# the future. Once extracted, the weights don't require DeepSpeed and can be used in any
# application.
```
- 模型转换命令如下：
```sh
# 当前目录下执行
python convert_model.py 
    --zero_model <微调好的模型路径> \
    --org_model  <原预训练模型路径> \
    --save_model <欲发布的模型名称> \
    --t <checkpoint id>

# 例子：
# python convert_model.py 
#     --zero_model deepspeed_distill/output/DS-Qwen/Qwen2.5-3B-BNB \
#     --org_model model/Qwen/Qwen2.5-3B \
#     --save_model Qwen2.5-3B-Distill-BiliComments \
#     --t 24 \
```
- 转换后的模型会保存在 ```./output_hf/<save_model>``` 目录下

2. **模型发布到 Hugging Face Hub**

- **1 注册 Hugging Face 账号 [链接](https://huggingface.co/)**

- **2 安装 Git LFS (Linux)**

    Git LFS 是一个 Git 的扩展，用于管理大文件，如模型权重等
```sh
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
```

- **3 安装并登录 huggingface-cli**

此过程需要 Access Token，需在 Hugging Face 个人中心设置
```sh
pip install huggingface_hub
huggingface-cli login
```

- **4 创建仓库**
```sh
huggingface-cli repo create <仓库名称>
```

- **5 仓库构建**

把创建好的仓库 clone 下来，将转换好的模型权重和配置文件 copy 再到仓库中，最后 commit
```sh
# 1 Clone 仓库
git clone https://huggingface.co/<username>/<model_name>
cd <model_name>

# 2 Copy 在模型转换部分处理好的所有模型权重和配置文件到仓库中
# 文件结构大致如下：
ls -l 
# -rw-r--r-- 1 username username       3840 Feb 14 11:53 README.md
# -rw-r--r-- 1 username username        683 Feb 14 11:53 config.json
# -rw-r--r-- 1 username username          2 Feb 14 11:53 configuration.json
# -rw-r--r-- 1 username username        138 Feb 14 11:53 generation_config.json
# -rw-r--r-- 1 username username    1671839 Feb 14 11:53 merges.txt
# -rw-r--r-- 1 username username 2489348338 Feb 14 11:53 pytorch_model.bin
# -rw-r--r-- 1 username username       7228 Feb 14 11:53 tokenizer_config.json
# -rw-r--r-- 1 username username    2776833 Feb 14 11:53 vocab.json

# 3 Push 到远程仓库
git add .
git commit -m "<commit message>"
git push

# 4 浏览器访问仓库：https://huggingface.co/<username>/<model_name>
```

# Reference
- https://github.com/datawhalechina/self-llm
- https://www.deepspeed.ai/getting-started/
- https://github.com/NanmiCoder/MediaCrawler
- https://xueqiu.com/3993902801/321957662
- https://blog.csdn.net/u011426236/article/details/135880314