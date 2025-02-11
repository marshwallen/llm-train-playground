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

2. **从 HuggingFace Hub/Modelscope 下载预训练模型权重到本地**
- 这里的model参数需为 HuggingFace 上的完整名称
```sh
python download_pt.py --model Qwen/Qwen2.5-0.5B
```

## DeepSpeed 微调训练
- LoRA 微调与 QLoRA 微调
- 跳转：https://github.com/marshwallen/llm-train-playground/tree/main/deepspeed_ft

## DeepSpeed + LLM API 蒸馏训练
- 知识蒸馏将复杂的大型模型（教师模型）的知识迁移到较小的模型（学生模型）
- 例如：DeepSeek-R1-Distill-Qwen、DeepSeek-R1-Distill-Llama 等
- 100B 以上大模型蒸馏到本地 0.5-32B 小模型
- 跳转：https://github.com/marshwallen/llm-train-playground/tree/main/deepspeed_distill

# Reference
- https://github.com/datawhalechina/self-llm
- https://www.deepspeed.ai/getting-started/
- https://github.com/NanmiCoder/MediaCrawler
- https://xueqiu.com/3993902801/321957662