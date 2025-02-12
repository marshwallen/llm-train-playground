# DeepSpeed 蒸馏训练
- 知识蒸馏将复杂的大型模型（教师模型）的知识迁移到较小的模型（学生模型）
- 这里以 DeekSeek-R1/Qwen2.5 为老师模型，Qwen2.5-3B 为学生模型为例，进行蒸馏训练
- 以下是蒸馏训练的流程图（图源：https://xueqiu.com/3993902801/321957662）
![image](https://xqimg.imedao.com/194af791f7f1af5e3fd7fea7.jpeg!800.jpg)

## Instructions
1. **准备数据集**
- 见 ```llm-train-playground/README.md``` 中的数据预处理部分

2. **下载学生模型权重**
- 模型名称见：https://www.modelscope.cn/models/
- 这里以 Qwen2.5-3B 为例，执行以下命令：
```sh
# 回到该项目的根目录下执行
cd llm-train-playground/
python download_pt.py --model Qwen/Qwen2.5-3B
```

3. **配置教师模型API**
- DeepSeek: https://platform.deepseek.com/
- 阿里千问：https://bailian.console.aliyun.com/
- 配置文件：```teacher_config.yaml```
```
model:
  deepseek:
    api_key:
    base_url: https://api.deepseek.com
  qwen:
    api_key:
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
```