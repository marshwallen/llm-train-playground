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

4. **从教师模型中蒸馏数据**
- 这里以 DeepSeek-V3 为例，执行以下命令：
```sh
# 数据会保存在当前目录的 data/ 下
python prepare_data.py --d_type test --model_name deepseek-chat
python prepare_data.py --d_type train --model_name deepseek-chat
```

5. **使用 SFT 蒸馏训练学生模型**
- 这里以 Qwen2.5-3B 为例，执行以下命令：
- 在训练阶段，学生模型会学习老师模型的输出
```sh
# 在训练时启用量化
python ft_LoRA_deepSpeed.py \
  --train_mode True \
  --train_bnb_enabled True \
  --student_model_name Qwen/Qwen2.5-3B \
  --teacher_model_name deepseek-chat \
  --ckpt_id 24

# 训练完成后进行推理时启用量化
python ft_LoRA_deepSpeed.py \
  --test_mode True \
  --test_bnb_enabled True \
  --student_model_name Qwen/Qwen2.5-3B \
  --teacher_model_name deepseek-chat \
  --ckpt_id 24
```

6. **其他设置**
- 包括 LoRA、QLoRA、Swanlab 等配置
- 详见 ```deepspeed_ft/README.md```

7. **在B站评论区数据集上的部分实验结果**
- 本 repo 实验设置：3 epochs、1000条数据（训练+测试）、3B模型(bf16)
- 分别基于 **通义千问-Max-Latest** 和 **DeepSeek-V3** 教师模型对学生模型 Qwen/Qwen2.5-3B 进行了蒸馏训练
```sh
# Response: 训练完成的学生模型在测试集上的输出
# Teacher Response：教师模型在测试集上的输出
# Real Response：真实用户评论（已隐去用户昵称）

{
"System Prompt": "你是一个喜欢网上冲浪的B站网友。", 
"User Prompt": "下面我将给出B站某视频及其评论区的相关信息，请根据这些信息，模仿B站网友的说话风格，直接给出对该评论或该视频的可能的回复（限50字）：视频标题: 英特尔Lunar Lake正式发布，能效大提升？\n视频简介: 这次云飞和飘哥来到了德国柏林，参加英特尔酷睿Ultra 200V的发布会，等待已久终于要见到Lunar Lake的真身了！这颗专为轻薄本打造的低功耗芯片到底能有多省电？Windows PC的续航还有看头吗？我们拭目以待……\n父级评论内容: 回复 @xxx :x86直接装Linux不行吗？", 
"Student Response (qwen-plus)": "楼上别带节奏，Linux确实香，但Windows在某些应用生态和兼容性上更有优势呢，而且这波Lunar Lake性能提升也太明显了。", 
"Teacher Response (qwen-plus)": "x86装Linux确实可以，但很多办公软件在Linux上没有很好适配哦，对于普通用户不太友好呢。", 
"Student Response (deepseek-chat)": "回复 @xxx : x86装Linux确实省电，但Windows的生态和兼容性还是香的，轻薄本还得看Windows啊！",
"Teacher Response (deepseek-chat)": "回复 @xxx : x86装Linux当然可以，但驱动和优化可能不如Windows，续航也可能受影响。期待Lunar Lake的表现！",
"Real Response": "回复 @xxx :其实要看测试软件支不支持，用Win还是因为兼容性好" 
}

{
"System Prompt": "你是一个喜欢网上冲浪的B站网友。", 
"User Prompt": "下面我将给出B站某视频及其评论区的相关信息，请根据这些信息，模仿B站网友的说话风格，直接给出对该评论或该视频的可能的回复（限50字）：视频标题: 英特尔酷睿Ultra 200S评测：无药可救！\n视频简介: 时隔三年，英特尔首个架构升级的台式机CPU终于来了！我们拿到了U5 245K、U7 265K和U9 285K三颗新U。本来我们无比期待这次的Arrow Lake处理器，结果他性能居然拉了！甚至除了功耗低之外，相比前代完全没有任何优势。更要命的是，我们想尽办法挽救这代CPU，却没有找到任何能显著改善性能的办法，简直无药可救。到底怎么回事？看完今天的视频你会得到答案……\n父级评论内容: 回复 @xxx :这你就不懂了，13 14缩肛了，名声大损，改名是为了跟缩肛划清界限，不要觉得别人是傻子别人精着呢", 
"Student Response (qwen-plus)": "这不就对了，改名也是为了挽回形象嘛，不过性能不行还是真有点说不过去，感觉英特尔这次有点难啊。", 
"Teacher Response (qwen-plus)": "英特尔这波操作确实让人迷惑，改名就能解决问题了？性能跟不上，怎么改名都是徒劳啊，还是得靠实力说话。", 
"Student Response (deepseek-chat)": "回复 @xxx : 改名确实是为了撇清关系，但性能拉胯也让人尴尬啊，英特尔这波操作有点迷啊🤔",
"Teacher Response (deepseek-chat)": "笑死，改名就能洗白缩肛？英特尔这波操作属实迷惑，建议直接改名“缩肛湖”算了，反正性能也拉胯，改名也救不了😂",
"Real Response": "回复 @xxx : 我周围的人（大部分是小白）其实根本不会太在意英特尔哪一代缩了，他们知道的更多的是英特尔缩了。所以像缩肛这么严重的事情，很多人很可能压根就不会管你是哪一代，只会和英特尔这个品牌挂上钩，真要扭转印象至少也得几代的好口碑……[吃瓜]" 
}
```
