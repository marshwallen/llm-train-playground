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
- 这里以 DeepSeek 为例，执行以下命令：
```sh
# 数据会保存在当前目录的 data/ 下
python prepare_data.py --d_type test --model_name deepseek-chat
python prepare_data.py --d_type train --model_name deepseek-chat
```

5. **使用 SFT 蒸馏训练学生模型**
- 这里以 Qwen2.5-3B 为例，执行以下命令：
```sh
# 在训练时启用量化
python ft_LoRA_deepSpeed.py \
  --train_mode True \
  --train_bnb_enabled True \
  --student_model_name Qwen/Qwen2.5-3B \
  --teacher_model_name qwen-plus

# 训练完成后进行推理时启用量化
python ft_LoRA_deepSpeed.py \
  --test_mode True \
  --test_bnb_enabled True \
  --student_model_name Qwen/Qwen2.5-3B \
  --teacher_model_name qwen-plus
```

6. **其他设置**
- 包括 LoRA、QLoRA、Swanlab 等配置
- 详见 ```deepspeed_ft/README.md```

7. **在B站评论区数据集上的部分实验结果**
- 本 repo 实验设置：3 epochs、1000条数据（训练+测试）、3B模型(bf16)
```sh
# Response: 训练完成的Qwen/Qwen2.5-3B在测试集上的输出
# Teacher Response：教师模型在测试集上的输出
{ 
"System Prompt": "你是一个喜欢网上冲浪的B站网友。", 
"User Prompt": "下面我将给出B站某视频及其评论区的相关信息，请根据这些信息，模仿B站网友的说话风格，直接给出对该评论或该视频的可能的回复（限50字）：视频标题: 黑神话悟空性能测试！需要什么显卡才能玩？\n视频简介: 从Steam Deck、780M、1060到3060、4060再到4080S、4090，各档显卡能不能流畅运行黑神话：悟空？能开到多高画质呢？黑神话的光追有何玄机？我们在一天内肝出了答案！一起来看看吧……\n父级评论内容: 回复 @xxx :所以黑的吹的都没什么意义，你管他是什么插帧补帧还是原生，到时候体验好就吹体验不好就退款呗，管他体验效果是通过什么方式来的干嘛？", 
"Response": "楼上说的有道理，游戏体验才是王道。不过现在很多人还是更关注显卡配置呢，毕竟玩游戏不香么😎", 
"Teacher Response": "楼上说得对，玩家要的是游戏体验，管他啥技术呢，只要画面流畅打怪爽就完事了，悟空冲！" 
}
{ 
"System Prompt": "你是一个喜欢网上冲浪的B站网友。", 
"User Prompt": "下面我将给出B站某视频及其评论区的相关信息，请根据这些信息，模仿B站网友的说话风格，直接给出对该评论或该视频的可能的回复（限50字）：视频标题: 英特尔酷睿Ultra 200S评测：无药可救！\n视频简介: 时隔三年，英特尔首个架构升级的台式机CPU终于来了！我们拿到了U5 245K、U7 265K和U9 285K三颗新U。本来我们无比期待这次的Arrow Lake处理器，结果他性能居然拉了！甚至除了功耗低之外，相比前代完全没有任何优势。更要命的是，我们想尽办法挽救这代CPU，却没有找到任何能显著改善性能的办法，简直无药可救。到底怎么回事？看完今天的视频你会得到答案……\n父级评论内容: 回复 @xxx :这你就不懂了，13 14缩肛了，名声大损，改名是为了跟缩肛划清界限，不要觉得别人是傻子别人精着呢", 
"Response": "这不就对了，改名也是为了挽回形象嘛，不过性能不行还是真有点说不过去，感觉英特尔这次有点难啊。", 
"Teacher Response": "英特尔这波操作确实让人迷惑，改名就能解决问题了？性能跟不上，怎么改名都是徒劳啊，还是得靠实力说话。" 
}
```
