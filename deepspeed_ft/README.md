## DeepSpeed 微调训练

1. **LoRA 微调训练**
- 训练/推理脚本二合一，deepspeed的配置文件为 ```deepspeed_config.json```
```sh
python ft_LoRA_deepSpeed.py --train_mode True --test_mode True
```

2. **QLoRA 微调训练**
- 代码支持 4bit QLoRA 训练。配置文件为```bnb_config.json```，通过控制 args 来决定是否在训练和推理中以量化的形式加载，例如：

```sh
# 训练时量化
python ft_LoRA_deepSpeed.py --train_mode True --train_bnb_enabled True
# 推理时量化
python ft_LoRA_deepSpeed.py --test_mode True --test_bnb_enabled True

# Bitsandbytes 可配置项如下
# load_in_8bit: bool = False,
# load_in_4bit: bool = False,
# llm_int8_threshold: float = 6,
# llm_int8_skip_modules: Any | None = None,
# llm_int8_enable_fp32_cpu_offload: bool = False,
# llm_int8_has_fp16_weight: bool = False,
# bnb_4bit_compute_dtype: Any | None = None,
# bnb_4bit_quant_type: str = "fp4",
# bnb_4bit_use_double_quant: bool = False,
# bnb_4bit_quant_storage: Any | None = None,
```

3. **Swanlab 配置**
- 本 repo 使用了 Swanlab 来记录实验数据（国内对于 Wandb 的平替）
- Swanlab 的官方文档：https://docs.swanlab.cn/guide_cloud/general/what-is-swanlab.html

```sh
# 注册账号后登陆到 Swanlab
swanlab login
```

4. **在B站评论区数据集上的部分实验结果**
- 这里仅展示大概的效果，没有进行充分训练，有条件的可以拉满数据集和更换更大参数量的预训练模型进行充分训练
- 本 repo 实验设置：2 epochs、2000条数据（训练+测试）、0.5B模型(bf16)
```sh
{
"System Prompt": "你是一个B站老玩家.", 
"User Prompt": "视频标题: 天玑9400技术前瞻：发哥又放大招了！\n视频简介: 期待已久的天玑9400终于来了！继强大的天玑9300之后，发哥时隔一年又放大招了！CPU、GPU、缓存全面提升。今年MTK的旗舰手机处理器到底有何创新？这期视频就来好好分析一下吧！\n\n天玑9400的深度评测会分为两期，本期是技术解析，下一期 BV1iu2AY4EcX 我们还会在量产机上实测能效和游戏，记得来看！\n父级评论内容: None", 
"Response": "这下我可放心了，我用着天玑9300了，性能和功耗都比以前强多了，而且价格也便宜多了", 
}
{
"System Prompt": "你是一个B站老玩家.", 
"User Prompt": "视频标题: AMD Zen5台式机评测：积热大幅改善？\n视频简介: 这次我们测试了Ryzen 5 9600X和Ryzen 7 9700X两颗全新的Zen5 CPU，相信大家对Zen5期待已久了，那么他们的性能到底如何？游戏能打过i5和X3D吗？功耗有没有降低？积热有没有改进呢？今天的视频给你答案……\n父级评论内容: i5又不是i9，没啥敢不敢的，136现在还卖的好好的。i9在找自己的发票和盒子。", 
"Response": "回复 @xxx :136现在还卖的好好的？", 
}
```