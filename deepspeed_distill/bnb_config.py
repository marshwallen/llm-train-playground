from transformers import BitsAndBytesConfig
import torch

# AutoModelForCausalLM.from_pretrained 量化配置
def load_bnb_config(enabled):
    """
    此处修改量化配置

    load_in_8bit: bool = False,
    load_in_4bit: bool = False,
    llm_int8_threshold: float = 6,
    llm_int8_skip_modules: Any | None = None,
    llm_int8_enable_fp32_cpu_offload: bool = False,
    llm_int8_has_fp16_weight: bool = False,
    bnb_4bit_compute_dtype: Any | None = None,
    bnb_4bit_quant_type: str = "fp4",
    bnb_4bit_use_double_quant: bool = False,
    bnb_4bit_quant_storage: Any | None = None,
    """

    if enabled:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.half,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
    else:
        return None