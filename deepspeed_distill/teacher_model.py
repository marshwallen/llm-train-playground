# 教师模型
from openai import OpenAI
import yaml

class TLLM:
    def __init__(self, name:str):
        """
        配置文件在 teacher_model.yaml
        : params name: in ["deepseek-chat", "qwen-plus"]
        """
        with open('teacher_config.yaml', 'r') as file:
            self.cfg = yaml.safe_load(file)

        self.name = name
        self.client = OpenAI(
            api_key=self.cfg["model"][name]['api_key'], 
            base_url=self.cfg["model"][name]['base_url']
            )

    def chat(self, system_prompt, user_prompt, stream=False):
        return self.client.chat.completions.create(      
            model=self.name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=stream
        ).choices[0].message.content