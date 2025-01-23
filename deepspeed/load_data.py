from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorForSeq2Seq
import sys
import json

class CustomLLMDataset(Dataset):
    def __init__(self, args, tokenizer):
        with open(args.data_dir + "/data_train.json", "r", encoding="utf-8") as f:
            self.json_data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = args.max_tokens

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, idx):
        sample = self.json_data[idx]
        input_content = sample["conversations"][0]["value"]
        output_content = sample["conversations"][1]["value"]

        instruction = self.tokenizer(
            f"<|im_start|>system\n你是一个B站老玩家.<|im_end|>\n<|im_start|>user\n{input_content}<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )
        response = self.tokenizer(
            f"{output_content}", 
            add_special_tokens=False
            )
        # input_ids: 将文本通过分词器（tokenizer）处理后得到的token ID序列, 代表了输入文本中的每个token，模型通过这些ID来理解文本内容
        input_ids = (
            instruction["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        )
        # attention_mask: 二进制序列，用于指示模型哪些token是实际输入的一部分，哪些是填充的（padding）
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
        # 模型在训练过程中需要预测的目标序列, 对于语言模型来说，labels 通常是输入序列本身（自回归任务)
        labels = (
            [-100] * len(instruction["input_ids"])
            + response["input_ids"]
            + [self.tokenizer.pad_token_id]
        )

        if len(input_ids) > self.max_length:  # 做一个截断
            input_ids = input_ids[:self.max_length]
            attention_mask = attention_mask[:self.max_length]
            labels = labels[:self.max_length]

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

class DataProcessor:
    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        
    def load_train_data(self):
        """
        读取data_train json文件, 并 tokenize
        """
        # tokenize 化的 train_data
        train_dataset = CustomLLMDataset(self.args, self.tokenizer)
        # 使用 DataCollatorForSeq2Seq 来对其批次数据（不然 DataLoader Batch 中的文本长度不一致时会报错）
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, padding=True)
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True, collate_fn=data_collator)

        return train_dataloader

    def load_test_data(self):
        """
        读取原始测试数据集
        """
        with open(self.args.data_dir + "/data_test.json", "r") as f:
            test_data = json.load(f)

        return test_data