# 下载预训练权重
import argparse
from modelscope import snapshot_download
import os
import shutil

def download(args):
    snapshot_download(args.model, cache_dir='model/', revision='master')
    shutil.rmtree(os.path.join("model", "._____temp"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()
    download(args)
