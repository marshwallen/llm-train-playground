# 处理爬取的 Bilibili 评论数据

import json
import os
from tqdm import tqdm
import random
import argparse

def get_comment_detail():
    """
    处理获取的 Bilibili 评论数据
    """
    # 读取json文件内容
    contents_dir = [x for x in os.listdir("./data/") if "contents" in x and "json" in x]
    comments_dir = [x for x in os.listdir("./data/") if "comments" in x and "json" in x]

    # 视频信息
    print("Processing video details...")
    video_detail = {}
    for d in contents_dir:
        with open(f'./data/{d}', 'r', encoding='utf-8') as f:
            contents_json = json.load(f)
        for item in tqdm(contents_json):
            video_detail[item["video_id"]] = {
                "create_time": item["create_time"], # 视频发布时间
                "title": item["title"],             # 视频标题
                "desc": item["desc"],               # 视频简介
            }

    # 评论信息
    print("Processing comment details...")
    comment_detail = {}
    for d in comments_dir:
        with open(f'./data/{d}', 'r', encoding='utf-8') as f:
            comments_json = json.load(f)
        for item in tqdm(comments_json):
            comment_detail[item["comment_id"]] = {
                "video_title": video_detail[item["video_id"]]["title"], # 视频标题
                "video_desc": video_detail[item["video_id"]]["desc"],   # 视频简介
                "parent_comment_id": item["parent_comment_id"],         # 父级评论 id
                "content": item["content"],                             # 评论内容
            }
    
    # 写入父级评论内容
    print("Processing parent comment details...")
    for k,v in tqdm(comment_detail.items()):
        # 父级评论不存在时跳过
        if v["parent_comment_id"] in comment_detail.keys():
            comment_detail[k]["parent_content"] = comment_detail[v["parent_comment_id"]]["content"]
        else:
            comment_detail[k]["parent_content"] = "None"
            
    return comment_detail

def init_dataset(args, c_detail: dict):
    """
    构造数据集
    """
    dataset = []
    for k,v in tqdm(c_detail.items()):
        dataset.append({
            "id": k,
            "conversations": [
                {
                    "from": "user",
                    "value": f"视频标题: {v['video_title']}\n视频简介: {v['video_desc']}\n父级评论内容: {v['parent_content']}"
                },
                {
                    "from": "assistant",
                    "value": v["content"]
                }
            ]
        })

    random.shuffle(dataset)
    if args.total > 0:
        dataset = dataset[:args.total]

    train_dataset = dataset[:-10]
    test_dataset = dataset[-10:]

    with open("./data/data_train.json", "w", encoding="utf-8") as f:
        json.dump(train_dataset, f, ensure_ascii=False, indent=4)

    with open("./data/data_test.json", "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=4)

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=-1, help="数据集大小")
    args = parser.parse_args()

    init_dataset(args, get_comment_detail())