# 建立教师模型得到的 Predictions
from teacher_model import TLLM
import json
import os
import argparse
from tqdm import tqdm

def create_teacher_data(args, auto_resume=True):
    """
    获取教师模型的输出
    d_type: train, test
    """
    save_chunk_size = 5

    assert args.d_type in ["train", "test"]
    load_js = args.org_data_dir + f"/data_{args.d_type}.json"
    save_js = f'data/data_{args.d_type}_teacher_{args.model_name}.json'
    with open(load_js, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # 创建 LLM
    llm = TLLM(name=args.model_name)
    
    os.makedirs(os.path.join(os.getcwd(),"data/"), exist_ok=True)
    if auto_resume and os.path.exists(save_js):
        with open(save_js, 'r', encoding='utf-8') as f:
            format_data = json.load(f)
        resume_index = len(format_data)
    else:
        format_data = []
        resume_index = 0

    for i, data in enumerate(tqdm(json_data)):
        if i<resume_index:
            continue

        id = data["id"]
        system_prompt = "你是一个喜欢网上冲浪的B站网友。"
        user_prompt = f"下面我将给出B站某视频及其评论区的相关信息，请根据这些信息，模仿B站网友的说话风格，直接给出对该评论或该视频的可能的回复（限50字）：{data["conversations"][0]["value"]}"

        response = llm.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        format_data.append({
            "id": id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "response": response
        }) 

        if (i+1)%save_chunk_size==0 or i==len(json_data)-1:
            with open(save_js, 'w', encoding='utf-8') as f:
                json.dump(format_data, f, ensure_ascii=False, indent=4)

    print(f"# create_data_{args.d_type}_teacher_{args.model_name} done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--org_data_dir", type=str, default="../data/")
    parser.add_argument("--d_type", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    args = parser.parse_args()

    max_retry = 10
    trys = 0
    while trys < max_retry:
        try:
            create_teacher_data(args)
            break
        except Exception as e:
            print(e)
            trys += 1



