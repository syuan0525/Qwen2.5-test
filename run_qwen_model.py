# run_qwen_model.py

# 引入所需模塊
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

def main():
    # 加載模型並指定低精度（可選：如需混合精度可指定 torch.bfloat16 或 torch.float16）
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct",
        torch_dtype=torch.bfloat16,   # 可根據硬件支持選擇 torch.float16
        device_map="auto"
    )
    
    # 設置較小的解析度
    # 原本可能使用 256*28*28 ~ 1280*28*28，這裡我們調整最大像素至 640*28*28
    min_pixels = 256 * 28 * 28      # 最小像素數量保持不變
    max_pixels = 640 * 28 * 28      # 將最大像素數量調低

    # 加載處理器時，傳入解析度參數
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        min_pixels=min_pixels, 
        max_pixels=max_pixels
    )

    # 定義對話消息，這裡我們使用一個圖片和一個文本提示
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                    "image": "/home/pmcl/下載/藪貓標本.jpg",
                },
                {"type": "text", "text": "請說明圖片內容"},
            ],
        }
    ]

    # 利用處理器生成輸入文本，並處理多模態數據
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 將輸入轉移至模型所在設備（如 GPU）
    inputs = inputs.to(model.device)

    # 執行模型推理，這裡設置最大新生成長度為 128 個 tokens
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    # 裁剪掉輸入部分，只保留新生成的部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 解碼生成的 tokens 成文本
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    
    print("模型回答：", output_text)

if __name__ == "__main__":
    main()
