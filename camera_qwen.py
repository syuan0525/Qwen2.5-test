import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# 模型名稱變數
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

# 全域載入模型與處理器
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,   # 根據硬件選擇適當精度
    device_map="auto"
)
min_pixels = 256 * 28 * 28
max_pixels = 640 * 28 * 28
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

def infer(image, prompt=""):
    # 如果未提供提示詞，則使用預設提示詞
    default_prompt = (
        "請使用繁體中文做物件影像辨識,等只須提供最主要識別到的物件,並給予物件在圖片中的大致座標,"
        "描述方式為例如: 手機,在圖片左上(1); 水壺,在圖片中上(2); 筆,在圖片右上(3); 書,在圖片左方(4); "
        "貓咪,在圖片正中(5); 書,在圖片右方(6); 狗,在圖片左下(7); 貓咪,在圖片中下(8); 鑰匙,在圖片的右下(9) "
        "等 其中()內的數字代表物品分別表示的位置座標"
    )
    if prompt.strip() == "":
        prompt = default_prompt

    # 構造對話訊息，圖片直接作為輸入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ],
        }
    ]
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
    inputs = inputs.to(model.device)
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 建立 Gradio 介面，Textbox 輸入的值會覆蓋 infer() 中 prompt 的預設值
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="上傳圖片"), 
        gr.Textbox(value="", label="提示詞")  # 預設值為空，使用者可以自行輸入
    ],
    outputs="text",
    title=" Qwen2.5-VL 視覺語言模型示例",
    description="上傳或拍攝一張圖片，模型將根據圖片內容生成描述。"
)

demo.launch()
