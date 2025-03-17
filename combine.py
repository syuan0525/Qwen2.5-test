import threading
import rospy
from sensor_msgs.msg import Image as ROSImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from PIL import Image as PILImage

import gradio as gr
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ---------------- ROS 部分 -------------------
# 全域變數用來存放最新的圖像 (PIL 格式)
latest_image = None
bridge = CvBridge()

def ros_image_callback(msg):
    global latest_image
    try:
        # 將 ROS 圖像訊息轉換為 OpenCV 圖像 (BGR)
        cv_img = bridge.imgmsg_to_cv2(msg, "bgr8")
        # 轉換成 RGB，再轉為 PIL Image
        cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        latest_image = PILImage.fromarray(cv_img_rgb)
    except Exception as e:
        rospy.logerr("圖像轉換錯誤: %s", e)

def ros_listener():
    # 初始化 ROS 節點 (請確保未重複初始化)
    rospy.init_node('gradio_qwen_listener', anonymous=True)
    # 訂閱 ROS 中攝像頭圖像主題，請根據實際情況調整 topic 名稱
    rospy.Subscriber("/Leader/cv_camera/image_raw", ROSImage, ros_image_callback)
    rospy.spin()

# 以執行緒啟動 ROS 訂閱，確保與 Gradio 介面同時運作
ros_thread = threading.Thread(target=ros_listener)
ros_thread.daemon = True
ros_thread.start()

# ---------------- Qwen2.5-VL 模型與推理函式 -------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
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
    # 若未上傳圖片且有 ROS 圖像，則使用 ROS 最新圖像
    if image is None and latest_image is not None:
        image = latest_image

    # 如果仍無圖像，則回傳錯誤訊息
    if image is None:
        return "尚未收到 ROS 攝像頭圖像，請上傳圖片或等待攝像頭資料。"

    # 預設提示詞
    default_prompt = (
        "請使用繁體中文做物件影像辨識,等只須提供最主要識別到的物件,並給予物件在圖片中的大致座標,"
        "描述方式為例如: 手機,在圖片左上(1); 水壺,在圖片中上(2); 筆,在圖片右上(3); 書,在圖片左方(4); "
        "貓咪,在圖片正中(5); 書,在圖片右方(6); 狗,在圖片左下(7); 貓咪,在圖片中下(8); 鑰匙,在圖片的右下(9) "
        "其中()內的數字代表物品分別表示的位置座標"
    )
    if prompt.strip() == "":
        prompt = default_prompt

    # 構造對話訊息：圖片直接作為輸入
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt}
            ],
        }
    ]
    # 利用處理器生成對話模板文字
    text = processor.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    # 處理圖像與影片資料（視情況而定）
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
    # 移除生成中與輸入重疊的部分
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, 
        skip_special_tokens=True, 
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# ---------------- 建立 Gradio 介面 -------------------
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="上傳圖片 (或使用 ROS 攝像頭的最新影像)"), 
        gr.Textbox(value="", label="提示詞")
    ],
    outputs="text",
    title="Qwen2.5-VL 視覺語言模型示例",
    description="上傳圖片或使用 ROS 攝像頭的最新影像，模型將根據圖片內容生成描述。"
)

# 啟動 Gradio 介面
demo.launch()
