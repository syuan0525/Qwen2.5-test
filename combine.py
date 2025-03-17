import threading
import time
import gradio as gr
import torch
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import roslibpy

# --------------------
# 模型與處理器設定（參考 camera_qwen.py）
# --------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
min_pixels = 256 * 28 * 28
max_pixels = 640 * 28 * 28

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME, 
    min_pixels=min_pixels, 
    max_pixels=max_pixels
)

# --------------------
# ROS 影像接收設定
# --------------------
# 全域變數，用於儲存 ROS 接收到的最新影像（BGR 格式）
ros_image = None

def image_callback(message):
    """
    處理從 ROS 接收到的 sensor_msgs/Image 訊息，並更新全域變數 ros_image
    """
    global ros_image
    try:
        height = message.get('height', 0)
        width = message.get('width', 0)
        encoding = message.get('encoding', 'rgb8')
        data = message.get('data', None)
        if height == 0 or width == 0 or data is None:
            return
        if isinstance(data, str):
            decoded_data = base64.b64decode(data)
            np_data = np.frombuffer(decoded_data, dtype=np.uint8)
        else:
            np_data = np.array(data, dtype=np.uint8)
        if encoding in ['rgb8', 'bgr8']:
            image = np_data.reshape((height, width, 3))
            if encoding == 'rgb8':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image = np_data.reshape((height, width))
        ros_image = image
    except Exception as e:
        print("影像處理錯誤:", e)

def ros_listener():
    """
    建立 roslibpy 客戶端並訂閱 ROS 影像 topic，不斷更新全域變數 ros_image
    """
    ros = roslibpy.Ros(host='localhost', port=9090)
    ros.run()
    image_topic = roslibpy.Topic(ros, '/Leader/cv_camera/image_raw', 'sensor_msgs/Image')
    image_topic.subscribe(image_callback)
    print("已訂閱 ROS 影像 topic，開始接收影像...")
    while ros.is_connected:
        time.sleep(0.1)

# 啟動 ROS 訊息接收的背景執行緒
threading.Thread(target=ros_listener, daemon=True).start()

def get_ros_screenshot():
    global ros_image
    if ros_image is None:
        # 建立一張預設圖片，顯示 "No ROS image received"
        placeholder = Image.new("RGB", (640, 480), (128, 128, 128))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "No ROS image received", fill=(255, 255, 255))
        return placeholder
    image_rgb = cv2.cvtColor(ros_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)
# --------------------
# Qwen2.5-VL 模型推論相關函式
# --------------------
def infer(image, prompt=""):
    """
    模型推論函式，若提示詞為空則使用預設物件辨識提示詞
    """
    default_prompt = (
        "請使用繁體中文做物件影像辨識,等只須提供最主要識別到的物件,並給予物件在圖片中的大致座標,"
        "描述方式為例如: 手機,在圖片左上(1); 水壺,在圖片中上(2); 筆,在圖片右上(3); 書,在圖片左方(4); "
        "貓咪,在圖片正中(5); 書,在圖片右方(6); 狗,在圖片左下(7); 貓咪,在圖片中下(8); 鑰匙,在圖片的右下(9)"
    )
    if prompt.strip() == "":
        prompt = default_prompt

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

def process_screenshot(prompt=""):
    """
    取得目前 ROS 影像，若有則將影像傳入 infer() 進行模型解析，並回傳解析結果
    """
    image = get_ros_screenshot()
    if image is None:
        return "目前尚未收到 ROS 影像，請確認 ROS 訊息是否正確傳送。"
    return infer(image, prompt)

# --------------------
# Gradio 介面建立
# --------------------
with gr.Blocks() as demo:
    gr.Markdown("## ROS 即時影像與截圖推論")
    with gr.Row():
        with gr.Column():
            gr.Markdown("### ROS 即時影像")
            # 顯示 ROS 影像，利用刷新按鈕定時更新
            ros_display = gr.Image(label="ROS 影像")
            # 隱藏的刷新按鈕，設定 elem_id 方便 JavaScript 控制
            refresh_btn = gr.Button("刷新", visible=False, elem_id="refresh_btn")
            # 按下刷新按鈕時更新 ros_display 影像
            refresh_btn.click(fn=get_ros_screenshot, inputs=[], outputs=ros_display)
            # 注入 JavaScript，定時點擊刷新按鈕（每秒更新一次）
            gr.HTML(
                """
                <script>
                    setInterval(function(){
                        document.getElementById("refresh_btn").click();
                    }, 1000);
                </script>
                """
            )
        with gr.Column():
            gr.Markdown("### 截圖推論")
            prompt_input = gr.Textbox(value="", label="提示詞 (留空使用預設)")
            infer_btn = gr.Button("截圖並推論")
            infer_output = gr.Textbox(label="模型解析結果")
            infer_btn.click(fn=process_screenshot, inputs=prompt_input, outputs=infer_output)

demo.launch()
