import threading
import time
import torch
import base64
import numpy as np
import cv2
from PIL import Image, ImageDraw
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import roslibpy

# -------------------------------------------------
# 模型與處理器設定
# -------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"
min_pixels = 256 * 28 * 28
max_pixels = 640 * 28 * 28

# 如果硬體支援 GPU 並使用 float16，可嘗試這樣設定
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # 改用 bfloat16
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    MODEL_NAME,
    min_pixels=min_pixels,
    max_pixels=max_pixels
)

# 將模型移到 GPU (如果可用)
# if torch.cuda.is_available():
#     model = model.to("cuda")

# -------------------------------------------------
# ROS 影像接收設定
# -------------------------------------------------
ros_image = None

def image_callback(message):
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
    ros = roslibpy.Ros(host='localhost', port=9090)
    ros.run()
    image_topic = roslibpy.Topic(ros, '/Leader/cv_camera/image_raw', 'sensor_msgs/Image')
    image_topic.subscribe(image_callback)
    print("已訂閱 ROS 影像 topic，等待影像資料...")
    while ros.is_connected:
        time.sleep(0.1)

threading.Thread(target=ros_listener, daemon=True).start()

def get_ros_screenshot():
    global ros_image
    if ros_image is None:
        placeholder = Image.new("RGB", (640, 480), (128, 128, 128))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "尚未收到 ROS 影像", fill=(255, 255, 255))
        return placeholder
    image_rgb = cv2.cvtColor(ros_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# -------------------------------------------------
# 模型推論函式 (加入預熱)
# -------------------------------------------------
def infer(image, prompt=""):
    default_prompt = (
        "使用繁體中文解析圖片中的資訊"
    )
    if prompt.strip() == "":
        prompt = default_prompt

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt}
        ]
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt"
    )
    if torch.cuda.is_available():
        inputs = inputs.to("cuda")
    generated_ids = model.generate(**inputs, max_new_tokens=128)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# 預熱模型 (可執行一個 dummy inference)
dummy_image = get_ros_screenshot()
_ = infer(dummy_image, "")

# -------------------------------------------------
# 持續解析，計算並輸出推論時間
# -------------------------------------------------
def continuous_inference():
    while True:
        image = get_ros_screenshot()
        start_time = time.time()
        try:
            result = infer(image)
            elapsed_time = time.time() - start_time
            print("【解析結果】", result)
            print("推論耗時：{:.2f} 秒".format(elapsed_time))
        except Exception as e:
            print("推論錯誤：", e)
        time.sleep(5)

threading.Thread(target=continuous_inference, daemon=True).start()

while True:
    time.sleep(1)
