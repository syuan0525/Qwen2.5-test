import asyncio
import concurrent.futures
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
# 模型與處理器設定 (參考 run_qwen_model.py)
# -------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-VL-7B-Instruct"
min_pixels = 256 * 28 * 28
max_pixels = 640 * 28 * 28

# 若 GPU 可用，使用 bfloat16 可加速推論
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
# if torch.cuda.is_available():
#     model = model.to("cuda")

# -------------------------------------------------
# ROS 影像接收設定 (參考 test_roslibpy.py)
# -------------------------------------------------
# 全域變數：儲存 ROS 接收到的最新影像 (BGR 格式)
ros_image = None

def image_callback(message):
    """
    處理 ROS 接收到的 sensor_msgs/CompressedImage 訊息，
    將壓縮後的影像資料解碼為 numpy 陣列後存入全域變數 ros_image。
    """
    global ros_image
    try:
        # 取得壓縮影像資料（假設 message['data'] 為 base64 編碼的字串）
        if isinstance(message.get('data', None), str):
            decoded_data = base64.b64decode(message['data'])
        else:
            decoded_data = message['data']
        
        # 將二進位資料轉換為 numpy 陣列
        np_arr = np.frombuffer(decoded_data, np.uint8)
        # 解碼影像 (cv2.IMREAD_COLOR 會回傳 BGR 格式的圖像)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        ros_image = image
    except Exception as e:
        print("影像處理錯誤:", e)


def ros_listener():
    """
    啟動 roslibpy 客戶端並訂閱 ROS 影像 topic，
    持續更新全域變數 ros_image
    """
    ros = roslibpy.Ros(host='localhost', port=9090)
    ros.run()
    image_topic = roslibpy.Topic(ros, '/Leader/cv_camera/image_raw/compressed', 'sensor_msgs/CompressedImage')
    image_topic.subscribe(image_callback)
    print("已訂閱 ROS 影像 topic，等待影像資料...")
    while ros.is_connected:
        time.sleep(0.1)

# 啟動 ROS listener 執行緒
threading.Thread(target=ros_listener, daemon=True).start()

def get_ros_screenshot():
    """
    取得目前 ROS 影像並轉換為 PIL 格式 (RGB)。
    若尚未接收到影像，回傳一張預設佔位圖。
    """
    global ros_image
    if ros_image is None:
        placeholder = Image.new("RGB", (640, 480), (128, 128, 128))
        draw = ImageDraw.Draw(placeholder)
        draw.text((10, 10), "尚未收到 ROS 影像", fill=(255, 255, 255))
        return placeholder
    image_rgb = cv2.cvtColor(ros_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(image_rgb)

# -------------------------------------------------
# 模型推論函式
# -------------------------------------------------
def infer(image, prompt=""):
    """
    使用 Qwen2.5-VL 模型進行推論。
    若提示詞為空，則使用預設物件辨識說明。
    """
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
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )
    return output_text[0]

# -------------------------------------------------
# 管線化各階段函式
# -------------------------------------------------
def acquire_image():
    """
    階段 1：取得 ROS 最新影像
    """
    return get_ros_screenshot()

def preprocess(image):
    """
    階段 2：前處理 (此處可依需求加入圖片縮放、裁剪等處理)
    目前直接傳回原始圖片。
    """
    # 若有需要，可在此處加入額外前處理步驟
    return image

def run_inference(preprocessed_image):
    """
    階段 3：模型推論
    """
    return infer(preprocessed_image, "")

def postprocess(result):
    """
    階段 4：後處理 (例如格式化結果)
    此處直接傳回結果字串。
    """
    return result

# -------------------------------------------------
# 非同步管線函式：依序執行各階段並輸出結果與耗時
# -------------------------------------------------
async def inference_pipeline(loop, executor):
    while True:
        # 階段 1：取得影像
        image = await loop.run_in_executor(executor, acquire_image)
        # 階段 2：前處理
        preprocessed = await loop.run_in_executor(executor, preprocess, image)
        # 階段 3：模型推論
        start_time = time.time()
        inference_result = await loop.run_in_executor(executor, run_inference, preprocessed)
        elapsed_time = time.time() - start_time
        # 階段 4：後處理
        final_output = await loop.run_in_executor(executor, postprocess, inference_result)
        print("【解析結果】", final_output)
        print("推論耗時：{:.2f} 秒".format(elapsed_time))
        # 控制整個 pipeline 的頻率 (例如每 5 秒一個循環)
        await asyncio.sleep(3)

# -------------------------------------------------
# 主程式：啟動非同步管線
# -------------------------------------------------
async def main():
    loop = asyncio.get_running_loop()
    # 使用 ThreadPoolExecutor 執行阻塞性任務
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        await inference_pipeline(loop, executor)

if __name__ == "__main__":
    asyncio.run(main())
