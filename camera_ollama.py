import gradio as gr
import ollama
import tempfile
from PIL import Image

# 將模型名稱設定為變數
MODEL_NAME = "gemma3:12b"

def infer(image, prompt="請描述這張圖片內容:"):
    # 將上傳的 PIL Image 儲存為暫存檔
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        temp_filename = tmp.name
        image.save(temp_filename, format="JPEG")
    
    # 呼叫 Ollama 模型，傳入 prompt 及圖片檔案路徑
    res = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                'role': 'user',
                'content': prompt,
                'images': [temp_filename]
            }
        ]
    )
    
    # 回傳模型生成的描述
    return res['message']['content']

# 建立 Gradio 介面，Textbox 輸入的值會覆蓋 infer() 中 prompt 的預設值
demo = gr.Interface(
    fn=infer,
    inputs=[
        gr.Image(type="pil", label="上傳圖片"), 
        gr.Textbox(value="", label="提示詞")  # 預設值為空，使用者可以自行輸入
    ],
    outputs="text",
    title="Ollama 視覺語言模型示例",
    description="上傳或拍攝一張圖片，模型將根據圖片內容生成描述。"
)

demo.launch()
