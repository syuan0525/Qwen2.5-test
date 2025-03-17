import roslibpy
import numpy as np
import cv2
import base64

def image_callback(message):
    """
    處理從 rosbridge 接收到的 sensor_msgs/Image 訊息。
    如果 data 欄位是字串（base64 編碼），先解碼後再轉換成 numpy 陣列。
    """
    try:
        height = message.get('height', 0)
        width = message.get('width', 0)
        encoding = message.get('encoding', 'rgb8')
        data = message.get('data', None)

        if height == 0 or width == 0 or data is None:
            return

        # 檢查 data 是否為字串，如果是，則表示資料被 base64 編碼
        if isinstance(data, str):
            decoded_data = base64.b64decode(data)
            np_data = np.frombuffer(decoded_data, dtype=np.uint8)
        else:
            # 如果已是列表格式，直接轉換
            np_data = np.array(data, dtype=np.uint8)

        # 根據 encoding 決定圖像的形狀與處理
        if encoding in ['rgb8', 'bgr8']:
            image = np_data.reshape((height, width, 3))
            if encoding == 'rgb8':
                # OpenCV 預設使用 BGR，所以如果是 rgb8，需轉換
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            # 其他編碼情況，簡單當成灰階處理
            image = np_data.reshape((height, width))

        # 顯示圖像
        cv2.imshow("ROS Image", image)
        cv2.waitKey(1)
    except Exception as e:
        print("影像處理錯誤:", e)

# 建立 roslibpy 客戶端與 topic 訂閱
ros = roslibpy.Ros(host='localhost', port=9090)
ros.run()

image_topic = roslibpy.Topic(ros, '/Leader/cv_camera/image_raw', 'sensor_msgs/Image')
image_topic.subscribe(image_callback)

print("已訂閱圖像 topic，等待 ROS 傳送影像資訊...")

# 保持程式運行
try:
    while ros.is_connected:
        pass
except KeyboardInterrupt:
    print("終止訂閱")
    image_topic.unsubscribe()
    ros.terminate()
    cv2.destroyAllWindows()
