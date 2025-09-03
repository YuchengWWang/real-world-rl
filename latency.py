import time
import requests
import numpy as np

url = "http://100.72.16.108:5001/upload_raw"

# 生成一张 720x1280x3 的 uint8 彩色图（HWC）
img = np.ones((720, 1280, 3), dtype=np.uint8) * 255  # 白色图
img = np.ascontiguousarray(img)  # 确保内存连续，避免复制

payload = img.tobytes()
headers = {
    "Content-Type": "application/octet-stream",
    # 用自定义头把形状和dtype告诉后端，方便还原
    "X-Shape": "720,1280,3",
    "X-Dtype": "uint8",
}

sess = requests.Session()  # 复用连接，减少TCP建连开销
print(f"原始大小: {len(payload)/1024:.1f} KB")

while True:
    start = time.perf_counter()
    r = sess.post(url, data=payload)
    end = time.perf_counter()
    print(f"HTTP往返延迟: {(end - start) * 1000:.2f} ms  状态码: {r.status_code}")
