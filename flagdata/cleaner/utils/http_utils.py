import requests

# 定义请求头和请求体
headers = {
    "Content-Type": "application/json"
}
payload = {
    "msg_type": "text",
    "content": {
        "text": "京能数据处理任务异常，任务id为2，具体异常为：java.net.BindException: Cannot assign requested address: Service 'sparkDriver' failed after 16 retries (on a random free port)! Consider explicitly setting the appropriate binding address for the service 'sparkDriver' (for example spark.driver.bindAddress for SparkDriver) to the correct binding address."
    }
}

# 发送 POST 请求
response = requests.post("https://open.feishu.cn/open-apis/bot/v2/hook/27c3cbd8-b0b5-4654-9124-131fc1824e7d",
                         json=payload, headers=headers)

# 打印响应结果
print(response.text)
