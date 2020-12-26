"""
@Author: Dawei Zhu
@Date: 2020-12-18
@Description: 为方便torchtext处理，将jsonl转成json格式
"""

import jsonlines
import json

with jsonlines.open("./train.json", "w") as wfd:
    with open("./train.jsonl", "r", encoding='utf-8') as rfd:
        for data in rfd:
            data = json.loads(data)
            wfd.write(data)


with jsonlines.open("./dev.json", "w") as wfd:
    with open("./dev.jsonl", "r", encoding='utf-8') as rfd:
        for data in rfd:
            data = json.loads(data)
            wfd.write(data)
