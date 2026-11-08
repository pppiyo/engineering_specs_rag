
"""
PDF OCR 人工校对记录

格式：
文件名 -> 页码 -> OCR错误文本 -> 正确文本
"""

OCR_CORRECTIONS = {
    "GB+50060-2008+3~110KV高压配电装置设计规范.pdf": {
        26: [
            ("1.2.4", "7.2.4"),
            ('1.3 ”屋内', '7.3 屋内'),
        ],
    },
    "test_8_13.pdf": {
        3: [
            (
                "宫导体的正常最高工作温度不应大于 70C",
                "裸导体的正常最高工作温度不应大于 70°C"
            )
        ],
        4: [
            (
                "110kV RUF RS AH AL RR",
                "110kV及以下软导线选用钢芯铝绞线"
            ),
            (
                "特种耐热导体的最高工作光度可根据制造厂提供的数据选择",
                "特种耐热导体的最高工作温度可根据制造厂提供的数据选择"
            ),
        ],
        5: [
            (
                "电瓷卜\n电距离",
                "电瓷爬电距离"
            ),
        ],
    },

}


def apply_corrections(text, source, page):
    """
    根据人工核对结果修正 OCR 文本。
    只修改明确记录的内容。
    """
    corrections = OCR_CORRECTIONS.get(source, {}).get(page, [])

    for wrong, correct in corrections:
        text = text.replace(wrong, correct)

    return text
