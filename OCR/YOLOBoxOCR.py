from pathlib import Path
from PIL import Image
from paddleocr import PaddleOCR
import numpy as np
import yaml
import re

class YOLOBoxOCR:
    def __init__(self, config_path="./config/config.yaml"):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        self.reader = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    def __call__(self, image: Image.Image, label: str) -> str:
        """
        1枚の切り抜き画像に対してOCRを実行し、labelに応じて結果を解析します。

        :param image: 切り抜き済みの画像 (PIL.Image.Image)
        :param label: ラベル名（例：'register_number'、'total_fee' など）
        :return: 認識されたテキスト（labelに応じて正規表現抽出される場合もあります）
        """
        try:
            # OCR 推論を実行
            ocr_result = self.reader.ocr(np.array(image), cls=False)
            texts = [line[1][0] for line in ocr_result[0]] if len(ocr_result) > 0 else []
            text_raw = "".join(texts).strip()
            text = text_raw  # ✅ デフォルト値

            # 特殊処理
            if "レジ" in label:
                match = re.search(r'#.*$', text_raw)
                text = match.group(0) if match else ''
                text = text.replace("&", "8")

                text = re.sub(r'#(\d)(\d+)', r'#\1 \2', text)
                
            elif "合計" in label:
                match = re.search(r'￥\d+(?:,\d{3})*(?:\.\d+)?', text_raw)
                text = match.group(0) if match else ''
            elif "電話" in label:
                match = re.search(r'[\d-]+$', text_raw)
                text = match.group(0) if match else ''

            print(f"✅ OCR [{label}] -> {text[:30]}...")
            return text

        except Exception as e:
            print(f"❌ OCR 処理失敗: label={label}, エラー: {e}")
            return ""



if __name__ == "__main__":
    from PIL import Image
    ocr_processor = YOLOBoxOCR()

    # 切り抜き画像を読み込む
    cropped_img = Image.open("some_cropped_region.jpg").convert("RGB")

    # ラベル名を指定
    label = "register_number"

    # OCR 実行
    text = ocr_processor(cropped_img, label)

    print("最終的なテキスト：", text)


