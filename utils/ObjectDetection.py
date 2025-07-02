import yaml
import csv
import os
from pathlib import Path
from yolo10 import YOLOv10
from utils.ModelClassifier import ModelClassifier
from PIL import Image
from OCR import YOLOBoxOCR
import numpy as np

class ObjectDetection:
    def __init__(self, config_path: str = "./config/config.yaml"):
        # 設定ファイルを読み込む
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

        # モデルとパラメータを初期化
        self.model = YOLOv10(model=self.config["yolo"]["predict_model"])
        self.device = self.config["yolo"]["device"]
        self.save = self.config["yolo"]["save"]
        self.output_csv = Path(self.config["yolo"].get("output_csv", "./storage/detect/results.csv"))
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.ocr = YOLOBoxOCR()

    def save_ocr_results(self, records, fieldnames):
        file_exists = Path(self.output_csv).exists()
        with open(self.output_csv, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            for row in records:
                writer.writerow(row)
        print(f"✅ OCR 結果を保存しました: {self.output_csv}")

    def __call__(self, image: Image.Image, image_name):
        # print("Received image_name:", image_name)
        self.image_name = image_name
        self.image = image
        classifier = ModelClassifier()
        self.image = classifier(self.image)  # 戻り値が PIL.Image.Image 型であることを確認
        if self.image == None:
            return False

        output_path = Path(self.config["yolo"]["output_path"])
        save_image_name = output_path / image_name

        results = self.model.predict(source=self.image, save=self.save, device=self.device, save_image_name=save_image_name)
        result = results[0]

        all_class_names = []
        ocr_records = []

        boxes = result.boxes
        im = self.image

        xyxy = boxes.xyxy.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy().astype(int)
        names = result.names

        desired_order = [2, 0, 3, 1]
        # cls_idsを並べ替え：desired_orderの順でインデックスを取得
        reordered_indices = []
        for cls in desired_order:
            indices = np.where(cls_ids == cls)[0]
            reordered_indices.extend(indices)
        # cls_idsとxyxyを並び替え
        cls_ids = cls_ids[reordered_indices]
        xyxy = xyxy[reordered_indices]

        ocr_result_dict = {"画像名": self.image_name}

        width, height = im.size  # 画像のサイズ

        for box, cls_id in zip(xyxy, cls_ids):
            class_name = names[cls_id]
            if class_name not in all_class_names:
                all_class_names.append(class_name)

            x1, y1, x2, y2 = map(int, box)
            # 境界を補正
            x1 = max(0, min(x1, width))
            x2 = max(0, min(x2, width))
            y1 = max(0, min(y1, height))
            y2 = max(0, min(y2, height))

            cropped = im.crop((x1, y1, x2, y2))

            try:
                text = self.ocr(cropped, class_name)
            except Exception as e:
                text = f"[OCR エラー: {e}]"

            if class_name in ocr_result_dict:
                ocr_result_dict[class_name] += f" | {text}"
            else:
                ocr_result_dict[class_name] = text

        ocr_records.append(ocr_result_dict)

        fieldnames = ["画像名"] + all_class_names

        self.save_ocr_results(ocr_records, fieldnames)

        return True


# ✅ デバッグ用エントリポイント（このスクリプトが直接実行されたときのみ）
if __name__ == "__main__":
    predictor = ObjectDetection()
    predictor(image)



