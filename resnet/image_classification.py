import torch
from PIL import Image

def image_classification(
    model,
    image: Image.Image,
    target_class_idx: int,
    transform
) -> Image.Image | None:
    device = next(model.parameters()).device
    model.eval()

    with torch.no_grad():
        try:
            # 入力画像をテンソルに変換し、バッチ次元を追加してデバイスへ送信
            img_tensor = transform(image).unsqueeze(0).to(device)
            # モデル推論
            outputs = model(img_tensor)
            # 予測クラスのインデックスを取得
            pred = torch.argmax(outputs, dim=1).item()
            # 指定クラスと一致する場合は画像を返す
            if pred == target_class_idx:
                return image  
            else:
                return None
        except Exception as e:
            print(f"画像処理に失敗しました: {e}")
            return None

