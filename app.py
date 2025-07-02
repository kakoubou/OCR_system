import streamlit as st
from PIL import Image
from utils import ObjectDetection
from pathlib import Path
import time
import csv


def read_csv_summary(csv_path: Path) -> str:
    try:
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            if len(rows) < 2:
                return "⚠️ CSV ファイルの内容が不足しています。"
            header, last_row = rows[0], rows[-1]
            return "\n".join(f"{k}：{v}" for k, v in zip(header, last_row))
    except Exception as e:
        return f"❌ CSV読み込みエラー: {e}"


def show_history_in_sidebar():
    with st.sidebar.expander("🕘 処理履歴", expanded=True):
        history = st.session_state.get("history", [])
        if history:
            for idx, record in enumerate(reversed(history[-5:]), 1):  # 最大5件まで表示
                #st.markdown(f"**{idx}. ファイル名:** `{record['filename']}`")
                st.text(record['summary'])
                st.image(record['image'], caption="処理済み画像", width=150)
        else:
            st.info("履歴がまだありません。")


def add_to_history(filename: str, summary: str, image: Image.Image):
    if "history" not in st.session_state:
        st.session_state.history = []
    # 画像のコピーを保存して、後の参照エラーを防止する
    st.session_state.history.append({
        "filename": filename,
        "summary": summary,
        "image": image.copy()
    })


def main():
    st.set_page_config(page_title="レシート識別システム", layout="centered")
    st.title("レシートOCR識別システム")

    show_history_in_sidebar()

    uploaded_file = st.file_uploader(
        "👉 画像をドラッグ＆ドロップ または クリックしてアップロード",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            predictor = ObjectDetection()

            with st.spinner("🔄 処理中...しばらくお待ちください。"):
                is_receipt = predictor(image, uploaded_file.name)
                time.sleep(1)

            if is_receipt:
                output_dir = Path("./storage/detect")
                output_path = output_dir / uploaded_file.name
                csv_path = output_dir / "results.csv"

                if output_path.exists():
                    result_image = Image.open(output_path)
                    summary_text = read_csv_summary(csv_path)

                    st.markdown("### 📋 OCR 识別結果")
                    st.text(summary_text)

                    display_width = max(10, result_image.width // 5)
                    st.image(result_image, caption="✅ 処理結果　", width=display_width)
                    st.success(" 処理が完了しました！")

                    # 処理成功時のみ履歴に追加する
                    add_to_history(uploaded_file.name, summary_text, result_image)

                else:
                    st.warning(f"⚠️ 処理済み画像が見つかりませんでした: {output_path}")

            else:
                st.markdown(
                    '<p style="font-size:30px; font-weight:bold; color:red;">⚠️ 入力画像はレシートではありません</p>',
                    unsafe_allow_html=True
                )

        except Exception as e:
            st.error(f"❌ 画像処理中にエラーが発生しました: {e}")


if __name__ == "__main__":
    main()









