import cv2
import numpy as np
import pytesseract


def preprocess_text_spotshade(img_gray, ksize=101):
    """
    對 text-spotshade.tif 做前處理：
    1. 以大尺寸高斯模糊估計背景 (spot shade)
    2. 以除法方式做光照補償，減少中心亮、邊緣暗的現象
    3. Otsu 二值化得到黑白文字影像

    回傳：
        corrected  : 補償後的灰階影像
        bw         : 二值影像
    """
    # 1. 轉成 float，避免除法溢位，加 1 避免除以 0
    img_f = img_gray.astype(np.float32) + 1.0

    # 2. 以大 kernel 高斯模糊估計平滑背景 (spot shade)
    bg = cv2.GaussianBlur(img_f, (ksize, ksize), 0) + 1.0

    # 3. 光照補償：原圖 / 背景 * 128 讓背景變得較平均
    corrected = (img_f / bg) * 128.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)

    # 4. Otsu 二值化
    _, bw = cv2.threshold(corrected, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return corrected, bw


def ocr_from_binary(bw):
    """
    使用二值影像做 OCR：
    1. 統一成黑字白底
    2. 放大 2 倍
    3. 呼叫 Tesseract
    """
    # 判斷目前是白底黑字還是黑底白字
    num_white = np.sum(bw == 255)
    num_black = bw.size - num_white

    # 我們要「黑字白底」給 Tesseract
    if num_white >= num_black:
        img_ocr = bw.copy()      # 背景白、字黑 → OK
    else:
        img_ocr = 255 - bw       # 反相

    # 放大 2 倍，讓線條不要太細
    img_ocr = cv2.resize(
        img_ocr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )

    # Tesseract 設定：--psm 6 當作一般段落文字
    config = "--psm 6"
    text = pytesseract.image_to_string(img_ocr, config=config, lang="eng")

    return text, img_ocr


def clean_ocr_text(text: str) -> str:
    """
    整理 Tesseract 的輸出：
    - 去掉 form feed (\x0c)
    - 去掉開頭/結尾空白行
    - 移除「前後都有文字、自己卻是空白」的孤立空白行
    """
    # 去掉控制字元
    text = text.replace('\x0c', '')

    # 切成一行一行
    lines = text.splitlines()

    # 去掉開頭/結尾空白行
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()

    # 移除中間的孤立空白行
    cleaned_lines = []
    for i, ln in enumerate(lines):
        if ln.strip() != '':
            cleaned_lines.append(ln)
        else:
            prev_has_text = (i > 0 and lines[i - 1].strip() != '')
            next_has_text = (i < len(lines) - 1 and lines[i + 1].strip() != '')
            if prev_has_text and next_has_text:
                # 前後都有文字 → 視為多出來的空行，丟掉
                continue
            cleaned_lines.append(ln)

    # 組回字串，最後補一個換行
    return '\n'.join(cleaned_lines) + '\n'


def main():
    # 讀入 text-spotshade.tif（灰階）
    img_gray = cv2.imread("text-spotshade.tif", cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("找不到 text-spotshade.tif，請確認檔案是否在同一資料夾。")

    # 1. 去除 spot shade ＋ Otsu 二值化
    corrected, bw = preprocess_text_spotshade(img_gray, ksize=101)

    # 這兩張影像可以放進報告中
    cv2.imwrite("text-spotshade_corrected.png", corrected)
    cv2.imwrite("text-spotshade_binary.png", bw)

    # 2. OCR 辨識
    raw_text, img_ocr = ocr_from_binary(bw)
    cv2.imwrite("text-spotshade_ocr_input.png", img_ocr)

    # 3. 整理文字
    final_text = clean_ocr_text(raw_text)

    # 4. 輸出文字檔（作業規定）
    with open("text-spotshade.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    print("text-spotshade.tif 辨識完成，結果已寫入 text-spotshade.txt")
    print("==== OCR 輸出預覽（前幾行）====")
    print('\n'.join(final_text.splitlines()[:10]))


if __name__ == "__main__":
    main()
