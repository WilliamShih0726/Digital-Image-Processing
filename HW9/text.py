import cv2
import numpy as np
import pytesseract


def preprocess_text_clear_border(img_gray):
    """
    對 text.tif 做前處理：
    1. Otsu 二值化
    2. 統一成白字黑底
    3. 利用連通元件標記，刪除所有接觸影像邊緣的字元
    回傳處理後的「白字黑底」二值影像
    """
    # 1. Otsu 二值化
    _, bw = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 統一成白字黑底
    num_white = np.sum(bw == 255)
    num_black = bw.size - num_white
    # 若白色比較多，通常是白底黑字 → 反相成白字黑底
    if num_white > num_black:
        bw = 255 - bw

    h, w = bw.shape

    # 3. 連通元件標記
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw)

    # 準備輸出影像，先全部設為黑
    result = np.zeros_like(bw)

    # 0 是背景，從 1 開始走訪每個元件
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        ww = stats[i, cv2.CC_STAT_WIDTH]
        hh = stats[i, cv2.CC_STAT_HEIGHT]

        # 算出此元件 bounding box 的四邊座標
        left = x
        top = y
        right = x + ww - 1
        bottom = y + hh - 1

        # 檢查是否碰觸到影像邊界
        touches_border = (
            left == 0 or
            top == 0 or
            right == w - 1 or
            bottom == h - 1
        )

        if touches_border:
            # 接觸邊界 → 整個字元刪除（不寫進 result）
            continue

        # 不接觸邊界 → 保留在結果影像中
        result[labels == i] = 255

    return result  # 白字黑底，邊緣字母已被清除


def ocr_text(bin_img):
    """
    對前處理後的二值影像做 OCR：
    1. 轉成黑字白底
    2. 放大 2 倍
    3. 呼叫 Tesseract
    """
    # 1. 白字黑底 → 黑字白底
    img_ocr = 255 - bin_img

    # 2. 放大
    img_ocr = cv2.resize(
        img_ocr, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_LINEAR
    )

    # 3. 呼叫 Tesseract
    # --psm 3：把整張視為一個文字區塊
    config = "--psm 3"
    text = pytesseract.image_to_string(img_ocr, config=config)
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

    # 移除中間「多出來」的孤立空白行
    cleaned_lines = []
    for i, ln in enumerate(lines):
        if ln.strip() != '':
            cleaned_lines.append(ln)
        else:
            prev_has_text = (i > 0 and lines[i - 1].strip() != '')
            next_has_text = (i < len(lines) - 1 and lines[i + 1].strip() != '')
            # 前後都有文字 → 判定為多出來的空行，直接丟掉
            if prev_has_text and next_has_text:
                continue
            cleaned_lines.append(ln)

    # 再組回字串，並在最後補一個換行
    return '\n'.join(cleaned_lines) + '\n'


def main():
    # 讀入 text.tif（灰階）
    img_gray = cv2.imread("text.tif", cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("找不到 text.tif，請確認檔案是否在同一資料夾。")

    # 前處理：刪除碰到邊緣的字母
    bin_clean = preprocess_text_clear_border(img_gray)

    # 儲存清除後的影像（白字黑底），可放到報告中
    cv2.imwrite("text_processed.png", bin_clean)

    # OCR 辨識
    raw_text, img_ocr = ocr_text(bin_clean)

    # 儲存送進 Tesseract 的影像（黑字白底、放大後）
    cv2.imwrite("text_ocr_input.png", img_ocr)

    # 整理文字：避免多出原圖沒有的空白行
    final_text = clean_ocr_text(raw_text)

    # 寫入 text.txt（作業要求）
    with open("text.txt", "w", encoding="utf-8") as f:
        f.write(final_text)

    print("text.tif 辨識完成，結果已寫入 text.txt")
    print("==== OCR 輸出預覽（前幾行）====")
    print('\n'.join(final_text.splitlines()[:10]))


if __name__ == "__main__":
    main()
