import cv2
import numpy as np
import pytesseract


def preprocess_text_broken(img_gray):
    """
    對 text-broken.tif 做前處理：
    1. Otsu 二值化
    2. 統一成白字黑底
    3. 使用 5x5 形態學 closing 接合破碎文字
    4. 去除太小的雜點
    回傳處理後的「白字黑底」二值影像
    """
    # 1. Otsu 二值化
    _, bw = cv2.threshold(img_gray, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 2. 統一成白字黑底
    num_white = np.sum(bw == 255)
    num_black = bw.size - num_white
    if num_white > num_black:
        bw = 255 - bw

    # 3. 形態學 closing 接合破碎文字
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)

    # 4. 去除太小的雜點
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    min_area = 20
    clean = np.zeros_like(closed)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean  # 白字黑底


def ocr_text_broken(bin_img):
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

    # 3. Tesseract
    config = "--psm 3"
    text = pytesseract.image_to_string(img_ocr, config=config)
    return text, img_ocr


def main():
    img_gray = cv2.imread("text-broken.tif", cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("找不到 text-broken.tif，請確認檔案是否在同一資料夾。")

    # 前處理與儲存影像（報告用）
    bin_clean = preprocess_text_broken(img_gray)
    cv2.imwrite("text-broken_processed.png", bin_clean)

    # OCR
    text, img_ocr = ocr_text_broken(bin_clean)
    cv2.imwrite("text-broken_ocr_input.png", img_ocr)

    # ========== 整理文字：避免多出原圖沒有的空白行 ==========

    # 1. 去掉 form feed 控制字元
    text = text.replace('\x0c', '')

    # 2. 切成一行一行
    lines = text.splitlines()

    # 3. 去掉開頭/結尾的空白行
    while lines and lines[0].strip() == '':
        lines.pop(0)
    while lines and lines[-1].strip() == '':
        lines.pop()

    # 4. 移除「前後都有文字、自己卻是空白」的孤立空白行
    cleaned_lines = []
    for i, ln in enumerate(lines):
        if ln.strip() != '':
            cleaned_lines.append(ln)
        else:
            prev_has_text = (i > 0 and lines[i - 1].strip() != '')
            next_has_text = (i < len(lines) - 1 and lines[i + 1].strip() != '')
            # 若前後都有文字，就判定為多出來的空行 → 丟掉
            if prev_has_text and next_has_text:
                continue
            # 否則保留（例如連續多行空白，用來分段的情況）
            cleaned_lines.append(ln)

    # 5. 組回字串，最後補一個換行
    text = '\n'.join(cleaned_lines) + '\n'

    # =====================================================

    with open("text-broken.txt", "w", encoding="utf-8") as f:
        f.write(text)

    print("辨識完成，結果已寫入 text-broken.txt")
    print("==== OCR 輸出預覽 ====")
    print(text)


if __name__ == "__main__":
    main()
