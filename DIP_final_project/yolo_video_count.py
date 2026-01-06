import argparse
from collections import Counter

import cv2
from ultralytics import YOLO


def get_class_ids(model, target_names):
    # model.names: {id: "name"}
    name_to_id = {name: i for i, name in model.names.items()}
    ids = []
    for n in target_names:
        if n not in name_to_id:
            raise ValueError(f"Class '{n}' not in model. Try one of: {list(name_to_id.keys())[:20]} ...")
        ids.append(name_to_id[n])
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="giraffe_zebra.mp4", help="input video path, e.g. giraffe_zebra.mp4")
    ap.add_argument("--output", default="output.mp4", help="output video path")
    ap.add_argument("--student-id", default="313512009", help="your student id")
    ap.add_argument("--targets", nargs="+", default=["giraffe", "zebra"], help="target classes, e.g. giraffe zebra")
    ap.add_argument("--model", default="yolov8x-seg.pt", help="yolov8n.pt / yolov8s.pt / yolov8m.pt / yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.9, help="NMS IoU threshold")
    ap.add_argument("--min-area", type=int, default=0, help="filter small boxes by area (0=off)")
    ap.add_argument("--show", action="store_true", help="show realtime window")
    args = ap.parse_args()

    model = YOLO(args.model)
    target_ids = get_class_ids(model, args.targets)

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open input video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))
    if not out.isOpened():
        raise RuntimeError("VideoWriter open failed. Try changing codec or output filename.")

    # BGR 顏色（OpenCV 用 BGR）
    COLOR_MAP = {
        "giraffe": (0, 255, 255),  # 黃
        "zebra":   (255, 0, 255),  # 紫
    }
    DEFAULT_COLOR = (0, 255, 0)    # 綠（保底）

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference (only keep target classes)
        results = model.predict(
            frame,
            conf=args.conf,
            iou=args.iou,
            classes=target_ids,
            max_det=300,
            imgsz=1280,
            verbose=False
        )[0]

        counts = Counter()

        if results.boxes is not None and len(results.boxes) > 0:
            for b in results.boxes:
                cls_id = int(b.cls.item())
                name = model.names[cls_id]
                score = float(b.conf.item())

                x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                area = (x2 - x1) * (y2 - y1)
                if args.min_area > 0 and area < args.min_area:
                    continue

                counts[name] += 1

                # draw bbox + label
                color = COLOR_MAP.get(name, DEFAULT_COLOR)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame, f"{name} {score:.2f}", (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
                )

        # Left-top overlay: student id + per-frame counts
        y = 40
        cv2.putText(frame, str(args.student_id), (20, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        y += 45
        for t in args.targets:
            cv2.putText(frame, f"{t} : {counts.get(t, 0)}", (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3)
            y += 35

        out.write(frame)

        if args.show:
            cv2.imshow("YOLO Detection", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

    cap.release()
    out.release()
    if args.show:
        cv2.destroyAllWindows()
    print(f"Done. Saved: {args.output}")


if __name__ == "__main__":
    main()
