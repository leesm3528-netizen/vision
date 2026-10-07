"""Real-time instance segmentation from a webcam using Ultralytics YOLO26-seg.

Usage:
    python webcam_segment.py
    python webcam_segment.py --model yolo26s-seg.pt --camera 1 --conf 0.4

Press 'q' or ESC to quit.
"""

import argparse
import time

import cv2
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="YOLO26 webcam instance segmentation")
    parser.add_argument("--model", default="yolo26n-seg.pt", help="YOLO26-seg weights (n/s/m/l/x)")
    parser.add_argument("--camera", type=int, default=0, help="webcam index")
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--device", default=None, help="e.g. 'cpu', '0' for GPU (auto if omitted)")
    parser.add_argument("--no-boxes", action="store_true", help="draw masks only, without boxes/labels")
    return parser.parse_args()


def main():
    args = parse_args()

    # Weights are downloaded automatically on first run.
    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open webcam (index {args.camera})")

    window = "YOLO26 Webcam Segmentation"
    prev_time = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Failed to read frame from webcam")
                break

            results = model.predict(
                frame,
                conf=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                retina_masks=True,
                verbose=False,
            )
            result = results[0]
            show_boxes = not args.no_boxes
            annotated = result.plot(boxes=show_boxes, labels=show_boxes, conf=show_boxes)

            now = time.perf_counter()
            fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            num_masks = 0 if result.masks is None else len(result.masks)
            info = f"FPS: {fps:.1f}  Masks: {num_masks}"
            cv2.putText(annotated, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow(window, annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
