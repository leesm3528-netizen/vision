# vision

Real-time webcam object detection and instance segmentation with [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO26.

## Setup

```
pip install -r requirements.txt
```

On Windows, PyTorch needs the [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist) (`winget install Microsoft.VCRedist.2015+.x64`).

## Usage

Object detection:

```
python webcam_detect.py
python webcam_detect.py --model yolo26s.pt --camera 1 --conf 0.4
```

Instance segmentation:

```
python webcam_segment.py
python webcam_segment.py --no-boxes   # masks only
```

Common options: `--model`, `--camera`, `--conf`, `--imgsz`, `--device` (`cpu` or `0` for GPU).
Press `q` or `ESC` to quit. Model weights are downloaded automatically on first run.
