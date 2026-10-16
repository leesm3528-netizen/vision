# YOLO26으로 웹캠 실시간 비전 만들기

> **강의노트** · 2026-10-07
> 웹캠 영상으로 **객체 탐지(Object Detection)** 와 **인스턴스 세그멘테이션(Instance Segmentation)** 을 실시간으로 실행합니다. 개발 환경 점검부터 GitHub 업로드까지 하루 동안 진행한 과정을 정리했습니다.

---

## 목차

1. [학습 목표](#1-학습-목표)
2. [개발 환경 점검: 파이썬 버전 문제](#2-개발-환경-점검-파이썬-버전-문제)
3. [라이브러리 설치와 트러블슈팅](#3-라이브러리-설치와-트러블슈팅)
4. [실습 1: 웹캠 객체 탐지](#4-실습-1-웹캠-객체-탐지)
5. [실습 2: 웹캠 세그멘테이션](#5-실습-2-웹캠-세그멘테이션)
6. [GitHub에 프로젝트 올리기](#6-github에-프로젝트-올리기)
7. [정리 및 복습 퀴즈](#7-정리-및-복습-퀴즈)
8. [빠른 실행 가이드](#8-빠른-실행-가이드)

---

## 1. 학습 목표

이 강의를 마치면 다음을 할 수 있습니다.

- [ ] Windows에 파이썬이 여러 개 설치됐을 때 **어떤 파이썬이 실행되는지** 확인할 수 있다.
- [ ] PyTorch DLL 오류(`WinError 1114`)의 원인을 찾아 해결할 수 있다.
- [ ] Ultralytics **YOLO26** 모델로 웹캠 객체 탐지 프로그램을 작성할 수 있다.
- [ ] 같은 구조의 코드를 **세그멘테이션 모델**로 바꿔 적용할 수 있다.
- [ ] `.gitignore`를 작성하고 `gh` CLI로 GitHub 저장소를 만들 수 있다.

---

## 2. 개발 환경 점검: 파이썬 버전 문제

### 증상

> "파이썬을 새로 받았는데 예전 버전이 실행돼요."

### 진단 방법

어떤 실행 파일이 먼저 실행되는지 확인합니다. 아래는 PowerShell 명령입니다.

```powershell
where.exe python        # PATH에서 찾은 python.exe 목록 (위에 있을수록 우선)
python --version        # 실제로 실행되는 버전
py list                 # Python Install Manager가 관리하는 버전 목록
$env:Path -split ';'    # PATH 순서 확인
```

### 진단 결과

| 설치 경로 | 버전 | PATH 등록 |
|---|---|---|
| Python Install Manager (`AppData\Local\Python`) | 3.14.6 (예전 버전) | 등록됨 |
| python.org 설치 파일 (`AppData\Local\Programs\Python\Python314`) | 3.14.8 (새로 받은 버전) | **등록 안 됨** |

### 원인

새 버전을 설치할 때 **"Add python.exe to PATH"** 옵션을 켜지 않았습니다. 그래서 `python` 명령은 PATH에 이미 등록된 예전 설치본을 실행합니다.

### 해결 방법 (둘 중 하나)

1. **Install Manager로 통일 (추천)**: python.org 설치본을 제거하고 `py install --update`를 실행합니다.
2. **python.org 설치본으로 통일**: 설치 파일에서 *Modify*를 눌러 PATH 옵션을 켭니다. 그다음 *설정 → 앱 → 고급 앱 설정 → 앱 실행 별칭*에서 `python.exe`, `python3.exe` 별칭을 끕니다.

> **핵심 포인트**: 파이썬이 여러 개 있으면 PATH에서 **먼저 나오는 것**이 실행됩니다. `pip`도 그 파이썬에 연결되므로 패키지가 엉뚱한 곳에 설치될 수 있습니다. `python -m pip`처럼 실행하면 지금 쓰는 파이썬에 정확히 설치됩니다.

---

## 3. 라이브러리 설치와 트러블슈팅

### 설치

```powershell
python -m pip install -U ultralytics
```

`ultralytics`를 설치하면 `torch`, `opencv-python`, `numpy` 등이 함께 설치됩니다.

| 패키지 | 실습 버전 |
|---|---|
| ultralytics | 8.4.174 |
| torch | 2.14.1 (CPU) |
| opencv-python | 5.0.0 |

### 트러블슈팅: `WinError 1114`

```
OSError: [WinError 1114] DLL 초기화 루틴을 실행할 수 없습니다.
Error loading "...\torch\lib\c10.dll" or one of its dependencies.
```

| 항목 | 내용 |
|---|---|
| **원인** | Windows에 **Microsoft Visual C++ 재배포 패키지**가 없습니다. (`vcruntime140_1.dll` 누락) |
| **확인** | `Test-Path C:\Windows\System32\vcruntime140_1.dll`의 결과가 `False` |
| **해결** | `winget install --id Microsoft.VCRedist.2015+.x64 -e` |

> **핵심 포인트**: `import torch`에서 DLL 오류가 나면 대부분 파이썬 코드가 아니라 **Windows 런타임 라이브러리** 문제입니다.

---

## 4. 실습 1: 웹캠 객체 탐지

파일: [`webcam_detect.py`](webcam_detect.py)

### 개념: 객체 탐지란?

이미지에서 **무엇이(클래스)**, **어디에(바운딩 박스)**, **얼마나 확실하게(신뢰도)** 있는지 찾는 작업입니다.

```
입력 프레임 ──▶ YOLO26 ──▶ [클래스, 박스(x1,y1,x2,y2), 신뢰도] × N개
```

### YOLO26 모델 크기

| 모델 | 특징 |
|---|---|
| `yolo26n.pt` | nano: 가장 빠름, CPU 실시간에 적합 (기본값) |
| `yolo26s.pt` | small |
| `yolo26m.pt` | medium |
| `yolo26l.pt` | large |
| `yolo26x.pt` | extra-large: 가장 정확함, 가장 느림 |

> 가중치 파일은 처음 실행할 때 **자동으로 다운로드**됩니다.

### 코드 흐름

```
① 인자 파싱 ─▶ ② 모델 로드 ─▶ ③ 웹캠 열기
                                   │
       ┌───────────────────────────┘
       ▼
   ④ 프레임 읽기 ─▶ ⑤ 추론 ─▶ ⑥ 결과 그리기 ─▶ ⑦ 화면 출력
       ▲                                             │
       └──────────── q / ESC 누를 때까지 반복 ───────┘
                                   │
                          ⑧ 자원 해제 (finally)
```

### 핵심 코드 해설

**① 모델 로드**

```python
model = YOLO(args.model)   # "yolo26n.pt"
```

**② 웹캠 열기**

```python
cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
```
- `args.camera`: 웹캠 번호입니다. 기본 카메라는 `0`입니다.
- `cv2.CAP_DSHOW`: Windows에서 DirectShow 백엔드를 사용합니다. 카메라가 더 빨리 열립니다.

**③ 추론**

```python
results = model.predict(frame, conf=args.conf, imgsz=args.imgsz,
                        device=args.device, verbose=False)
result = results[0]          # 프레임 1장이므로 결과도 1개
```

| 파라미터 | 의미 | 기본값 |
|---|---|---|
| `conf` | 신뢰도가 이 값보다 낮은 결과는 버림 | 0.25 |
| `imgsz` | 추론할 때 쓰는 입력 이미지 크기 | 640 |
| `device` | `cpu` 또는 `0`(첫 번째 GPU) | 자동 |
| `verbose` | 프레임마다 로그 출력 여부 | False |

**④ 결과 시각화와 FPS 계산**

```python
annotated = result.plot()                 # 박스, 라벨, 신뢰도를 그린 이미지
fps = 1.0 / max(now - prev_time, 1e-6)    # 0으로 나누기 방지
```

**⑤ 종료 처리**

```python
try:
    while True: ...
finally:
    cap.release()            # 오류가 나도 카메라는 반드시 해제
    cv2.destroyAllWindows()
```

> **핵심 포인트**: `try / finally`를 쓰면 프로그램이 비정상 종료돼도 웹캠이 해제되므로 다른 프로그램에서 카메라를 계속 쓸 수 있습니다.

---

## 5. 실습 2: 웹캠 세그멘테이션

파일: [`webcam_segment.py`](webcam_segment.py)

### 개념: 탐지 vs 세그멘테이션

| 구분 | 객체 탐지 | 인스턴스 세그멘테이션 |
|---|---|---|
| 출력 | 바운딩 박스(사각형) | **픽셀 단위 마스크** + 박스 |
| 모델 | `yolo26n.pt` | `yolo26n-seg.pt` |
| 결과 속성 | `result.boxes` | `result.boxes` + `result.masks` |
| 용도 | 위치와 개수 파악 | 정확한 모양, 면적, 배경 분리 |

### 탐지 코드에서 바뀐 부분

전체 구조는 같고 **세 군데**만 다릅니다.

**① 모델 변경**

```python
parser.add_argument("--model", default="yolo26n-seg.pt", ...)
```

**② 고해상도 마스크**

```python
results = model.predict(frame, ..., retina_masks=True)
```
- `retina_masks=True`: 마스크를 원본 해상도로 만들어 경계가 매끄럽습니다. 대신 연산량이 늘어납니다.

**③ 마스크만 보기 옵션과 마스크 개수**

```python
annotated = result.plot(boxes=show_boxes, labels=show_boxes, conf=show_boxes)
num_masks = 0 if result.masks is None else len(result.masks)
```
- 객체가 하나도 없으면 `result.masks`는 `None`입니다. 그래서 `None`인지 먼저 확인해야 합니다.

> **핵심 포인트**: Ultralytics는 작업(task)이 달라도 API가 같습니다. **모델 파일만 바꾸면** 탐지, 세그멘테이션, 포즈 추정으로 전환할 수 있습니다.

---

## 6. GitHub에 프로젝트 올리기

### `.gitignore`: 올리지 않을 파일

```gitignore
*.pt           # 모델 가중치: 용량이 크고 자동으로 다운로드됨
__pycache__/   # 파이썬 캐시
.venv/         # 가상환경
runs/          # Ultralytics 결과물
```

### `requirements.txt`: 의존성 명시

```
ultralytics>=8.4
opencv-python
```

### 업로드 명령

```bash
git add .
git commit -m "Add YOLO26 webcam detection and segmentation scripts"
git branch -M main                                   # 브랜치 이름을 main으로 변경
gh repo create vision --public --source=. --remote=origin --push
```

> **핵심 포인트**: `gh repo create --source=. --push` 한 줄로 **저장소 생성, 원격 연결, 푸시**를 모두 처리합니다.

---

## 7. 정리 및 복습 퀴즈

### 오늘 배운 내용

| 주제 | 핵심 |
|---|---|
| 파이썬 버전 | `where.exe python`으로 PATH 우선순위 확인 |
| torch DLL 오류 | VC++ 재배포 패키지 설치 |
| 객체 탐지 | `YOLO("yolo26n.pt")` → `predict()` → `plot()` |
| 세그멘테이션 | 모델을 `-seg`로 바꾸고 `result.masks` 사용 |
| GitHub | `.gitignore`로 가중치 제외, `gh repo create` |

### 복습 퀴즈

<details>
<summary><b>Q1.</b> 새 파이썬을 설치했는데 <code>python --version</code>에 예전 버전이 나옵니다. 가장 먼저 확인할 것은?</summary>

`where.exe python`으로 PATH에서 어떤 `python.exe`가 먼저 잡히는지 확인합니다.
</details>

<details>
<summary><b>Q2.</b> <code>import torch</code>에서 <code>WinError 1114</code>가 납니다. 원인은?</summary>

Microsoft Visual C++ 재배포 패키지가 없는 경우가 대부분입니다.
</details>

<details>
<summary><b>Q3.</b> 탐지 코드를 세그멘테이션 코드로 바꾸려면 최소한 무엇을 바꿔야 하나요?</summary>

모델 파일을 `yolo26n.pt`에서 `yolo26n-seg.pt`로 바꾸면 됩니다. `result.plot()`이 마스크까지 그려 줍니다.
</details>

<details>
<summary><b>Q4.</b> <code>result.masks</code>를 쓸 때 주의할 점은?</summary>

탐지된 객체가 없으면 `None`이므로 `len()`을 쓰기 전에 `None`인지 확인해야 합니다.
</details>

<details>
<summary><b>Q5.</b> <code>*.pt</code> 파일을 GitHub에 올리지 않는 이유는?</summary>

용량이 크고, Ultralytics가 처음 실행할 때 자동으로 다운로드하기 때문입니다.
</details>

### 도전 과제

- [ ] `--model yolo26n-pose.pt`로 **포즈 추정**을 해 보세요.
- [ ] `model.track(..., persist=True)`로 **객체 추적(ID 부여)** 을 해 보세요.
- [ ] 특정 클래스만 탐지해 보세요. (예: 사람만 `classes=[0]`)
- [ ] `s` 키를 누르면 현재 프레임을 이미지로 저장하는 기능을 추가해 보세요.

---

## 8. 빠른 실행 가이드

### 설치

```bash
pip install -r requirements.txt
```

> Windows에서는 [Visual C++ 재배포 패키지](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)가 필요합니다.
> `winget install Microsoft.VCRedist.2015+.x64`

### 실행

```bash
# 객체 탐지
python webcam_detect.py
python webcam_detect.py --model yolo26s.pt --camera 1 --conf 0.4

# 세그멘테이션
python webcam_segment.py
python webcam_segment.py --no-boxes      # 마스크만 표시
```

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--model` | 모델 가중치 | `yolo26n.pt` / `yolo26n-seg.pt` |
| `--camera` | 웹캠 번호 | `0` |
| `--conf` | 신뢰도 기준값 | `0.25` |
| `--imgsz` | 추론 이미지 크기 | `640` |
| `--device` | `cpu` 또는 `0`(GPU) | 자동 |
| `--no-boxes` | 마스크만 표시 (세그멘테이션 전용) | off |

**종료**: `q` 또는 `ESC`를 누르거나 창을 닫습니다.
