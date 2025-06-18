import time

import torch
import torchvision
from tqdm import tqdm
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from argparse import ArgumentParser

sys.path.append("src/models")
sys.path.append("src/tools")

from TrackNet import TrackNet
from ultralytics import YOLO
import pandas as pd
import pickle
import json
from utils import extract_numbers, write_json, read_json

# from src.tools.utils import extract_numbers, write_json, read_json
from denoise import smooth
from event_detection import event_detect
import logging
import traceback

from ..models.TrackNet import TrackNet
from .utils import extract_numbers, write_json, read_json
from .denoise import smooth


# from yolov5 detect.py
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


def ball_detect(video_path, result_path):
    imgsz = [288, 512]
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    orivi_name, start_frame = extract_numbers(video_name)

    cd_save_dir = os.path.join(f"{result_path}/courts", f"court_kp")
    cd_json_path = f"{cd_save_dir}/{orivi_name}.json"
    court = read_json(cd_json_path)["court_info"]

    d_save_dir = os.path.join(f"{result_path}/ball", f"loca_info/{orivi_name}")
    f_source = str(video_path)

    if not os.path.exists(d_save_dir):
        os.makedirs(d_save_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TrackNet().to(device)
    model.load_state_dict(torch.load("src/models/weights/ball_track.pt"))
    model.eval()

    shuttle_tracker = YOLO("src/models/weights/best.pt")
    shuttle_tracker.eval()

    vid_cap = cv2.VideoCapture(f_source)
    video_len = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("video length: ", video_len)
    fps = vid_cap.get(cv2.CAP_PROP_FPS)
    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    count = 0
    results = {}

    with tqdm(total=video_len) as pbar:
        while vid_cap.isOpened() and count < video_len:
            print(
                "Processing frame: vid frame:",
                count,
                int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES)),
            )
            ret, frame = vid_cap.read()
            if not ret:
                break

            # 检查是否还有足够的帧进行TrackNet处理
            current_pos = int(vid_cap.get(cv2.CAP_PROP_POS_FRAMES))
            remaining_frames = video_len - current_pos

            result = shuttle_tracker.track(frame)[0]
            boxes = result.boxes

            if boxes is not None and boxes.conf.numel() > 0:
                # 选置信度最大的框
                max_conf, max_idx = torch.max(boxes.conf, dim=0)
                if max_conf > 0.7:
                    xyxy = boxes.xyxy[max_idx].cpu().numpy()
                    x1, y1, x2, y2 = xyxy.astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    results[f"{count + start_frame}"] = {"visible": 1, "x": cx, "y": cy}

                    count += 1
                    pbar.update(1)
                    continue

                # 置信度不够高，使用TrackNet处理
                elif remaining_frames >= 2:  # 确保还有足够的帧
                    imgs = [frame]  # 包含当前帧

                    # 读取接下来的2帧
                    for i in range(2):
                        ret, img = vid_cap.read()
                        if not ret:
                            break
                        imgs.append(img)

                    # 如果没有读取到足够的帧，处理已有的帧
                    if len(imgs) < 3:
                        # 补充帧数不够的情况，使用最后一帧重复
                        while len(imgs) < 3:
                            imgs.append(imgs[-1])

                    imgs_torch = []
                    for img in imgs:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img_torch = torchvision.transforms.ToTensor()(img).to(device)
                        img_torch = torchvision.transforms.functional.resize(
                            img_torch, imgsz, antialias=True
                        )
                        imgs_torch.append(img_torch)

                    imgs_torch = torch.cat(imgs_torch, dim=0).unsqueeze(0)
                    preds = model(imgs_torch)
                    preds = preds[0].detach().cpu().numpy()

                    y_preds = preds > 0.6
                    y_preds = y_preds.astype("float32")
                    y_preds = y_preds * 255
                    y_preds = y_preds.astype("uint8")

                    # 处理3帧的结果
                    frames_to_process = min(len(imgs), 3)
                    for i in range(frames_to_process):
                        if count >= video_len:
                            break

                        if np.amax(y_preds[i]) <= 0:
                            results[f"{count + start_frame}"] = {
                                "visible": 0,
                                "x": 0,
                                "y": 0,
                            }
                        else:
                            pred_img = cv2.resize(
                                y_preds[i], (w, h), interpolation=cv2.INTER_AREA
                            )

                            (cnts, _) = cv2.findContours(
                                pred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            if cnts:  # 确保找到轮廓
                                rects = [cv2.boundingRect(ctr) for ctr in cnts]
                                max_area_idx = 0
                                max_area = (
                                    rects[max_area_idx][2] * rects[max_area_idx][3]
                                )

                                for ii in range(len(rects)):
                                    area = rects[ii][2] * rects[ii][3]
                                    if area > max_area:
                                        max_area_idx = ii
                                        max_area = area

                                target = rects[max_area_idx]
                                cx_pred = int(target[0] + target[2] / 2)
                                cy_pred = int(target[1] + target[3] / 2)
                                results[f"{count + start_frame}"] = {
                                    "visible": 1,
                                    "x": cx_pred,
                                    "y": cy_pred,
                                }
                            else:
                                results[f"{count + start_frame}"] = {
                                    "visible": 0,
                                    "x": 0,
                                    "y": 0,
                                }

                        count += 1
                        pbar.update(1)

                else:
                    # 剩余帧数不足，标记为不可见
                    results[f"{count + start_frame}"] = {"visible": 0, "x": 0, "y": 0}
                    count += 1
                    pbar.update(1)
            else:
                # 没有检测到目标
                results[f"{count + start_frame}"] = {"visible": 0, "x": 0, "y": 0}
                count += 1
                pbar.update(1)

    # 填充剩余的帧
    while count < video_len:
        results[f"{count + start_frame}"] = {"visible": 0, "x": 0, "y": 0}
        count += 1
        pbar.update(1)

    # 保存结果
    with open(f"{d_save_dir}/{video_name}.json", "w") as f:
        json.dump(results, f)

    # denoise file save path
    dd_save_dir = os.path.join(
        f"{result_path}/ball", f"loca_info(denoise)/{orivi_name}"
    )
    os.makedirs(dd_save_dir, exist_ok=True)

    # smooth trajectory
    try:
        json_path = f"{d_save_dir}/{video_name}.json"
        smooth(json_path, court, dd_save_dir)
    except KeyboardInterrupt:
        print(
            "Caught exception type on main.py ball_detect:",
            type(KeyboardInterrupt).__name__,
        )
        logging.basicConfig(
            filename="logs/error.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.error(traceback.format_exc())
        exit()
    except Exception:
        print("Caught exception type on main.py ball_detect:", type(Exception).__name__)
        logging.basicConfig(
            filename="logs/error.log",
            level=logging.ERROR,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )
        logging.error(traceback.format_exc())

    vid_cap.release()
