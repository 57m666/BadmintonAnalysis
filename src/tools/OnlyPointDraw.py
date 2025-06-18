import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from src.tools.utils import read_json
import os

# --------------------- 加载数据 ---------------------
players_dict = read_json("../../res/players/player_kp/test1.json")
ball_dict = read_json("../../res/ball/loca_info(denoise)/test1/test1_273-547.json")
court_dict = read_json("../../res/courts/court_kp/test1.json")

court = court_dict['court_info']
net = court_dict['net_info']

# --------------------- 视频参数 ---------------------
start_frame = 273
end_frame = 547
video_size = (1920, 1080)  # 可以根据你的图像实际大小设置
video_writer = cv2.VideoWriter('../../res/onlyPoints.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20, video_size)

# --------------------- 可视化绘制函数 ---------------------
def draw_frame(frame_id):
    fig, ax = plt.subplots(figsize=(12.8, 9.6))  # figsize 对应 video_size
    canvas = FigureCanvas(fig)
    ax.invert_yaxis()

    ax.set_xlim(0, video_size[0])
    ax.set_ylim(video_size[1], 0)

    # --- 球场 ---
    x = [p[0] for p in court]
    y = [p[1] for p in court]
    ax.scatter(x, y, c='y')
    for edge in [(0, 1), (2, 3), (4, 5), (0, 4), (1, 5)]:
        ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')

    # --- 球网 ---
    x = [p[0] for p in net]
    y = [p[1] for p in net]
    ax.scatter(x, y, c='y')
    for edge in [(0, 1), (1, 2), (2, 3), (0, 3)]:
        ax.plot([x[edge[0]], x[edge[1]]], [y[edge[0]], y[edge[1]]], 'y-')

    joints = players_dict.get(str(frame_id), None)
    if joints:
        # --- 顶部球员 ---
        for player_key, color in [('top', 'b'), ('bottom', 'r')]:
            player = joints[player_key]
            x = [p[0] for p in player]
            y = [p[1] for p in player]
            if len(x) >= 17:
                x.append(int((x[15] + x[16]) / 2))
                y.append(int((y[15] + y[16]) / 2))

            ax.scatter(x, y, c=color)
            for i in range(len(x)):
                ax.annotate(
                    str(i),
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha="center",
                )

            for edge in [
                (0, 1),
                (0, 2),
                (2, 4),
                (1, 3),
                (6, 8),
                (8, 10),
                (11, 12),
                (5, 7),
                (7, 9),
                (5, 11),
                (11, 13),
                (13, 15),
                (6, 12),
                (12, 14),
                (14, 16),
                (5, 6),
            ]:
                if edge[0] < len(x) and edge[1] < len(x):
                    if (x[edge[0]], y[edge[0]]) != (0, 0) and (
                        x[edge[1]],
                        y[edge[1]],
                    ) != (0, 0):
                        ax.plot(
                            [x[edge[0]], x[edge[1]]],
                            [y[edge[0]], y[edge[1]]],
                            color + "-",
                        )

    # --- 球 ---
    ball = ball_dict.get(str(frame_id), None)
    if ball:
        ax.scatter(ball["x"], ball["y"], c="g")
        ax.annotate(
            "ball",
            (ball["x"], ball["y"]),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
        )

    ax.set_title(f"Frame {frame_id}")
    ax.axis("off")

    # --- 保存到图像并转为 OpenCV 格式 ---
    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
    img = img.reshape(canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img


# --------------------- 主循环生成视频 ---------------------
for frame_id in range(start_frame, end_frame + 1):
    print(f"Rendering frame {frame_id}")
    try:
        img = draw_frame(frame_id)
        img = cv2.resize(img, video_size)  # 强制匹配视频尺寸
        video_writer.write(img)
    except Exception as e:
        print(f"Frame {frame_id} error: {e}")
        continue

video_writer.release()
print("Video saved to output.mp4")
