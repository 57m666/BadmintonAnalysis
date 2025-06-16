import cv2
import numpy as np
from ultralytics import YOLO


class PoseDetect:
    def __init__(self):
        self.device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        self.setup_yolo()

    def reset(self):
        self.got_info = False

    def setup_yolo(self):
        # 加载yolov8n-pose模型
        self.model = YOLO('yolo11n-pose.pt')

    def del_yolo(self):
        del self.model

    def get_human_joints(self, frame):
        results = self.model(frame, verbose=False)
        keypoints = []
        for kp in results[0].keypoints.xy:
            keypoints.append(kp.cpu().numpy())
        return results, keypoints

    def draw_key_points(self, filtered_outputs, image, human_limit=-1):
        image_copy = image.copy()
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (4, 6), (3, 5), (6, 8), (8, 10), (11, 12),
                 (5, 7), (7, 9), (5, 11), (11, 13), (13, 15), (6, 12),
                 (12, 14), (14, 16), (5, 6)]

        # top player is blue and bottom one is red
        top_color_edge = (255, 0, 0)
        bot_color_edge = (0, 0, 255)
        top_color_joint = (115, 47, 14)
        bot_color_joint = (35, 47, 204)

        for i in range(len(filtered_outputs)):

            if i > human_limit and human_limit != -1:
                break

            color = top_color_edge if i == 0 else bot_color_edge
            color_joint = top_color_joint if i == 0 else bot_color_joint

            keypoints = np.array(filtered_outputs[i])  # 17, 2
            keypoints = keypoints[:, :].reshape(-1, 2)
            for p in range(keypoints.shape[0]):
                cv2.circle(image_copy,
                           (int(keypoints[p, 0]), int(keypoints[p, 1])),
                           3,
                           color_joint,
                           thickness=-1,
                           lineType=cv2.FILLED)

            for e in edges:
                if (int(keypoints[e[0]][0])!=0 and int(keypoints[e[1]][0])!=0 and
                    int(keypoints[e[0]][1])!=0 and int(keypoints[e[1]][1])!=0):
                    cv2.line(image_copy,
                             (int(keypoints[e[0]][0]), int(keypoints[e[0]][1])),
                             (int(keypoints[e[1]][0]), int(keypoints[e[1]][1])),
                             color,
                             2,
                             lineType=cv2.LINE_AA)
        return image_copy

