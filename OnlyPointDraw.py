import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import argparse
import os
import re
from datetime import datetime
from src.tools.utils import read_json


# --------------------- 可视化绘制函数 ---------------------
def draw_frame(frame_id, players_dict, ball_dict, court_dict, video_size):
    # Adjust figsize and dpi to match pixel size, avoiding Matplotlib's automatic scaling
    fig, ax = plt.subplots(figsize=(video_size[0] / 100, video_size[1] / 100), dpi=100)
    canvas = FigureCanvas(fig)

    try:  # Use try-finally to ensure the figure is closed regardless
        ax.invert_yaxis()  # Matplotlib default y-axis is up, invert here to match image coordinate system

        ax.set_xlim(0, video_size[0])
        ax.set_ylim(video_size[1], 0)

        # --- Court ---
        court = court_dict["court_info"]
        x_court = [p[0] for p in court]
        y_court = [p[1] for p in court]
        ax.scatter(x_court, y_court, c="y")
        for edge in [(0, 1), (2, 3), (4, 5), (0, 4), (1, 5)]:
            ax.plot(
                [x_court[edge[0]], x_court[edge[1]]],
                [y_court[edge[0]], y_court[edge[1]]],
                "y-",
            )

        # --- Net ---
        net = court_dict["net_info"]
        x_net = [p[0] for p in net]
        y_net = [p[1] for p in net]
        ax.scatter(x_net, y_net, c="y")
        for edge in [(0, 1), (1, 2), (2, 3), (0, 3)]:
            ax.plot(
                [x_net[edge[0]], x_net[edge[1]]], [y_net[edge[0]], y_net[edge[1]]], "y-"
            )

        joints = players_dict.get(str(frame_id), None)
        if joints:
            # --- Top player and bottom player ---
            for player_key, color in [("top", "b"), ("bottom", "r")]:
                player_data = joints.get(
                    player_key, None
                )  # Use .get() to safely retrieve player data

                if player_data:  # Only draw if player data is not None
                    # Filter out any None or non-list/tuple keypoints, and ensure they have at least two coordinates (x, y)
                    valid_keypoints = [
                        kp
                        for kp in player_data
                        if kp is not None
                        and isinstance(kp, (list, tuple))
                        and len(kp) >= 2
                    ]

                    if not valid_keypoints:  # If no valid keypoints remain after filtering, skip drawing this player
                        continue

                    # Extract x, y coordinates from valid keypoints
                    x = [p[0] for p in valid_keypoints]
                    y = [p[1] for p in valid_keypoints]

                    # Process ankle joint average: ensure indices 15 and 16 exist in original data and are valid keypoints
                    if (
                        len(player_data) > 16
                        and isinstance(player_data[15], (list, tuple))
                        and len(player_data[15]) >= 2
                        and isinstance(player_data[16], (list, tuple))
                        and len(player_data[16]) >= 2
                    ):
                        x_ankle = (player_data[15][0] + player_data[16][0]) / 2
                        y_ankle = (player_data[15][1] + player_data[16][1]) / 2
                        x.append(int(x_ankle))
                        y.append(int(y_ankle))

                    ax.scatter(x, y, c=color)
                    for i in range(len(x)):
                        # Ensure annotation only happens for valid coordinates (not (0,0) or default/missing values)
                        if x[i] != 0 or y[i] != 0:
                            ax.annotate(
                                str(i),
                                (x[i], y[i]),
                                textcoords="offset points",
                                xytext=(0, 5),
                                ha="center",
                            )

                    # Draw connections (skeleton)
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
                        # Check if edge indices are within the original player_data range
                        # and ensure the points themselves are valid (not None, list/tuple, with x,y coords)
                        if (
                            edge[0] < len(player_data)
                            and edge[1] < len(player_data)
                            and player_data[edge[0]] is not None
                            and isinstance(player_data[edge[0]], (list, tuple))
                            and len(player_data[edge[0]]) >= 2
                            and player_data[edge[1]] is not None
                            and isinstance(player_data[edge[1]], (list, tuple))
                            and len(player_data[edge[1]]) >= 2
                        ):
                            p1_x, p1_y = (
                                player_data[edge[0]][0],
                                player_data[edge[0]][1],
                            )
                            p2_x, p2_y = (
                                player_data[edge[1]][0],
                                player_data[edge[1]][1],
                            )

                            # Only draw connection if both points are valid and not (0,0)
                            if (p1_x, p1_y) != (0, 0) and (p2_x, p2_y) != (0, 0):
                                ax.plot(
                                    [p1_x, p2_x],
                                    [p1_y, p2_y],
                                    color + "-",
                                )

        # --- Ball ---
        ball = ball_dict.get(str(frame_id), None)
        # Only draw ball data if it exists and is visible (assuming 'visible' key in ball_dict)
        if ball and ball.get("visible", 0) == 1:
            ax.scatter(ball["x"], ball["y"], c="g", s=200)
            ax.annotate(
                "ball",
                (ball["x"], ball["y"]),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
            )

        ax.set_title(f"Frame {frame_id}")
        ax.axis("off")  # Turn off axes, only display content
        ax.set_aspect("equal", adjustable="box")  # Maintain aspect ratio

        # --- Save to image and convert to OpenCV format ---
        canvas.draw()
        # Use buffer_rgba instead of tostring_rgb, and handle RGBA format
        img = np.asarray(canvas.buffer_rgba(), dtype=np.uint8)
        # buffer_rgba returns RGBA, so reshape to 4 channels (width, height, RGBA)
        img = img.reshape(canvas.get_width_height()[::-1] + (4,))
        # Convert RGBA to BGR (OpenCV's common format)
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)

        return img

    finally:
        # This block will be executed regardless of what happens in the try block, ensuring the figure is closed
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate video with keypoints, court, and ball data."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Base name for data files (e.g., test1, test2).",
    )
    args = parser.parse_args()

    base_name = args.name

    # Construct file paths
    players_file = f"res/players/player_kp/{base_name}.json"
    court_file = f"res/courts/court_kp/{base_name}.json"
    ball_dir = f"res/ball/loca_info(denoise)/{base_name}/"
    video_info_file = f"res/videos/{base_name}/{base_name}.json"

    # Load video dimensions
    try:
        video_info = read_json(video_info_file)
        width = video_info.get("width")
        height = video_info.get("height")
        if width is None or height is None:
            raise ValueError(f"Width or height not found in {video_info_file}")
        video_size = (width, height)
        print(f"Video size set to: {video_size} from {video_info_file}")
    except FileNotFoundError:
        print(
            f"Error: {video_info_file} not found. Using default video size (1920, 1080)."
        )
        video_size = (1920, 1080)
    except Exception as e:
        print(
            f"Error reading video info from {video_info_file}: {e}. Using default video size (1920, 1080)."
        )
        video_size = (1920, 1080)

    # Load static data
    players_dict = read_json(players_file)
    court_dict = read_json(court_file)

    # List ball data files and let user choose
    ball_files = [f for f in os.listdir(ball_dir) if f.endswith(".json")]
    if not ball_files:
        print(f"No ball data files found in {ball_dir}")
        return

    print(f"\nAvailable ball data files in {ball_dir}:")
    for i, f in enumerate(ball_files):
        print(f"{i + 1}. {f}")

    while True:
        try:
            choice = int(input("Please select a ball data file by number: "))
            if 1 <= choice <= len(ball_files):
                selected_ball_file = os.path.join(ball_dir, ball_files[choice - 1])
                break
            else:
                print("Invalid choice. Please enter a number within the range.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    ball_dict = read_json(selected_ball_file)

    # Extract start and end frames from selected ball file name
    match = re.search(r"_(\d+)-(\d+)\.json", ball_files[choice - 1])
    if match:
        start_frame = int(match.group(1))
        end_frame = int(match.group(2))
        print(f"Frame range extracted from ball file: {start_frame}-{end_frame}")
    else:
        print(
            "Could not extract frame range from ball file name. Using default range (0-443)."
        )
        start_frame = 0
        end_frame = 443

    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_video_path = f"res/onlyPoints_{base_name}_{current_datetime}.mp4"
    video_writer = cv2.VideoWriter(
        output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 20, video_size
    )

    # Main loop to generate video
    for frame_id in range(start_frame, end_frame + 1):
        print(f"Rendering frame {frame_id}")
        try:
            img = draw_frame(frame_id, players_dict, ball_dict, court_dict, video_size)
            # Resize is only necessary if Matplotlib's output doesn't precisely match video_size
            # due to floating point inaccuracies in figsize/dpi calculation, or if aspect ratio changes
            img = cv2.resize(img, video_size)
            video_writer.write(img)
        except Exception as e:
            print(f"Frame {frame_id} error: {e}")
            continue

    video_writer.release()
    print(f"Video saved to {output_video_path}")


if __name__ == "__main__":
    main()
