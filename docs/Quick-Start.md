You can download weights in <https://drive.google.com/drive/folders/16mVjXrul3VaXKfHHYauY0QI-SG-JVLvL?usp=sharing>

对于国内用户，可以在下面的链接中下载模型权重和ShuttleSet数据集 [通过网盘分享的文件：BadmintonAnalysis 提取码: u624](https://pan.baidu.com/s/1Eo3f9RtlqxN7cLJVIoreLQ?pwd=u624)

add the weight folder to the "src/models"

add the ShuttleSet to the Project root path "{Project Root Path}/ShuttleSet"

# Creating a Environment

```bash
conda create --name SoloShuttlePose python=3.9
```

# Activate environment

```bash
conda activate SoloShuttlePose
```

# Install torch

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

# Install the other packages

```bash
pip install -r docs/requirements.txt
```

# Install yt-dlp

```bash
pip install yt-dlp
```

# Download the youtube BWF video for ShuttleSet

```python
python src/tools/yt-dlp4ShuttleSet.py
```

# If you want to manually select the valid frames, you can run the following code

```python
python src/tools/FrameSelect.py --folder_path "videos"
```

# Run the following code for player, court ,net detect

Process only unprocessed video.

```python
python main.py --folder_path "videos" --result_path "res"
```

Force processing of all videos, including those that have already been processed.

```python
python main.py --folder_path "videos" --result_path "res" --force
```

# Draw the court,  net, and players

Process only unprocessed video.

```python
python src/tools/VideoDraw.py --folder_path "videos" --result_path "res" --court --net --players --ball
```

Force processing of all videos, including those that have already been processed.

```python
python src/tools/VideoDraw.py --folder_path "videos" --result_path "res" --force --court --net --players --ball --trajectory
```

```python
python -m src.tools.VideoDraw --folder_path "videos" --result_path "res" --force --court --net --players --ball --trajectory
```
