import os
import subprocess
import json
import math


def get_video_info(video_path):
    # cmd = ["ffmpeg", "-i", video_path]
    cmd = [
        "ffprobe",
        "-v", "error",  # 忽略日志信息
        # "-count_frames",
        "-select_streams", "v:0",
        "-show_streams",  # 显示流信息（视频流、音频流等）
        "-print_format", "json",
        video_path
    ]

    # 执行命令并捕获输出
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    metadata = json.loads(result.stdout)

    return metadata


def extract_frames(video_path, output_dir='',
                   seg_num=3, frame_nums=10, all_frames=False):
    """
    从视频中提取分段的帧
    :param video_path: 视频的路径
    :param output_dir: 提取的帧输出的文件夹
    :param seg_num: 分成的段数
    :param frame_nums: 每段取多少帧（取帧从段的中间开始向两边扩展）
    :param all_frames: 用来控制是否提取整个视频的帧，如果为True则提取全部帧
    :return: None
    """
    if not isinstance(video_path, str):
        raise TypeError(f"video path is not str!")

    if not output_dir:
        output_dir = os.path.splitext(video_path)[0] + "_frames"
    os.makedirs(output_dir, exist_ok=True)

    video_metas = get_video_info(video_path)
    if not video_metas or 'streams' not in video_metas or not video_metas['streams']:
        print(f"无法获取 {video_path} 的视频信息")
        return

    video_name = os.path.basename(video_path)

    if all_frames:
        # 如果是all参数，提取所有帧
        cmd = [
            "ffmpeg", "-y",
            "-hwaccel", "cuda",
            "-i", video_path,
            "-vsync", "0",
            os.path.join(output_dir, f"{video_name}_%06d.png")  # 保存所有帧
        ]
    else:
        duration = eval(video_metas['streams'][0]['duration'])
        fps = eval(video_metas['streams'][0]['avg_frame_rate'])

        # ffprobe读取的帧数有时候不正确，通过duration * fps的方式计算得到的帧数更准确
        total_frame_nums = int(duration * fps)
        segment_frame_nums = total_frame_nums // seg_num

        selected_frames = []
        half_range = frame_nums / 2.0

        for i in range(seg_num):
            start_idx = i * segment_frame_nums
            mid = start_idx + segment_frame_nums // 2

            # 连续帧范围
            start_f = math.floor(mid - half_range)
            end_f = start_f + frame_nums
            selected_frames.extend(list(range(start_f, end_f)))

            # 5️⃣ 构建 select 表达式（一次性提取）
            select_expr = "+".join([f"eq(n\\,{f})" for f in selected_frames])

        cmd = [
            "ffmpeg",
            # "-y", "-hwaccel", "cuda",
            "-i", video_path,
            "-vf", f"select='{select_expr}',setpts=N/FRAME_RATE/TB",
            "-vsync", "0",
            os.path.join(output_dir, f"{video_name}_%04d.png")
        ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
