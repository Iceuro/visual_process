import os
import subprocess
import json
import math

import numpy as np


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


def generate_segment_indices(total_frames, seg_num=3, frame_nums=10):
    """

    :param total_frames: 视频总共的帧数
    :param seg_num: 切分的片段
    :param frame_nums: 每个片段切分的帧数
    :return:
    """
    segment_frame_nums = total_frames // seg_num

    indices = []
    half_range = frame_nums // 2

    for i in range(seg_num):
        start = i * segment_frame_nums
        mid = start + segment_frame_nums // 2
        start_f = mid - half_range
        end_f = start_f + frame_nums
        indices.extend(range(start_f, end_f))

    return sorted(set(indices))


def extract_frames(video_path, output_dir='',
                   seg_num=3, frame_nums=10, all_frames=False):
    """
    Extract frames(.png) from video(.mp4)
    从视频中提取图片帧
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
            os.path.join(output_dir, f"{video_name}_%05d.png")  # 保存所有帧
        ]
    else:
        duration = eval(video_metas['streams'][0]['duration'])
        fps = eval(video_metas['streams'][0]['avg_frame_rate'])

        # ffprobe读取的帧数有时候不正确，通过duration * fps的方式计算得到的帧数更准确
        total_frames = int(fps * duration)

        selected_frames = generate_segment_indices(total_frames, seg_num=seg_num,
                                                   frame_nums=frame_nums)

        # 构建 select 表达式
        select_expr = "+".join([f"eq(n\\,{i})" for i in selected_frames])

        output_pattern = os.path.join(output_dir, f"{video_name}_%05d.png")

        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-vf", f"select='{select_expr}'",
            "-vsync", "0",
            "-frame_pts", "1",
            output_pattern
        ]

        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def video2yuv(video_path, output_dir='convert_video_out'):
    """
    Convert video to YUV files
    将视频转换为YUV文件
    :param video_path: 输入视频路径
    :param out_dir: 输出文件夹路径
    :return:
    """

    os.makedirs(output_dir, exist_ok=True)

    infos = get_video_info(video_path)['streams'][0]
    pix_fmt = infos['pix_fmt']
    # color_primaries = infos.get('color_primaries', 'bt2020')
    # colorspace = infos.get('color_primaries', 'bt2020nc')
    # color_trc = infos.get('color_primaries', 'smpte2084')

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    yuv_path = os.path.join(output_dir, f"{base_name}.yuv")

    cmd_ffmpeg = [
        "ffmpeg", "-i", video_path,
        "-pix_fmt", pix_fmt,
        "-vframes", '50', # 可以指定切前多少帧
        # "-color_primaries", color_primaries,
        # "-colorspace", colorspace,
        # "-color_trc", color_trc,
        "-f", "rawvideo", yuv_path,
    ]

    print(f"Converting to {pix_fmt} → {yuv_path}")
    subprocess.run(cmd_ffmpeg, check=True)
    print("✅ Conversion finished!")
