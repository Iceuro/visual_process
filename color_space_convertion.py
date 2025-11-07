import os

import matplotlib.pyplot as plt
import numpy as np
import cv2
##### 关于色彩转换函数的一些操作 #####

"""
输入一个HDR视频的YUV路径，转到Rec.2020色域的rgb值
input: yuv_path, yuv的路径
    yuv_type, 指定yuv的形式是yuv422还是420,
output: n_frames×RGB(Rec.2020)
"""
def read_yuv(yuv_path, yuv_type='yuv422p10le', width=3840, height=2160, frame_num=None):
    """
    读取 YUV 文件并返回所有帧的 RGB 值

    参数:
        yuv_path (str): YUV 文件路径
        width (int): 图像宽度
        height (int): 图像高度
        yuv_type (str): 'yuv422' 或 'yuv420'
        frame_num (int, optional): 要读取的帧数，如果为 None 则读取全部镇

    返回:
        np.ndarray: shape 为 (frame_num, height, width, 3)
    """
    # 这里可以看到420p10le的每一帧的大小是分辨率的3(1.5×2)倍，422是4(2×2)倍
    if yuv_type == 'yuv422p10le':
        y_len = width * height
        uv_len = (width // 2) * height
        frame_size_bytes = (y_len + 2 * uv_len) * 2
    elif yuv_type == 'yuv420p10le':
        y_len = width * height
        uv_len = (width // 2) * (height // 2)
        frame_size_bytes = (y_len + 2 * uv_len) * 2
    else:
        raise ValueError(f"Unsupported YUV type: {yuv_type}")

    if frame_num is None:
        file_size = os.path.getsize(yuv_path)
        frame_num = file_size // frame_size_bytes

    frames = []

    with open(yuv_path, 'rb') as f:
        for i in range(frame_num):
            raw = f.read(frame_size_bytes)
            if len(raw) < frame_size_bytes:
                print(f"文件不足，实际读取 {i} 帧")
                break

            # 解码为 uint16 数组
            data = np.frombuffer(raw, dtype=np.uint16)

            # 提取 YUV 分量
            y = data[:y_len].reshape((height, width))

            if yuv_type == 'yuv422p10le':
                u = data[y_len:y_len + uv_len].reshape((height, width // 2))
                v = data[y_len + uv_len:].reshape((height, width // 2))
            elif yuv_type == 'yuv420p10le':
                u = data[y_len:y_len + uv_len].reshape((height // 2, width // 2))
                v = data[y_len + uv_len:].reshape((height // 2, width // 2))

            # 水平 + 垂直双倍插值
            u = cv2.resize(u, (width, height), interpolation=cv2.INTER_LINEAR)
            v = cv2.resize(v, (width, height), interpolation=cv2.INTER_LINEAR)

            frames.append(cv2.merge([y, u, v]))

    return frames


def yuv2rgb_bt2020(yuv, yuv_range='limited', color_space='bt2020'):
    """
    将 BT.2020 标准的 YUV 图像转换为 RGB 图像
    """
    yuv = yuv.astype(np.float64)

    # Y、Cb、Cr 通道归一化 (Limited Range)
    ycbcr = yuv_norm(yuv, yuv_range=yuv_range)
    rgb = ycbcr2rgb(ycbcr, color_space=color_space)

    return rgb


def ycbcr2rgb(ycbcr, color_space='bt2020'):
    y, cb, cr = cv2.split(ycbcr)
    if color_space == 'bt2020':
        # BT.2020 YUV 转 RGB 公式
        r = y + 1.4746 * cr
        g = y - 0.1645531268 * cb - 0.5713531268 * cr
        b = y + 1.8814 * cb
    elif color_space == 'bt709':
        r = y + 1.5748 * cr
        g = y - 0.1873 * cb - 0.4681 * cr
        b = y + 1.8556 * cb
    else:
        raise ValueError(f"Unsupported Color Space: {color_space}")

    rgb = np.clip(cv2.merge([r, g, b]), 0.0, 1.0)

    return rgb


def yuv_norm(yuv, yuv_range='limited'):
    y, u, v = cv2.split(yuv)

    # yuv一般都是limited range的
    if yuv_range == 'limited' or yuv_range == 'tv':
        y = (y - 64) / (940 - 64)
        cb = (u - 512) / (960 - 64)
        cr = (v - 512) / (960 - 64)
    elif yuv_range == 'full':
        y = y / 1023
        cb = (u - 512) / 1023
        cr = (v - 512) / 1023
    else:
        raise ValueError(f"Unsupported Range type: {yuv_range}")

    return cv2.merge([y, cb, cr])


def load_yuv(yuv_path, yuv_type='yuv422p10le', width=3840, height=2160, frame_num=None):
    """
    :param yuv_path: yuv文件的路径
    :param yuv_type: yuv的类型，一般都是yuv420p10le，少数是yuv422p10le的类型
    :param width: 视频的宽度
    :param height: 视频的高度
    :param frame_num: 提取前多少帧，为None提取全部
    :return: 已经转换为rgb的list
    """
    yuv_list = read_yuv(yuv_path, yuv_type=yuv_type,
                        width=width, height=height, frame_num=frame_num)
    rgb_list = [yuv2rgb_bt2020(yuv) for yuv in yuv_list]
    return rgb_list


def rgb2ycbcr(rgb, color_space='bt2020'):
    r, g, b = cv2.split(rgb)
    if color_space == 'bt2020':
        y = 0.2627 * r + 0.6780 * g + 0.0593 * b
        cb = (b - y) / 1.8814
        cr = (r - y) / 1.4746
        # cb = -0.13963 * r + -0.36037 * g + 0.5 * b
        # cr = 0.5 * r - 0.4597857 * g - 0.0402143 * b
    elif color_space == 'bt709':
        y = 0.2126 * r + 0.7152 * g + 0.0722 * b
        cb = -0.1146 * r - 0.3854 * g + 0.5 * b
        cr = 0.5 * r - 0.4542 * g - 0.0458 * b
    else:
        raise ValueError(f"Unsupported Color Space: {color_space}")

    yuv = np.clip(cv2.merge([y, cb, cr]), -1, 1)
    return yuv


def rgb2gray_bt2020(rgb):
    return 0.2627 * rgb[..., 0] + 0.6780 * rgb[..., 1] + 0.0593 * rgb[..., 2]


def rgb2gray_bt709(rgb):
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def load_img(img_path):
    return cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR), cv2.COLOR_BGR2RGB)
