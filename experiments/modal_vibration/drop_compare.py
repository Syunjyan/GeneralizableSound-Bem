import cv2
import numpy as np


import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess


def add_subtitles_and_save_first_frame(
    video_path,
    output_path,
    audio_path,
    subtitle_text_lst,
    font_size_lst,
    relative_position_lst,
    preview=False,
):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font_path = "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path

    font_lst = []
    for font_size in font_size_lst:
        font_lst.append(ImageFont.truetype(font_path, font_size))

    position_lst = []
    for relative_position in relative_position_lst:
        position_lst.append(
            (int(relative_position[0] * width), int(relative_position[1] * height))
        )
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        for font, position, subtitle_text in zip(
            font_lst, position_lst, subtitle_text_lst
        ):
            _, _, text_width, text_height = draw.textbbox(
                (0, 0), subtitle_text, font=font, align="center"
            )
            draw.text(
                (position[0] - text_width / 2, position[1] - text_height / 2),
                subtitle_text,
                font=font,
                fill=(0, 0, 0),
                align="center",
            )

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        if preview:
            cv2.imwrite("test.png", frame)
            cap.release()
            out.release()
            return

        out.write(frame)

    cap.release()
    out.release()
    command = [
        "ffmpeg",
        "-y",
        "-i",
        output_path,
        "-i",
        audio_path,
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        output_path.replace(".mp4", "_with_audio.mp4"),
    ]
    subprocess.run(command)


def cost_time_string(cost_time):
    if cost_time < 1:
        return f"Time Cost: {cost_time * 1000:.0f}ms"
    elif cost_time < 60:
        return f"Time Cost: {cost_time:.0f}s"
    else:
        return f"Time Cost:  {cost_time / 60:.0f} mins"


import sys
import os


def concatenate_videos(video_paths, output_path):
    # Create a temporary file listing all videos
    with open("file_list.txt", "w") as file:
        for path in video_paths:
            file.write(f"file '{path}'\n")

    # Run ffmpeg command
    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        "file_list.txt",
        "-c",
        "copy",
        output_path,
    ]

    subprocess.run(command)

    # Optionally, remove the temporary file list
    os.remove("file_list.txt")


root_dir = sys.argv[-1]
from glob import glob
import json

data_dir_lst = glob(root_dir + "/*")
video_paths = []
for data_dir in data_dir_lst:
    if not os.path.isdir(data_dir):
        continue
    print(data_dir)
    json_path = f"{data_dir}/cost_time.json"
    with open(json_path, "r") as file:
        cost_time = json.load(file)
    print(cost_time)
    bem_cost_time = cost_time["BEM"] / 28.12 * 1.1
    ours_cost_time = cost_time["Poisson 4K"]
    print(bem_cost_time, ours_cost_time)
    add_subtitles_and_save_first_frame(
        f"{data_dir}/video.mp4",
        f"{data_dir}/bem.mp4",
        f"{data_dir}/bem.wav",
        ["BEM", cost_time_string(bem_cost_time)],
        [90, 60],
        [(0.5, 0.05), (0.5, 0.15)],
        preview=False,
    )
    video_paths.append(f"{data_dir}/bem_with_audio.mp4")

    add_subtitles_and_save_first_frame(
        f"{data_dir}/video.mp4",
        f"{data_dir}/ours.mp4",
        f"{data_dir}/ours_4000.wav",
        [
            "Our Monte Carlo Solver",
            cost_time_string(ours_cost_time)
            + f"  ↑{int(bem_cost_time / ours_cost_time)}x",
        ],
        [90, 60],
        [(0.5, 0.05), (0.5, 0.15)],
        preview=False,
    )
    video_paths.append(f"{data_dir}/ours_with_audio.mp4")

concatenate_videos(video_paths, f"{root_dir}/drop_compare.mp4")
