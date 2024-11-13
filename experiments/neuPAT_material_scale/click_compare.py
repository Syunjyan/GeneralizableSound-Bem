import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import subprocess
from tqdm import tqdm


def create_video_from_image(
    image_path,
    output_path,
    audio_path,
    duration,
    subtitle_text_lst,
    font_size_lst,
    relative_position_lst,
    preview=False,
):
    # Load the image with alpha channel
    image_pil = Image.open(image_path).convert("RGBA")
    width, height = image_pil.size

    # Calculate fps for a given duration
    fps = 60  # Standard fps; you can adjust it as needed
    total_frames = int(duration * fps)

    # Create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    font_path = "/home/jxt/.local/share/fonts/LinBiolinum_R.ttf"  # Replace with your font file path
    font_lst = [ImageFont.truetype(font_path, font_size) for font_size in font_size_lst]

    position_lst = [
        (int(relative_position[0] * width), int(relative_position[1] * height))
        for relative_position in relative_position_lst
    ]
    frame_pil = image_pil.copy()

    # Convert PIL image to numpy array
    frame = np.array(frame_pil)

    # Add background if the image has alpha channel
    if frame.shape[2] == 4:
        background = np.ones((height, width, 3), dtype=np.uint8) * [
            180,
            180,
            200,
        ]  # Grey background
        alpha_channel = frame[:, :, 3]
        rgb = frame[:, :, :3]
        alpha_factor = alpha_channel[:, :, np.newaxis].astype(np.float32) / 255.0
        background_factor = 1.0 - alpha_factor
        frame = (alpha_factor * rgb + background_factor * background).astype(np.uint8)
    # Convert back to PIL Image for adding text
    frame_pil = Image.fromarray(frame)
    draw = ImageDraw.Draw(frame_pil)
    for font, position, subtitle_text in zip(font_lst, position_lst, subtitle_text_lst):
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
    for _ in tqdm(range(total_frames)):
        out.write(frame)

        if preview and _ == 0:
            cv2.imwrite("preview.png", frame)

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
        "-shortest",
        output_path.replace(".mp4", "_with_audio.mp4"),
    ]
    subprocess.run(command)


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


root_dir = "dataset/NeuPAT_new/scale/test"
from glob import glob

data_dir_lst = glob(root_dir + "/*")
video_paths = []


def click_video(click_camera_image):
    create_video_from_image(
        click_camera_image,
        click_camera_image.replace(".png", "_bem.mp4"),
        click_camera_image.replace(".png", "_bem.wav"),
        duration,
        ["BEM"],
        [90],
        [[0.5, 0.05]],
    )
    video_paths.append(click_camera_image.replace(".png", "_bem_with_audio.mp4"))

    create_video_from_image(
        click_camera_image,
        click_camera_image.replace(".png", "_neuPAT.mp4"),
        click_camera_image.replace(".png", "_neuPAT.wav"),
        duration,
        ["Ours"],
        [90],
        [[0.5, 0.05]],
    )
    video_paths.append(click_camera_image.replace(".png", "_neuPAT_with_audio.mp4"))

    create_video_from_image(
        click_camera_image,
        click_camera_image.replace(".png", "_NeuralSound.mp4"),
        click_camera_image.replace(".png", "_NeuralSound.wav"),
        duration,
        ["NeuralSound [Jin et al. 2022]"],
        [90],
        [[0.5, 0.05]],
    )
    video_paths.append(
        click_camera_image.replace(".png", "_NeuralSound_with_audio.mp4")
    )


for data_dir in data_dir_lst:
    if not os.path.isdir(data_dir):
        continue
    print(data_dir)
    click_lst = glob(data_dir + "/click/*_0.png")
    click_lst.sort()
    click_lst = click_lst[::-1]
    idx = 0
    duration = 2
    for click_image in click_lst:
        if idx == 2:
            for i in range(3):
                click_camera_image = click_image.replace("_0.png", f"_{i}.png")
                click_video(click_camera_image)
        else:
            click_camera_image = click_image
            click_video(click_camera_image)
        idx += 1

concatenate_videos(video_paths, f"{root_dir}/../mat_size_edit.mp4")
