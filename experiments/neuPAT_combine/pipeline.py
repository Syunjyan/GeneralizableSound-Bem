import sys

data_dir = "dataset/NeuPAT_new/combine_new"

import subprocess

subprocess.run(["blender", "--background", "--python", "BlenderToolbox/combine.py"])
subprocess.run(["python", "experiments/neuPAT/sound.py"])
subprocess.run(
    [
        "python",
        "BlenderToolbox/merge_animation.py",
        data_dir,
    ]
)

subprocess.run(
    [
        "ffmpeg",
        "-y",
        "-i",
        f"{data_dir}/video.mp4",  # Input video file
        "-i",
        f"{data_dir}/audio_1.wav",  # Input audio file
        "-c:v",
        "copy",  # Copy video as is
        "-c:a",
        "aac",  # Audio codec
        "-strict",
        "experimental",
        "-shortest",  # Finish encoding when the shortest input stream ends
        f"{data_dir}/video_with_sound.mp4",  # Final output file
    ]
)
