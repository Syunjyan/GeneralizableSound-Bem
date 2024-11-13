import subprocess
from glob import glob
import os
import sys

data_dir = "dataset/NeuPAT/shape"

audio_lst = glob(data_dir + "/*.wav")

for audio in audio_lst:
    # Command to add audio to the video
    base_name = os.path.basename(audio).replace("audio_", "").replace(".wav", "")
    add_audio_command = [
        "ffmpeg",
        "-y",
        "-i",
        f"{data_dir}/video.mp4",  # Input video file
        "-i",
        audio,  # Input audio file
        "-c:v",
        "copy",  # Copy video as is
        "-c:a",
        "aac",  # Audio codec
        "-strict",
        "experimental",
        "-shortest",  # Finish encoding when the shortest input stream ends
        f"{data_dir}/video_with_{base_name}.mp4",  # Final output file
    ]

    # Execute the command
    subprocess.run(add_audio_command)
