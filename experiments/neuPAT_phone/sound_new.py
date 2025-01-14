import sys

sys.path.append("./")
import argparse
import torch
from tqdm import tqdm
import json
import numpy as np
from src.SoundGen import generate_sound_static

def main(data_dir, output_file, n_fft, sampling_rate, cache):
    generate_sound_static(data_dir, output_file, n_fft=n_fft, sampling_rate=sampling_rate, cache=cache)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound using stable method.")
    parser.add_argument("--data_dir", "-d",type=str, default="dataset/fix", help="Directory of the data.")
    parser.add_argument("--output_file", "-o", type=str, default="audio_1.wav", help="Output audio file name.")
    parser.add_argument("--n_fft", "-n",type=int, default=128, help="Number of FFT components.")
    parser.add_argument("--sampling_rate", "-sr", type=int, default=16000, help="Sampling rate.")
    parser.add_argument("--cache", "-c", type=bool, default=False, help="Use cache or not.")

    args = parser.parse_args()
    
    main(args.data_dir, args.output_file, args.n_fft, args.sampling_rate, args.cache)