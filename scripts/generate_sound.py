import sys

sys.path.append("./")
import argparse
import torch
from tqdm import tqdm
import json
import numpy as np
from src.SoundGen import generate_sound_static

def main(data_dir, output_file, n_fft, sampling_rate, cache, save_fig, stereo=False):
    generate_sound_static(data_dir, output_file, n_fft=n_fft, sampling_rate=sampling_rate, cache=cache, save_fig=save_fig, stereo=stereo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sound using stable method.")
    parser.add_argument("--data_dir", "-d",type=str, default="dataset/fix", help="Directory of the dataset, for example, dataset/fix")
    parser.add_argument("--output_filename", "-o", type=str, default="audio_1.wav", help="Output audio file name.")
    parser.add_argument("--n_fft", "-n",type=int, default=128, help="Number of FFT components.")
    parser.add_argument("--sampling_rate", "-sr", type=int, default=16000, help="Sampling rate.")
    parser.add_argument("--cache", "-c", type=bool, default=False, help="Use cache or not.")
    parser.add_argument("--figure", "-f", type=bool, default=False, help="Save the spectrograms as figures.")
    parser.add_argument("--stereo", "-s", type=bool, default=False, help="Generate stereo audio.")

    args = parser.parse_args()
    
    main(args.data_dir, args.output_filename, args.n_fft, args.sampling_rate, args.cache, save_fig=args.figure)