#!/usr/bin/env python3
"""
Audio resampling script for converting RNNoise training data from 48kHz to 8kHz
"""

import argparse
import numpy as np
import soundfile as sf
from scipy import signal
import os
import glob

def resample_audio(input_file, output_file, target_sr=8000):
    """Resample audio file to target sample rate"""
    print(f"Resampling {input_file} to {output_file} at {target_sr}Hz")

    # Read audio file
    audio, sr = sf.read(input_file)

    # If stereo, convert to mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    # Resample to target sample rate
    if sr != target_sr:
        # Calculate resampling ratio
        ratio = target_sr / sr
        # Resample
        audio_resampled = signal.resample(audio, int(len(audio) * ratio))
        # Normalize to prevent clipping
        audio_resampled = audio_resampled / np.max(np.abs(audio_resampled))
    else:
        audio_resampled = audio

    # Write output file
    sf.write(output_file, audio_resampled, target_sr, subtype='PCM_16')
    print(f"Resampled {len(audio)} samples at {sr}Hz to {len(audio_resampled)} samples at {target_sr}Hz")

def resample_directory(input_dir, output_dir, target_sr=8000, pattern="*.wav"):
    """Resample all audio files in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    files = glob.glob(os.path.join(input_dir, pattern))
    print(f"Found {len(files)} audio files to resample")

    for input_file in files:
        filename = os.path.basename(input_file)
        output_file = os.path.join(output_dir, filename)
        resample_audio(input_file, output_file, target_sr)

def main():
    parser = argparse.ArgumentParser(description='Resample audio files for RNNoise training')
    parser.add_argument('input', help='Input audio file or directory')
    parser.add_argument('output', help='Output audio file or directory')
    parser.add_argument('--sample-rate', type=int, default=8000, help='Target sample rate (default: 8000)')
    parser.add_argument('--pattern', default='*.wav', help='File pattern for directory processing (default: *.wav)')

    args = parser.parse_args()

    if os.path.isdir(args.input):
        resample_directory(args.input, args.output, args.sample_rate, args.pattern)
    else:
        resample_audio(args.input, args.output, args.sample_rate)

if __name__ == '__main__':
    main()
