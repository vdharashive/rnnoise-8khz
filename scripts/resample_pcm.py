#!/usr/bin/env python3
"""
PCM resampling script for converting RNNoise training data from 48kHz to 8kHz
"""

import argparse
import numpy as np
from scipy import signal
import os
import struct

def resample_pcm(input_file, output_file, input_sr=48000, output_sr=8000):
    """Resample raw PCM file from input_sr to output_sr"""
    print(f"Resampling {input_file} to {output_file} from {input_sr}Hz to {output_sr}Hz")

    # Read raw 16-bit PCM data
    with open(input_file, 'rb') as f:
        pcm_data = f.read()

    # Convert to numpy array (16-bit signed integers)
    audio = np.frombuffer(pcm_data, dtype=np.int16)

    # Convert to float and normalize
    audio_float = audio.astype(np.float32) / 32768.0

    # Resample
    ratio = output_sr / input_sr
    audio_resampled = signal.resample(audio_float, int(len(audio_float) * ratio))

    # Normalize to prevent clipping
    audio_resampled = audio_resampled / np.max(np.abs(audio_resampled))

    # Convert back to 16-bit PCM
    audio_int16 = (audio_resampled * 32767).astype(np.int16)

    # Write output file
    with open(output_file, 'wb') as f:
        f.write(audio_int16.tobytes())

    print(f"Resampled {len(audio)} samples to {len(audio_int16)} samples")

def main():
    parser = argparse.ArgumentParser(description='Resample raw PCM files for RNNoise training')
    parser.add_argument('input', help='Input PCM file')
    parser.add_argument('output', help='Output PCM file')
    parser.add_argument('--input-rate', type=int, default=48000, help='Input sample rate (default: 48000)')
    parser.add_argument('--output-rate', type=int, default=8000, help='Output sample rate (default: 8000)')

    args = parser.parse_args()
    resample_pcm(args.input, args.output, args.input_rate, args.output_rate)

if __name__ == '__main__':
    main()
