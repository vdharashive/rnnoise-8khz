#!/usr/bin/env python3
"""
PCM resampling script for converting RNNoise training data from 48kHz to 8kHz
Supports large files by processing in chunks to avoid memory errors
"""

import argparse
import numpy as np
from scipy import signal
import os
import struct

def resample_pcm_chunked(input_file, output_file, input_sr=48000, output_sr=8000, chunk_size_mb=100):
    """Resample raw PCM file from input_sr to output_sr using chunked processing"""
    print(f"Resampling {input_file} to {output_file} from {input_sr}Hz to {output_sr}Hz")
    print(f"Using chunked processing with {chunk_size_mb}MB chunks")

    # Calculate chunk size in samples (16-bit = 2 bytes per sample)
    chunk_samples = (chunk_size_mb * 1024 * 1024) // 2
    ratio = output_sr / input_sr

    # Get input file size
    input_size = os.path.getsize(input_file)
    total_samples = input_size // 2  # 16-bit samples
    processed_samples = 0

    print(f"Input file size: {input_size / (1024**3):.2f} GB")
    print(f"Total samples: {total_samples}")
    print(f"Expected output samples: {int(total_samples * ratio)}")

    with open(input_file, 'rb') as f_in, open(output_file, 'wb') as f_out:
        while True:
            # Read chunk
            chunk_data = f_in.read(chunk_samples * 2)  # 2 bytes per sample
            if not chunk_data:
                break

            # Convert to numpy array (16-bit signed integers)
            audio_chunk = np.frombuffer(chunk_data, dtype=np.int16)

            # Convert to float and normalize
            audio_float = audio_chunk.astype(np.float32) / 32768.0

            # Resample this chunk
            resampled_chunk = signal.resample(audio_float, int(len(audio_float) * ratio))

            # Normalize to prevent clipping (per chunk to maintain quality)
            if np.max(np.abs(resampled_chunk)) > 1.0:
                resampled_chunk = resampled_chunk / np.max(np.abs(resampled_chunk))

            # Convert back to 16-bit PCM
            audio_int16 = (resampled_chunk * 32767).astype(np.int16)

            # Write chunk to output
            f_out.write(audio_int16.tobytes())

            processed_samples += len(audio_chunk)
            progress = processed_samples / total_samples * 100
            print(".1f")

    print(f"Resampling completed. Output file: {output_file}")

def resample_pcm(input_file, output_file, input_sr=48000, output_sr=8000, chunk_size_mb=100):
    """Resample raw PCM file from input_sr to output_sr - automatically chooses chunked processing for large files"""
    file_size_gb = os.path.getsize(input_file) / (1024**3)

    # Use chunked processing for files larger than 1GB
    if file_size_gb > 1.0:
        resample_pcm_chunked(input_file, output_file, input_sr, output_sr, chunk_size_mb)
    else:
        # Original method for smaller files
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
    parser.add_argument('--chunk-size', type=int, default=100, help='Chunk size in MB for large files (default: 100)')

    args = parser.parse_args()
    resample_pcm(args.input, args.output, args.input_rate, args.output_rate, args.chunk_size)

if __name__ == '__main__':
    main()
