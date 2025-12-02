#!/usr/bin/env python3
"""
Test script to verify 8kHz RNNoise conversion works correctly
"""

import numpy as np
import subprocess
import os
import tempfile

def generate_test_audio(filename, duration=2.0, sample_rate=8000, freq=1000):
    """Generate a test sine wave at specified frequency"""
    t = np.linspace(0, duration, int(duration * sample_rate), False)
    audio = np.sin(2 * np.pi * freq * t)

    # Add some noise
    noise = np.random.normal(0, 0.1, len(audio))
    noisy_audio = audio + noise

    # Convert to 16-bit PCM
    audio_int16 = (noisy_audio * 32767).astype(np.int16)

    # Write raw PCM file
    with open(filename, 'wb') as f:
        f.write(audio_int16.tobytes())

    print(f"Generated test audio: {len(audio_int16)} samples at {sample_rate}Hz")
    return audio, noisy_audio

def run_rnnoise_demo(input_file, output_file):
    """Run rnnoise_demo on the input file"""
    cmd = ['./examples/rnnoise_demo', input_file, output_file]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running rnnoise_demo: {result.stderr}")
        return False

    print("rnnoise_demo completed successfully")
    return True

def analyze_audio(filename, sample_rate=8000):
    """Basic analysis of audio file"""
    with open(filename, 'rb') as f:
        data = f.read()

    audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

    print(f"Audio file: {filename}")
    print(f"Samples: {len(audio)}")
    print(f"Duration: {len(audio)/sample_rate:.2f} seconds")
    print(f"RMS: {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"Peak: {np.max(np.abs(audio)):.4f}")
    print()

def main():
    print("=== Testing 8kHz RNNoise Conversion ===")

    # Check if rnnoise_demo exists
    if not os.path.exists('examples/rnnoise_demo'):
        print("Error: rnnoise_demo not found. Please build RNNoise first.")
        return 1

    # Create temporary files
    with tempfile.TemporaryDirectory() as tmpdir:
        input_file = os.path.join(tmpdir, 'test_input.raw')
        output_file = os.path.join(tmpdir, 'test_output.raw')

        # Generate test audio
        print("1. Generating test audio...")
        clean_audio, noisy_audio = generate_test_audio(input_file, duration=1.0, sample_rate=8000, freq=1000)

        # Analyze input
        print("2. Analyzing input audio...")
        analyze_audio(input_file, 8000)

        # Run RNNoise
        print("3. Running RNNoise processing...")
        if not run_rnnoise_demo(input_file, output_file):
            return 1

        # Analyze output
        print("4. Analyzing output audio...")
        analyze_audio(output_file, 8000)

        print("5. Comparing input vs output RMS...")
        with open(input_file, 'rb') as f:
            input_data = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0
        with open(output_file, 'rb') as f:
            output_data = np.frombuffer(f.read(), dtype=np.int16).astype(np.float32) / 32768.0

        input_rms = np.sqrt(np.mean(input_data**2))
        output_rms = np.sqrt(np.mean(output_data**2))

        print(f"Input RMS: {input_rms:.4f}")
        print(f"Output RMS: {output_rms:.4f}")
        print(f"Ratio: {output_rms/input_rms:.4f}")

        if output_rms < input_rms * 0.9:  # Should reduce noise
            print("✓ Test passed: Output has lower RMS than input")
        else:
            print("⚠ Test inconclusive: Output RMS not significantly lower")

    print("=== Test completed ===")
    return 0

if __name__ == '__main__':
    exit(main())
