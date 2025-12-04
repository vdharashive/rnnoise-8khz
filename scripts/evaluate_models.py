#!/usr/bin/env python3
"""
Automated script to evaluate all trained RNNoise 8kHz models
For each .pth model:
1. Export model to C code
2. Recompile RNNoise with that model
3. Test the compiled binary against sample audio
4. Generate denoised output and spectrograms
5. Compare all models
"""

import os
import sys
import glob
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import soundfile as sf
import subprocess
import shutil
import tempfile

# Add torch path for rnnoise
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'torch'))

def load_audio_file(audio_path):
    """Load audio file and return as float32 numpy array"""
    try:
        # Try soundfile first (better for various formats)
        audio, sr = sf.read(audio_path)
        if sr != 8000:
            print(f"Warning: Audio sample rate is {sr}Hz, expected 8000Hz")
    except:
        # Fallback to scipy wavfile
        sr, audio = wavfile.read(audio_path)
        if sr != 8000:
            print(f"Warning: Audio sample rate is {sr}Hz, expected 8000Hz")
        # Convert to float32
        if audio.dtype != np.float32:
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

    # Ensure mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)

    return audio, sr

def save_audio_file(audio, output_path, sample_rate=8000):
    """Save audio array to file as raw 16-bit PCM"""
    # Ensure proper range for 16-bit PCM
    audio_norm = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_norm * 32767).astype(np.int16)

    # Save as raw PCM (no WAV header) - this is what rnnoise_demo expects
    with open(output_path, 'wb') as f:
        f.write(audio_int16.tobytes())

def process_audio_with_binary(binary_path, input_audio_path, output_audio_path):
    """Process audio using compiled RNNoise binary"""
    try:
        # Run rnnoise_demo
        cmd = [binary_path, input_audio_path, output_audio_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

        if result.returncode != 0:
            print(f"Error running {binary_path}: {result.stderr}")
            return False

        return True
    except subprocess.TimeoutExpired:
        print(f"Timeout processing with {binary_path}")
        return False
    except Exception as e:
        print(f"Error processing with binary {binary_path}: {e}")
        return False

def export_model_to_c(model_path, export_dir, model_name):
    """Export PyTorch model to C code using dump_rnnoise_weights.py"""
    try:
        # Create export directory
        model_export_dir = os.path.join(export_dir, model_name)
        os.makedirs(model_export_dir, exist_ok=True)

        # Run export script from the torch/rnnoise directory
        torch_rnnoise_dir = os.path.join(os.path.dirname(__file__), '..', 'torch', 'rnnoise')

        # Make model_path absolute
        abs_model_path = os.path.abspath(model_path)

        cmd = [
            sys.executable,
            'dump_rnnoise_weights.py',
            '--quantize',
            abs_model_path,
            model_export_dir
        ]

        print(f"Exporting {model_name} to C code...")
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=torch_rnnoise_dir)

        if result.returncode != 0:
            print(f"Export failed for {model_name}: {result.stderr}")
            return False, None

        # Check if files were created in model_export_dir
        expected_files = ['rnnoise_data.c', 'rnnoise_data.h']
        created_files = [f for f in expected_files if os.path.exists(os.path.join(model_export_dir, f))]

        if not created_files:
            # Files might have been created in torch_rnnoise_dir instead
            torch_files = [os.path.join(torch_rnnoise_dir, f) for f in expected_files]
            existing_torch_files = [f for f in torch_files if os.path.exists(f)]

            if existing_torch_files:
                print(f"Files created in torch directory, copying to {model_export_dir}...")
                for src_file in existing_torch_files:
                    dst_file = os.path.join(model_export_dir, os.path.basename(src_file))
                    shutil.copy2(src_file, dst_file)
                    os.remove(src_file)  # Clean up
            else:
                print(f"Export completed but no C files found in expected locations")
                return False, None

        print(f"Export completed: {len(created_files)} files created")
        return True, model_export_dir

    except Exception as e:
        print(f"Error exporting model {model_path}: {e}")
        return False, None

def compile_with_model(model_export_dir, model_name, rnnoise_root):
    """Compile RNNoise with the exported model"""
    try:
        # Copy model files to src directory
        src_dir = os.path.join(rnnoise_root, 'src')

        # Check and backup original files
        original_checksums = {}
        for filename in ['rnnoise_data.c', 'rnnoise_data.h']:
            src_file = os.path.join(src_dir, filename)
            if os.path.exists(src_file):
                # Get file size as a simple checksum
                original_checksums[filename] = os.path.getsize(src_file)
                shutil.copy2(src_file, src_file + '.backup')

        # Copy new model files
        new_checksums = {}
        for filename in ['rnnoise_data.c', 'rnnoise_data.h']:
            exported_file = os.path.join(model_export_dir, filename)
            src_file = os.path.join(src_dir, filename)
            if os.path.exists(exported_file):
                shutil.copy2(exported_file, src_file)
                new_checksums[filename] = os.path.getsize(src_file)

        # Verify files were replaced
        files_changed = False
        for filename in ['rnnoise_data.c', 'rnnoise_data.h']:
            if filename in original_checksums and filename in new_checksums:
                if original_checksums[filename] != new_checksums[filename]:
                    files_changed = True
                    break

        if not files_changed:
            print(f"Warning: Model files for {model_name} appear identical to original")

        # Force complete rebuild by deleting binary and touching source files
        print(f"Compiling RNNoise with {model_name}...")

        # Delete the binary to force complete rebuild
        binary_path = os.path.join(rnnoise_root, 'examples', 'rnnoise_demo')
        if os.path.exists(binary_path):
            os.remove(binary_path)

        # Touch the modified source files to force rebuild
        src_files_to_touch = ['src/rnnoise_data.c', 'src/rnnoise_data.h', 'src/rnn.c', 'src/rnn.h']
        for src_file in src_files_to_touch:
            full_path = os.path.join(rnnoise_root, src_file)
            if os.path.exists(full_path):
                # Update file timestamp to force rebuild
                os.utime(full_path, None)

        # Clean and rebuild
        cmd = ['make', 'clean']
        subprocess.run(cmd, cwd=rnnoise_root)

        # Remove all object files to ensure clean rebuild
        cmd = ['rm', '-f', 'src/*.o', 'src/*.lo', 'examples/*.o']
        subprocess.run(cmd, cwd=rnnoise_root)

        # Rebuild
        cmd = ['make', '-j', str(os.cpu_count())]
        result = subprocess.run(cmd, cwd=rnnoise_root, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Compilation failed for {model_name}: {result.stderr}")
            return False

        # Check if binary was created
        binary_path = os.path.join(rnnoise_root, 'examples', 'rnnoise_demo')
        if not os.path.exists(binary_path):
            print(f"Binary not found after compilation: {binary_path}")
            return False

        print(f"Successfully compiled {model_name}")
        return True, binary_path

    except Exception as e:
        print(f"Error compiling with model {model_name}: {e}")
        return False, None

def restore_original_model(rnnoise_root):
    """Restore original model files"""
    try:
        src_dir = os.path.join(rnnoise_root, 'src')
        for filename in ['rnnoise_data.c', 'rnnoise_data.h']:
            backup_file = os.path.join(src_dir, filename + '.backup')
            src_file = os.path.join(src_dir, filename)
            if os.path.exists(backup_file):
                shutil.copy2(backup_file, src_file)
                os.remove(backup_file)
        return True
    except Exception as e:
        print(f"Error restoring original model: {e}")
        return False

def generate_spectrogram(audio, sample_rate, output_path, title=""):
    """Generate and save spectrogram"""
    plt.figure(figsize=(12, 8))

    # Generate spectrogram
    f, t, Sxx = signal.spectrogram(audio, fs=sample_rate, nperseg=256, noverlap=128)

    # Plot
    plt.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(f'Spectrogram - {title}')
    plt.colorbar(label='Power [dB]')
    plt.ylim(0, sample_rate/2)  # Show up to Nyquist frequency

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

def load_raw_pcm(file_path, sample_rate=8000):
    """Load raw 16-bit PCM file"""
    with open(file_path, 'rb') as f:
        data = f.read()

    # Convert to numpy array
    audio_int16 = np.frombuffer(data, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0

    return audio_float

def calculate_audio_metrics(original, processed):
    """Calculate basic audio quality metrics"""
    if processed is None or len(processed) == 0:
        return None

    # Use the minimum length to compare
    min_len = min(len(original), len(processed))
    orig_compare = original[:min_len]
    proc_compare = processed[:min_len]

    # RMS levels
    original_rms = np.sqrt(np.mean(orig_compare**2))
    processed_rms = np.sqrt(np.mean(proc_compare**2))

    # Peak levels
    original_peak = np.max(np.abs(orig_compare))
    processed_peak = np.max(np.abs(proc_compare))

    # SNR improvement (rough estimate)
    # Assume processed is the "clean" version and calculate noise as difference
    noise = orig_compare - proc_compare
    snr_improvement = 10 * np.log10(
        (np.mean(orig_compare**2) + 1e-10) / (np.mean(noise**2) + 1e-10)
    )

    return {
        'original_rms': original_rms,
        'processed_rms': processed_rms,
        'original_peak': original_peak,
        'processed_peak': processed_peak,
        'snr_improvement': snr_improvement,
        'length_diff': len(original) - len(processed)
    }

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained RNNoise 8kHz models by compiling and testing each one')
    parser.add_argument('--input-audio', required=True, help='Path to input audio file (8kHz, 16-bit PCM)')
    parser.add_argument('--models-dir', default='../rnnoise_8khz_model/checkpoints',
                       help='Directory containing .pth model files')
    parser.add_argument('--output-dir', default='./model_evaluation',
                       help='Output directory for results')
    parser.add_argument('--max-models', type=int, default=None,
                       help='Maximum number of models to test (for testing)')

    args = parser.parse_args()

    # Get RNNoise root directory
    rnnoise_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    # Create output directories
    args.output_dir = os.path.abspath(args.output_dir)
    audio_output_dir = os.path.join(args.output_dir, 'denoised_audio')
    spectrogram_dir = os.path.join(args.output_dir, 'spectrograms')
    export_dir = os.path.join(args.output_dir, 'model_exports')
    os.makedirs(audio_output_dir, exist_ok=True)
    os.makedirs(spectrogram_dir, exist_ok=True)
    os.makedirs(export_dir, exist_ok=True)

    print("=== RNNoise 8kHz Model Evaluation ===")
    print(f"Input audio: {args.input_audio}")
    print(f"Models directory: {args.models_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"RNNoise root: {rnnoise_root}")
    print()

    # Load and prepare input audio
    print("Loading input audio...")
    try:
        input_audio, sample_rate = load_audio_file(args.input_audio)
        print(f"Loaded {len(input_audio)} samples at {sample_rate}Hz")

        # Save as raw PCM for rnnoise_demo
        input_raw_path = os.path.join(args.output_dir, 'input.raw')
        save_audio_file(input_audio, input_raw_path, sample_rate)
        print(f"Saved as raw PCM: {input_raw_path}")

    except Exception as e:
        print(f"Error loading audio: {e}")
        return 1

    # Generate input spectrogram
    print("Generating input spectrogram...")
    input_spec_path = os.path.join(spectrogram_dir, 'input_spectrogram.png')
    generate_spectrogram(input_audio, sample_rate, input_spec_path, "Input Audio")

    # Find all model files
    model_pattern = os.path.join(args.models_dir, 'rnnoise_*.pth')
    model_files = sorted(glob.glob(model_pattern))

    if not model_files:
        # Try with full path from rnnoise root
        full_models_dir = os.path.join(rnnoise_root, args.models_dir)
        model_pattern = os.path.join(full_models_dir, 'rnnoise_*.pth')
        model_files = sorted(glob.glob(model_pattern))

    # Convert all paths to absolute paths
    model_files = [os.path.abspath(f) for f in model_files]

    if not model_files:
        print(f"No model files found in {args.models_dir}")
        return 1

    # Limit number of models if specified
    if args.max_models:
        model_files = model_files[:args.max_models]

    print(f"Found {len(model_files)} model files to evaluate")
    print()

    # Process each model
    results = []
    successful_models = 0

    for i, model_path in enumerate(model_files, 1):
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        print(f"[{i}/{len(model_files)}] Processing model: {model_name}")

        # Export model to C code
        export_success, model_export_dir = export_model_to_c(model_path, export_dir, model_name)
        if not export_success:
            print(f"  ✗ Export failed for {model_name}")
            results.append({
                'model': model_name,
                'status': 'export_failed',
                'audio_path': None,
                'spectrogram_path': None,
                'metrics': None
            })
            continue

        # Compile with this model
        compile_success, binary_path = compile_with_model(model_export_dir, model_name, rnnoise_root)
        if not compile_success:
            print(f"  ✗ Compilation failed for {model_name}")
            results.append({
                'model': model_name,
                'status': 'compile_failed',
                'audio_path': None,
                'spectrogram_path': None,
                'metrics': None
            })
            continue

        # Process audio with compiled binary
        output_raw_path = os.path.join(args.output_dir, f'{model_name}_output.raw')
        process_success = process_audio_with_binary(binary_path, input_raw_path, output_raw_path)

        if not process_success:
            print(f"  ✗ Processing failed for {model_name}")
            results.append({
                'model': model_name,
                'status': 'process_failed',
                'audio_path': None,
                'spectrogram_path': None,
                'metrics': None
            })
            continue

        # Load processed audio
        try:
            output_audio = load_raw_pcm(output_raw_path, sample_rate)
        except Exception as e:
            print(f"  ✗ Failed to load processed audio: {e}")
            results.append({
                'model': model_name,
                'status': 'load_failed',
                'audio_path': None,
                'spectrogram_path': None,
                'metrics': None
            })
            continue

        # Save as WAV for easier listening
        audio_output_path = os.path.join(audio_output_dir, f'{model_name}_denoised.wav')
        sf.write(audio_output_path, output_audio, sample_rate)

        # Generate spectrogram
        spec_output_path = os.path.join(spectrogram_dir, f'{model_name}_spectrogram.png')
        generate_spectrogram(output_audio, sample_rate, spec_output_path, f"{model_name} Denoised")

        # Calculate metrics
        metrics = calculate_audio_metrics(input_audio, output_audio)

        results.append({
            'model': model_name,
            'status': 'success',
            'audio_path': audio_output_path,
            'spectrogram_path': spec_output_path,
            'metrics': metrics
        })

        successful_models += 1
        print(f"  ✓ Successfully processed {model_name}")

    # Restore original model
    print("\nRestoring original model...")
    restore_original_model(rnnoise_root)

    # Generate summary report
    print()
    print("=== Evaluation Summary ===")
    print(f"Total models tested: {len(model_files)}")
    print(f"Successful: {successful_models}")
    print(f"Failed: {len(model_files) - successful_models}")
    print()

    # Save detailed results
    results_file = os.path.join(args.output_dir, 'evaluation_results.txt')
    with open(results_file, 'w') as f:
        f.write("RNNoise 8kHz Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Input audio: {args.input_audio}\n")
        f.write(f"Sample rate: {sample_rate}Hz\n")
        f.write(f"Duration: {len(input_audio)/sample_rate:.2f} seconds\n\n")

        f.write("Model Results:\n")
        f.write("-" * 50 + "\n")

        for result in results:
            f.write(f"\nModel: {result['model']}\n")
            f.write(f"  Status: {result['status']}\n")
            if result['status'] == 'success':
                if result['audio_path']:
                    f.write(f"  Audio: {result['audio_path']}\n")
                if result['spectrogram_path']:
                    f.write(f"  Spectrogram: {result['spectrogram_path']}\n")
                if result['metrics']:
                    f.write(f"  SNR Improvement: {result['metrics']['snr_improvement']:.2f} dB\n")
                    f.write(f"  RMS Ratio: {result['metrics']['processed_rms']/result['metrics']['original_rms']:.3f}\n")
            f.write("\n")

    print(f"Results saved to: {results_file}")
    print(f"Denoised audio saved to: {audio_output_dir}")
    print(f"Spectrograms saved to: {spectrogram_dir}")
    print(f"Model exports saved to: {export_dir}")

    # Find best model based on SNR improvement
    valid_results = [r for r in results if r['status'] == 'success' and r['metrics'] is not None]
    if valid_results:
        best_model = max(valid_results, key=lambda x: x['metrics']['snr_improvement'])
        print()
        print("🎯 BEST MODEL:")
        print(f"  Model: {best_model['model']}")
        print(f"  SNR Improvement: {best_model['metrics']['snr_improvement']:.2f} dB")
        print(f"  Audio: {best_model['audio_path']}")
        print(f"  Spectrogram: {best_model['spectrogram_path']}")

    print()
    print("=== Evaluation Complete ===")
    print(f"Check {args.output_dir} for all results!")

    return 0

if __name__ == '__main__':
    sys.exit(main())
