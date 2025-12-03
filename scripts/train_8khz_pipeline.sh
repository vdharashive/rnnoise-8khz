#!/bin/bash
# RNNoise 8kHz Training Pipeline Script

set -e

# Configuration
ORIGINAL_DATA_DIR="."  # Current directory - adjust if your 48kHz files are elsewhere
RESAMPLED_DATA_DIR="./training_data_8khz"
FEATURES_FILE="./features_8khz.f32"
MODEL_OUTPUT_DIR="./rnnoise_8khz_model"
SEQUENCE_COUNT=10000

# Create directories
mkdir -p "$RESAMPLED_DATA_DIR"
mkdir -p "$MODEL_OUTPUT_DIR"

echo "=== RNNoise 8kHz Training Pipeline ==="
echo "Original data directory: $ORIGINAL_DATA_DIR"
echo "Resampled data directory: $RESAMPLED_DATA_DIR"
echo "Features file: $FEATURES_FILE"
echo "Model output directory: $MODEL_OUTPUT_DIR"
echo ""

# Step 1: Prepare training data (resample or use existing 8kHz files)
echo "Step 1: Preparing training data..."

# Check if 8kHz files already exist in current directory
if [ -f "./speech_8khz.pcm" ] && [ -f "./background_noise.sw" ] && [ -f "./foreground_noise.sw" ]; then
    echo "Found existing 8kHz files in current directory, using them directly..."
    RESAMPLED_DATA_DIR="."
elif [ -f "./tts_speech_48k.sw" ] || [ -f "./background_noise.sw" ] || [ -f "./foreground_noise.sw" ]; then
    echo "Found original 48kHz files, will resample them..."
    ORIGINAL_DATA_DIR="."
else
    echo "No training data found in current directory."
    echo "Please ensure you have either:"
    echo "  - 8kHz files: speech_8khz.pcm, background_noise.sw, foreground_noise.sw"
    echo "  - 48kHz files: tts_speech_48k.sw, background_noise.sw, foreground_noise.sw"
    exit 1
fi

# Resample or copy files as needed
if [ ! -f "$RESAMPLED_DATA_DIR/speech_8khz.pcm" ]; then
    if [ -f "./speech_8khz.pcm" ]; then
        echo "Using existing 8kHz speech data..."
        cp ./speech_8khz.pcm "$RESAMPLED_DATA_DIR/speech_8khz.pcm"
    elif [ -f "./tts_speech_48k.sw" ]; then
        echo "Resampling speech data from 48kHz to 8kHz..."
        python3 scripts/resample_pcm.py "./tts_speech_48k.sw" "$RESAMPLED_DATA_DIR/speech_8khz.pcm"
    else
        echo "ERROR: No speech data found (speech_8khz.pcm or tts_speech_48k.sw)"
        exit 1
    fi
else
    echo "Speech data already prepared, skipping..."
fi

if [ ! -f "$RESAMPLED_DATA_DIR/noise_8khz.pcm" ]; then
    if [ -f "./background_noise.sw" ]; then
        echo "Resampling background noise data from 48kHz to 8kHz..."
        python3 scripts/resample_pcm.py "./background_noise.sw" "$RESAMPLED_DATA_DIR/noise_8khz.pcm"
    else
        echo "ERROR: No background noise data found (background_noise.sw)"
        exit 1
    fi
else
    echo "Background noise data already prepared, skipping..."
fi

if [ ! -f "$RESAMPLED_DATA_DIR/fgnoise_8khz.pcm" ]; then
    if [ -f "./foreground_noise.sw" ]; then
        echo "Resampling foreground noise data from 48kHz to 8kHz..."
        python3 scripts/resample_pcm.py "./foreground_noise.sw" "$RESAMPLED_DATA_DIR/fgnoise_8khz.pcm"
    else
        echo "ERROR: No foreground noise data found (foreground_noise.sw)"
        exit 1
    fi
else
    echo "Foreground noise data already prepared, skipping..."
fi

echo "Data preparation completed."
echo ""

# Step 2: Build the dump_features tool
echo "Step 2: Building dump_features tool..."
if [ ! -f "src/dump_features" ]; then
    ./autogen.sh
    ./configure
    make clean
    make -j$(nproc)
else
    echo "dump_features tool already built, skipping..."
fi

echo "Build completed."
echo ""

# Step 3: Generate training features
echo "Step 3: Generating training features..."
if [ ! -f "$FEATURES_FILE" ]; then
    echo "Running dump_features with $SEQUENCE_COUNT sequences..."
    ./src/dump_features "$RESAMPLED_DATA_DIR/speech_8khz.pcm" \
                       "$RESAMPLED_DATA_DIR/noise_8khz.pcm" \
                       "$RESAMPLED_DATA_DIR/fgnoise_8khz.pcm" \
                       "$FEATURES_FILE" \
                       "$SEQUENCE_COUNT"
else
    echo "Features file already exists, skipping generation..."
fi

echo "Feature generation completed."
echo ""

# Step 4: Train the model
echo "Step 4: Training RNNoise model..."
cd torch/rnnoise

if [ ! -f "rnnoise_50.pth" ]; then
    echo "Starting training..."
    python3 train_rnnoise.py "$FEATURES_FILE" "$MODEL_OUTPUT_DIR" \
                            --epochs 50 \
                            --batch-size 64 \
                            --sequence-length 1000
else
    echo "Model already trained, skipping..."
fi

cd ../..
echo "Training completed."
echo ""

# Step 5: Export model to C code
echo "Step 5: Exporting model to C code..."
cd torch/rnnoise

if [ ! -f "rnnoise_8khz_data.c" ]; then
    echo "Exporting model weights..."
    python3 dump_rnnoise_weights.py --quantize rnnoise_50.pth rnnoise_8khz
else
    echo "Model already exported, skipping..."
fi

cd ../..
echo "Model export completed."
echo ""

# Step 6: Copy model files to src directory
echo "Step 6: Installing model files..."
cp torch/rnnoise/rnnoise_8khz/rnnoise_8khz_data.c src/
cp torch/rnnoise/rnnoise_8khz/rnnoise_8khz_data.h src/

echo "Model files installed."
echo ""

# Step 7: Rebuild RNNoise with new model
echo "Step 7: Rebuilding RNNoise with 8kHz model..."
make clean
make -j$(nproc)

echo "Rebuild completed."
echo ""

echo "=== Pipeline completed successfully! ==="
echo ""
echo "The RNNoise library is now configured for 8kHz operation."
echo "You can test it with:"
echo "  ./examples/rnnoise_demo <8khz_input.raw> <8khz_output.raw>"
echo ""
echo "Note: Input and output files should be raw 16-bit PCM at 8kHz."
