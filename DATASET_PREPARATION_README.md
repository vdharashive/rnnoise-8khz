# RNNoise 8kHz Dataset Preparation & Training Pipeline

This comprehensive script handles the complete process of preparing the Xiph.Org RNNoise training dataset and training an optimized 8kHz model.

## 🚀 Quick Start

```bash
# Run the complete pipeline
./scripts/prepare_dataset_and_train.sh

# Or run individual steps
./scripts/prepare_dataset_and_train.sh download  # Download data only
./scripts/prepare_dataset_and_train.sh resample  # Resample to 8kHz only
./scripts/prepare_dataset_and_train.sh train     # Train model only
```

## 📋 Prerequisites

### System Requirements
- **Disk Space**: 150GB+ (downloads + processing)
- **RAM**: 8GB+ recommended
- **CPU**: Multi-core recommended
- **Network**: Fast internet for downloads

### Dependencies
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install wget python3 python3-pip gcc make autoconf automake libtool

# Install Python packages
pip3 install --user numpy scipy torch torchvision torchaudio matplotlib soundfile
```

## 📊 Pipeline Overview

### Phase 1: Data Acquisition (~2-4 hours)
```
Xiph.Org Dataset (66GB)
    ↓ Download
48kHz Raw Files
    ↓ Resample
8kHz Processed Files (~15GB)
```

### Phase 2: Model Training (~24-48 hours)
```
8kHz Training Data
    ↓ Feature Extraction
Training Features (.f32 file)
    ↓ Neural Network Training
PyTorch Models (.pth files)
    ↓ Export to C Code
Optimized 8kHz Library
```

## 📁 Directory Structure

After running the pipeline:

```
workspace/
├── rnnoise_training_data/
│   ├── 48khz/           # Original downloaded files
│   │   ├── tts_speech_48k.sw (63GB)
│   │   ├── background_noise_v2.sw (1.2GB)
│   │   └── ...
│   └── 8khz/            # Resampled 8kHz files
│       ├── speech_8khz.pcm (~10GB)
│       ├── background_noise_8khz.pcm
│       └── ...
├── training_output/
│   ├── features_8khz_full.f32  # Training features
│   ├── models/                  # PyTorch checkpoints
│   └── TRAINING_SUMMARY.txt     # Results summary
├── torch/rnnoise/
│   ├── rnnoise_100.pth          # Final trained model
│   └── rnnoise_8khz_full/       # Exported C code
└── examples/
    └── rnnoise_demo             # Optimized 8kHz binary
```

## ⚙️ Configuration

### Environment Variables
```bash
export DATA_DIR="/path/to/large/disk/rnnoise_data"  # Where to store training data
export WORKSPACE_DIR="/path/to/rnnoise/repo"       # Repository location
```

### Training Parameters
Edit the script to modify:
```bash
SEQUENCE_COUNT=100000  # Training sequences (higher = better but slower)
EPOCHS=100            # Training epochs (higher = better convergence)
```

## 🏃‍♂️ Usage Examples

### Complete Pipeline (Recommended)
```bash
./scripts/prepare_dataset_and_train.sh
```

### Step-by-Step Execution
```bash
# 1. Check dependencies
./scripts/prepare_dataset_and_train.sh check-deps

# 2. Download data (66GB - takes several hours)
./scripts/prepare_dataset_and_train.sh download

# 3. Resample to 8kHz (takes significant CPU time)
./scripts/prepare_dataset_and_train.sh resample

# 4. Build RNNoise
./scripts/prepare_dataset_and_train.sh build

# 5. Generate training features
./scripts/prepare_dataset_and_train.sh features

# 6. Train the model (24-48 hours)
./scripts/prepare_dataset_and_train.sh train

# 7. Export and install model
./scripts/prepare_dataset_and_train.sh export

# 8. Final verification
./scripts/prepare_dataset_and_train.sh verify
```

### Resume After Interruption
```bash
# The script is designed to resume where it left off
# Just run the complete pipeline again
./scripts/prepare_dataset_and_train.sh
```

## 📊 Expected Results

### Training Data Statistics
- **Original**: 66GB Xiph.Org dataset
- **Resampled**: ~15GB 8kHz data
- **Features**: ~20-30GB .f32 file
- **Training time**: 24-48 hours

### Performance Improvements
- **Previous training**: -3 dB SNR (limited data)
- **This training**: Expected +8 to +15 dB SNR
- **Quality**: Significantly better noise suppression

## 🔧 Troubleshooting

### Disk Space Issues
```bash
# Monitor disk usage
df -h

# Clean up intermediate files if needed
rm -rf rnnoise_training_data/48khz/*.sw  # Remove original 48kHz files after resampling
```

### Memory Issues
```bash
# Reduce chunk size for resampling
sed -i 's/--chunk-size 100/--chunk-size 50/' scripts/prepare_dataset_and_train.sh
```

### Training Issues
```bash
# Reduce batch size if GPU memory issues
sed -i 's/--batch-size 128/--batch-size 64/' scripts/prepare_dataset_and_train.sh

# Reduce sequence length
sed -i 's/--sequence-length 2000/--sequence-length 1000/' scripts/prepare_dataset_and_train.sh
```

### Network Issues
```bash
# Resume downloads
./scripts/prepare_dataset_and_train.sh download

# Manual download if wget fails
wget https://media.xiph.org/rnnoise/data/tts_speech_48k.sw
```

## 🎯 Final Output

After successful completion:

### 1. Optimized Binary
```bash
./examples/rnnoise_demo input_8khz.raw output_8khz.raw
```

### 2. Performance Metrics
- SNR improvement: +8 to +15 dB
- Low latency: 10ms frames
- Telephony optimized: 8kHz, mono, 16-bit

### 3. Integration Ready
- C code: `src/rnnoise_8khz_full_data.*`
- Library: `librnnoise.a`
- Headers: `include/rnnoise.h`

## 📞 Next Steps

1. **Deploy**: Use `rnnoise_demo` in your telephony pipeline
2. **Integrate**: Link `librnnoise` into your application
3. **Tune**: Adjust VAD thresholds for your specific use case
4. **Monitor**: Track real-world performance metrics

## 📚 References

- [Xiph.Org RNNoise Data](https://media.xiph.org/rnnoise/data/)
- [RNNoise Paper](https://arxiv.org/pdf/1709.08243.pdf)
- [Original Implementation](https://gitlab.xiph.org/xiph/rnnoise)

---

**This pipeline transforms the academic RNNoise into a production-ready 8kHz telephony noise suppressor with comprehensive training on the full Xiph.Org dataset.**
