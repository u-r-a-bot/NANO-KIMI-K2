# Efficient Transformer Language Model with Multi-Head Latent Attention

A PyTorch implementation of an efficient transformer language model featuring Multi-Head Latent Attention (MLA) with KV compression and Mixture of Experts (MoE) architecture, optimized with the MuonClip optimizer for stable and efficient training.

## 🎯 Overview

This project implements a compact yet powerful language model architecture that combines:

- **Multi-Head Latent Attention (MLA)**: Memory-efficient attention mechanism using KV compression with LoRA-style techniques, significantly reducing memory requirements for the key-value cache
- **Mixture of Experts (MoE)**: Sparse expert activation (2 of 8 experts per token) for efficient scaling with shared experts for common patterns
- **MuonClip Optimizer**: Advanced optimizer combining Muon optimizer (Newton-Schulz orthogonalization) with QK-Clip for stable training
- **Flexible Data Pipeline**: Support for multiple dataset formats including local files, HuggingFace streaming datasets, and pre-tokenized binaries

## 📊 Model Specifications

- **Parameters**: ~243M (243,263,104)
- **Architecture**: 6 layers (1 dense + 5 MoE)
- **Embedding Dimension**: 896
- **Attention Heads**: 8
- **Vocabulary Size**: 49,152 (SmolLM tokenizer)
- **Max Sequence Length**: 1024
- **MoE Configuration**: 
  - 8 routed experts, 2 activated per token
  - 2 shared experts
  - Expert groups: 1

## 📦 Pre-trained Checkpoints

Due to the large file size, pre-trained model checkpoints are hosted on Google Drive:

**[Download Pre-trained Checkpoint from Google Drive](https://drive.google.com/drive/folders/YOUR_FOLDER_ID)**

> 📝 **Note**: Place the downloaded checkpoint file (`.pt` format) in the `checkpoints/` directory before running inference.

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended) or Apple Silicon (MPS)

### Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd NANO-KIMI-K2-main
   ```

2. **Create and activate virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running Locally

#### Interactive Chatbot Mode

Run the model in interactive chatbot mode:

```bash
python inference.py checkpoints/latest.pt --interactive
```

For streaming generation (tokens appear as they're generated):

```bash
python inference.py checkpoints/latest.pt --interactive --stream
```

#### Single Prompt Generation

Generate text from a prompt:

```bash
python inference.py checkpoints/latest.pt \
    --prompt "Once upon a time" \
    --max_tokens 100 \
    --temperature 0.8 \
    --top_p 0.9
```

With streaming:

```bash
python inference.py checkpoints/latest.pt \
    --prompt "The future of artificial intelligence is" \
    --stream \
    --max_tokens 200
```

## 🏋️ Training

### Basic Training Setup

Train on a local text file:

```bash
python train.py \
    --dataset local \
    --data_path data/train.txt \
    --batch_size 2 \
    --learning_rate 0.02 \
    --max_steps 1000 \
    --save_interval 500
```

### Training with HuggingFace Dataset

Train using streaming data from HuggingFace Hub:

```bash
python train.py \
    --dataset hfstreaming \
    --hf_repo_name HuggingFaceTB/cosmopedia-100k \
    --batch_size 2 \
    --learning_rate 0.02 \
    --max_steps 5000 \
    --gradient_accumulation 4 \
    --eval_interval 100 \
    --save_interval 500
```

### Training with FineWeb Dataset

```bash
python train.py \
    --dataset fineweb \
    --num_samples 10000 \
    --batch_size 2 \
    --max_steps 2000 \
    --save_interval 500
```

### Key Training Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset` | Dataset type: `local`, `hfstreaming`, `fineweb`, `localstreaming`, `tokenbin` | `local` |
| `--data_path` | Path to local data file | `data/train.txt` |
| `--hf_repo_name` | HuggingFace dataset name (for `hfstreaming`) | `None` |
| `--batch_size` | Batch size | `2` |
| `--learning_rate` | Learning rate | `0.02` |
| `--momentum` | Momentum (Muon optimizer) | `0.95` |
| `--weight_decay` | Weight decay | `0.01` |
| `--qk_clip_tau` | QK-Clip threshold | `100.0` |
| `--max_steps` | Maximum training steps | `1000` |
| `--eval_interval` | Validation interval | `25` |
| `--save_interval` | Checkpoint save interval | `500` |
| `--warmup_steps` | Learning rate warmup steps | `100` |
| `--gradient_accumulation` | Gradient accumulation steps | `2` |
| `--device` | Device: `cuda`, `mps`, or `cpu` | Auto-detect |
| `--mixed_precision` | Enable mixed precision (CUDA) | Auto-enabled for CUDA |
| `--resume` | Resume from checkpoint | `False` |
| `--checkpoint_path` | Path to checkpoint file | `None` |

## 🔧 Inference Options

| Argument | Description | Default |
|----------|-------------|---------|
| `checkpoint_path` | Path to model checkpoint | Required |
| `--prompt` | Text prompt for generation | `None` |
| `--max_tokens` | Maximum new tokens to generate | `100` |
| `--temperature` | Sampling temperature (higher = more random) | `1.0` |
| `--top_p` | Top-p (nucleus) sampling | `0.9` |
| `--device` | Device: `cuda`, `mps`, or `cpu` | Auto-detect |
| `--interactive` | Interactive chatbot mode | `False` |
| `--stream` | Stream tokens as they generate | `False` |

## 📁 Project Structure

```
.
├── config.py              # Model configuration and hyperparameters
├── models.py              # Core model architecture (Transformer, MLA, MoE)
├── optimizer.py           # MuonClip optimizer implementation
├── dataset.py             # Dataset loaders (local, streaming, HF)
├── finewebdataset.py      # FineWeb dataset integration
├── train.py               # Main training script
├── inference.py           # Inference and chatbot script
├── base.ipynb            # Development and experimentation notebook
├── verify_muon.py        # Optimizer verification utilities
├── requirements.txt       # Python dependencies
└── checkpoints/          # Model checkpoints directory
```

## ⚙️ Configuration

Edit `config.py` to customize model architecture:

```python
@dataclass
class Config:
    tokenizer_name: str = "HuggingFaceTB/SmolLM-135M"
    dim: int = 896                    # Embedding dimension
    n_heads: int = 8                  # Number of attention heads
    n_layers: int = 6                 # Total transformer layers
    max_seq_length: int = 1024        # Maximum sequence length
    n_routed_experts: int = 8         # MoE: total number of experts
    n_activated_experts: int = 2      # MoE: experts activated per token
    n_dense_layers: int = 1           # Dense layers before MoE layers
    kv_lora_rank: int = 256           # KV compression rank
    # ... more configuration options
```

## 💡 Key Features

### Multi-Head Latent Attention (MLA)

MLA reduces memory consumption by compressing key-value representations:
- K/V compression using low-rank adaptation (LoRA-style)
- Compressed KV cache reduces memory footprint
- Maintains model performance with efficient representations
- Supports both positional (RoPE) and non-positional attention components

### Mixture of Experts (MoE)

Efficient scaling through sparse expert activation:
- 8 total experts, 2 activated per token
- Shared experts handle common patterns
- Group-based routing for efficient computation
- Configurable expert groups and routing functions

### MuonClip Optimizer

Advanced training optimization:
- Muon optimizer with Newton-Schulz orthogonalization
- QK-Clip for attention logit stabilization
- Separate parameter groups for Q/K weights
- Momentum-based updates with Nesterov acceleration

## 📚 Dataset Formats

### 1. Local Text File (`--dataset local`)
- Plain text files
- Supports caching with memory mapping
- Example: `data/train.txt`

### 2. HuggingFace Streaming (`--dataset hfstreaming`)
- Streams directly from HuggingFace Hub
- Auto-detects dataset configurations
- No local storage required
- Example: `--hf_repo_name HuggingFaceTB/cosmopedia-100k`

### 3. FineWeb (`--dataset fineweb`)
- Pre-processed FineWeb dataset
- Standard and streaming modes
- High-quality web data

### 4. Token Bin (`--dataset tokenbin`)
- Pre-tokenized binary format (uint16)
- Fastest loading for large datasets
- Example: `--data_path tokens/train.bin`

### 5. Local Streaming (`--dataset localstreaming`)
- Streams from local files
- Memory-efficient for large files
- Worker-based parallel processing

## 🎓 Training Tips

### Memory Optimization
- Use gradient accumulation: `--gradient_accumulation 4`
- Enable mixed precision: `--mixed_precision` (auto for CUDA)
- Reduce batch size if OOM errors occur
- Use streaming datasets for large data

### Performance Optimization
- Use `torch.compile()` on CUDA (auto-enabled for compute capability >= 7.0)
- Increase `num_workers` for data loading (non-streaming datasets)
- Use pre-tokenized binary format for fastest loading
- Adjust `max_seq_length` based on your GPU memory

### Best Practices
- Start with smaller `max_steps` to verify setup
- Monitor training metrics in `logs/training_metrics.png`
- Use validation intervals to track progress
- Save checkpoints regularly with `--save_interval`

## 📈 Output Files

Training generates:

- `checkpoints/latest.pt` - Latest checkpoint
- `checkpoints/best.pt` - Best validation checkpoint  
- `checkpoints/final.pt` - Final training checkpoint
- `logs/training_metrics.png` - Training plots (loss, LR, grad norm)
- `logs/training_metrics.json` - Detailed training metrics

## 🔍 Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Increase `--gradient_accumulation`
- Use smaller `max_seq_length` in config
- Enable gradient checkpointing (if implemented)

### Slow Training
- Enable mixed precision (auto for CUDA)
- Use streaming datasets for large data
- Set `num_workers=0` for streaming datasets
- Check GPU utilization with `nvidia-smi`

### Checkpoint Loading
- Checkpoint format is handled automatically
- Supports both compiled and non-compiled models
- Use `--resume` flag with `--checkpoint_path` to continue training

## 📄 License

Please refer to the repository license file for license information.

## 🙏 Acknowledgments

- Inspired by the KIMI K2 architecture
- Muon optimizer with QK-Clip techniques
- Multi-Head Latent Attention (MLA) implementation
- Mixture of Experts (MoE) architecture

## 📞 Contact & Contributions

For issues, questions, or contributions, please open an issue or submit a pull request.

---

**Happy Training! 🚀**
