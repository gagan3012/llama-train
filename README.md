# Train LLama on Arabic

### Installation

We expect you have CUDA 11.8 installed.

On tpus run

```bash
./scripts/setup_tpu.sh
```

#### Install Pytorch Nightly

```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
```

#### Build XFormers from Source

Note: as of 2023/09/02, xformers does not provide pre-built binaries for torch 2.1. You have to build it from source.

```bash
pip uninstall ninja -y && install ninja -U
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```

#### Install Flash-Attention 2 and other fused operators

```bash
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
```

#### Install Remaining Dependencies

```
pip install -r requirements.txt tokenizers sentencepiece
```

to install other dependencies.
It may take >= 5 minutes to build xformers/flash-attention. Do not worry if the process seemly stagnant or the terminal print out many warnings.

Then you are ready to go 🎉!

### Data Preparation

#### Tokenize data

Use the provided scripts to tokenize the datasets and divide them into chunks.

```bash
python scripts/prepare_ar.py --source_path /path/to/data/ --tokenizer_path data/llama --destination_path data/ar_processed_train --split train --percentage 1.0
python scripts/prepare_ar.py --source_path /path/to/data/ --tokenizer_path data/llama --destination_path data/ar_processed --split val --percentage 1.0
```

### Pretraining

#### Train LLama

```bash
lightning run model \
    --node-rank=0  \
    --main-address=172.16.101.5 \
    --accelerator=tpu \
    pretrain/arllama.py --devices 8 --train_data_dir data/ar_processed_train  --val_data_dir data/ar_processed_val
```
