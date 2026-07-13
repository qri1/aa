$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    throw 'NVIDIA driver is not available. Install or update it, then run this script again.'
}

& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
if ($LASTEXITCODE -ne 0) {
    throw 'nvidia-smi failed. The NVIDIA driver must work before training.'
}

& py -3.12 -m venv .venv
$python = Join-Path $root '.venv\Scripts\python.exe'

& $python -m pip install --upgrade pip
& $python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& $python -m pip install transformers==4.57.1 trl==0.23.1 peft==0.17.1 accelerate==1.10.1 "bitsandbytes>=0.48.0" datasets==4.0.0 scikit-learn==1.7.2 jupyterlab

& $python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot see CUDA'; print({'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'vram_gb': round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2)})"

$env:AA_LOCAL_8GB = '1'
& $python -m jupyter lab notebooks/main.ipynb
