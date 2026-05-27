# CUDA Hardware Blocked

CUDA feasibility microbench was not run on this host because the A100 hardware is visible on PCI, but the NVIDIA kernel driver is not active for the running kernel.

## Observed State

- Host: `smartcar`
- Running kernel: `6.17.0-23-generic`
- PCI GPUs:
  - `04:00.0 3D controller: NVIDIA Corporation GA100 [A100 SXM4 40GB]`
  - `05:00.0 3D controller: NVIDIA Corporation GA100 [A100 SXM4 40GB]`
- `/dev/nvidia*`: missing
- `/proc/driver/nvidia`: missing
- `nvidia-smi`: cannot communicate with NVIDIA driver
- loaded GPU driver: `nouveau`
- `modinfo nvidia`: module not found for current kernel
- DKMS NVIDIA modules installed only for older kernels:
  - `6.14.0-36-generic`
  - `6.8.0-111-generic`
  - `6.8.0-88-generic`
- `nvcc`: unavailable in PATH
- conda env `falcom` has CUDA PyTorch (`torch 2.9.1+cu128`), but `torch.cuda.is_available()` is `False` because the driver/device nodes are unavailable.

## Diagnosis

The machine has two physical A100 GPUs, but the current userspace cannot use them. This is a driver/kernel mismatch or driver activation issue, not a compressor code issue.

## Likely Fix Options

1. Reboot into a kernel that already has the NVIDIA DKMS module built, e.g. `6.14.0-36-generic` or `6.8.0-111-generic`.
2. Or install/build the NVIDIA driver module for the current `6.17.0-23-generic` kernel.
3. Ensure `nouveau` is blacklisted and the proprietary NVIDIA module loads.
4. After fixing, verify:
   - `/dev/nvidia0` and `/dev/nvidia1` exist
   - `nvidia-smi` lists both A100 GPUs
   - `/home/exouser/miniconda3/envs/falcom/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"` prints `True 2`

No GPU performance claim is made until the driver is active.


## Post-reboot driver recovery update

Detailed post-reboot diagnosis: `logs/gpu_driver_recovery_20260525_052057/status_after_reboot.md`

Current conclusion: hardware is visible on PCIe, but Ubuntu 535-server rejects PCI ID `10de:20b0`; matching GRID/vGPU guest driver package is required.
