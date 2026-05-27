# GPU Driver Recovery Status After Reboot

This log records the post-reboot state after trying Ubuntu open/server NVIDIA drivers and rebooting.

## date_utc

```text
Mon May 25 05:20:57 AM UTC 2026
```

## uname

```text
Linux smartcar 6.17.0-23-generic #23~24.04.1-Ubuntu SMP PREEMPT_DYNAMIC Tue Apr 14 16:11:48 UTC 2 x86_64 x86_64 x86_64 GNU/Linux
```

## lspci_nvidia

```text
04:00.0 3D controller [0302]: NVIDIA Corporation GA100 [A100 SXM4 40GB] [10de:20b0] (rev a1)
05:00.0 3D controller [0302]: NVIDIA Corporation GA100 [A100 SXM4 40GB] [10de:20b0] (rev a1)
```

## dev_nodes

```text
crw-rw-rw- 1 root root 195, 255 May 25 05:10 /dev/nvidiactl
```

## lsmod

```text

```

## nvidia_smi

```text
NVIDIA-SMI has failed because it couldn't communicate with the NVIDIA driver. Make sure that the latest NVIDIA driver is installed and running.
```

## modinfo_nvidia

```text
filename:       /lib/modules/6.17.0-23-generic/updates/dkms/nvidia.ko.zst
description:    NVIDIA core GPU kernel module
version:        535.309.01
license:        NVIDIA
srcversion:     FD5B554D7D0C3E23A9E7317
```

## dkms

```text
nvidia-srv/535.309.01, 6.17.0-23-generic, x86_64: installed
```

## nvidia_packages

```text
ii  libnvidia-compute-535-server:amd64                535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA libcompute package
rc  libnvidia-compute-595:amd64                       595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA libcompute package
rc  libnvidia-compute-595-server:amd64                595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA libcompute package
ii  nvidia-compute-utils-535-server                   535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA compute utilities
rc  nvidia-compute-utils-595                          595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA compute utilities
rc  nvidia-compute-utils-595-server                   595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA compute utilities
ii  nvidia-dkms-535-server                            535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA DKMS package
rc  nvidia-dkms-595-open                              595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA DKMS package (open kernel module)
rc  nvidia-dkms-595-server                            595.71.05-0ubuntu0.24.04.1               amd64        NVIDIA DKMS package
ii  nvidia-driver-535-server                          535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA Server Driver metapackage
ii  nvidia-kernel-common-535-server                   535.309.01-0ubuntu0.24.04.1              amd64        Shared files used with the kernel module
rc  nvidia-kernel-common-595                          595.71.05-0ubuntu0.24.04.1               amd64        Shared files used with the kernel module
rc  nvidia-kernel-common-595-server                   595.71.05-0ubuntu0.24.04.1               amd64        Shared files used with the kernel module
ii  nvidia-kernel-source-535-server                   535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA kernel source package
rc  nvidia-linux-grid-535                             535.247.01                               amd64        NVIDIA GRID driver - version 535.247.01
ii  nvidia-utils-535-server                           535.309.01-0ubuntu0.24.04.1              amd64        NVIDIA Server Driver support binaries
```

## grid_package_policy

```text
nvidia-linux-grid-535:
  Installed: (none)
  Candidate: (none)
  Version table:
     535.247.01 -1
        100 /var/lib/dpkg/status
```

## grid_deb_search

```text

```

## dmesg_nvidia_tail

```text
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:57 2026] nvidia 0000:05:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:57 2026] NVRM: The NVIDIA probe routine failed for 2 device(s).
[Mon May 25 05:20:57 2026] NVRM: None of the NVIDIA devices were initialized.
[Mon May 25 05:20:57 2026] nvidia-nvlink: Unregistered Nvlink Core, major device number 238
[Mon May 25 05:20:57 2026] nvidia-nvlink: Nvlink Core is being initialized, major device number 238
[Mon May 25 05:20:57 2026] NVRM: The NVIDIA GPU 0000:04:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:57 2026] nvidia 0000:04:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:57 2026] NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:57 2026] nvidia 0000:05:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:57 2026] NVRM: The NVIDIA probe routine failed for 2 device(s).
[Mon May 25 05:20:57 2026] NVRM: None of the NVIDIA devices were initialized.
[Mon May 25 05:20:57 2026] nvidia-nvlink: Unregistered Nvlink Core, major device number 238
[Mon May 25 05:20:58 2026] nvidia-nvlink: Nvlink Core is being initialized, major device number 238
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA GPU 0000:04:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:58 2026] nvidia 0000:04:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:58 2026] nvidia 0000:05:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA probe routine failed for 2 device(s).
[Mon May 25 05:20:58 2026] NVRM: None of the NVIDIA devices were initialized.
[Mon May 25 05:20:58 2026] nvidia-nvlink: Unregistered Nvlink Core, major device number 238
[Mon May 25 05:20:58 2026] nvidia-nvlink: Nvlink Core is being initialized, major device number 238
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA GPU 0000:04:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:58 2026] nvidia 0000:04:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:58 2026] nvidia 0000:05:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:58 2026] NVRM: The NVIDIA probe routine failed for 2 device(s).
[Mon May 25 05:20:58 2026] NVRM: None of the NVIDIA devices were initialized.
[Mon May 25 05:20:58 2026] nvidia-nvlink: Unregistered Nvlink Core, major device number 238
[Mon May 25 05:20:59 2026] nvidia-nvlink: Nvlink Core is being initialized, major device number 238
[Mon May 25 05:20:59 2026] NVRM: The NVIDIA GPU 0000:04:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:59 2026] nvidia 0000:04:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:59 2026] NVRM: The NVIDIA GPU 0000:05:00.0 (PCI ID: 10de:20b0)
                           NVRM: installed in this system is not supported by the
                           NVRM: NVIDIA 535.309.01 driver release.
                           NVRM: Please see 'Appendix A - Supported NVIDIA GPU Products'
                           NVRM: in this release's README, available on the operating system
                           NVRM: specific graphics driver download page at www.nvidia.com.
[Mon May 25 05:20:59 2026] nvidia 0000:05:00.0: probe with driver nvidia failed with error -1
[Mon May 25 05:20:59 2026] NVRM: The NVIDIA probe routine failed for 2 device(s).
[Mon May 25 05:20:59 2026] NVRM: None of the NVIDIA devices were initialized.
[Mon May 25 05:20:59 2026] nvidia-nvlink: Unregistered Nvlink Core, major device number 238
```

## Diagnosis

The machine exposes two NVIDIA GA100/A100 devices at PCI IDs 10de:20b0, but the currently installed Ubuntu 535-server driver rejects them during probe.
The previous nvidia-linux-grid-535 package is no longer installed and no local .deb cache was found in the searched paths. The current apt metadata has no install candidate for nvidia-linux-grid-535.
CUDA remains blocked until a matching NVIDIA GRID/vGPU guest driver package is restored, or the platform provider exposes these GPUs in a mode supported by a standard NVIDIA data-center driver.
