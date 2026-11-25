# Node specs: gpu-grad-01.cs.mcgill.ca

2026-11-25 17:02:25 EST; job 30722 (all / oana2026 / research12h), GPUs 0,1,2,3, 16 CPUs, job memory limit 206158430208 bytes, kernel 6.8.0-124-generic.

## GPUs

| GPU | name | memory MiB | bus | NUMA | PCIe now / max | power W | ECC | driver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA RTX A5000 | 24564 | 00000000:06:00.0 | 0 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 1 | NVIDIA RTX A5000 | 24564 | 00000000:45:00.0 | 0 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 2 | NVIDIA RTX A5000 | 24564 | 00000000:46:00.0 | 0 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 3 | NVIDIA RTX A5000 | 24564 | 00000000:47:00.0 | 0 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |

PCIe 'now' is read while the GPUs idle; they drop the link speed to save power, so 'max' is what a transfer gets.

```
[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NODE	NODE	NODE	0-7,128-135	0		N/A
GPU1	NODE	 X 	PIX	PIX	0-7,128-135	0		N/A
GPU2	NODE	PIX	 X 	PIX	0-7,128-135	0		N/A
GPU3	NODE	PIX	PIX	 X 	0-7,128-135	0		N/A

Legend:

  X    = Self
  SYS  = Connection traversing PCIe as well as the SMP interconnect between NUMA nodes (e.g., QPI/UPI)
  NODE = Connection traversing PCIe as well as the interconnect between PCIe Host Bridges within a NUMA node
  PHB  = Connection traversing PCIe as well as a PCIe Host Bridge (typically the CPU)
  PXB  = Connection traversing multiple PCIe bridges (without traversing the PCIe Host Bridge)
  PIX  = Connection traversing at most a single PCIe bridge
  NV#  = Connection traversing a bonded set of # NVLinks

[4mGPU0	GPU1	GPU2	GPU3	[0m
 GPU0	X	OK	OK	OK	
 GPU1	OK	X	OK	OK	
 GPU2	OK	OK	X	OK	
 GPU3	OK	OK	OK	X	

Legend:

  X    = Self
  OK   = Status Ok
  CNS  = Chipset not supported
  GNS  = GPU not supported
  TNS  = Topology not supported
  NS   = Not supported
  U    = Unknown
```

## Bandwidth

Not measured: no python with torch (KV_HOME/venv, KV_HOME/venv_vllm, ~/venvs/vllm-stock): run scripts/a5000_setup.sh first, or pass --python

## CPU, memory, disk, network

```
CPU(s):                                  256
Model name:                              AMD EPYC 9554 64-Core Processor
Thread(s) per core:                      2
Core(s) per socket:                      64
Socket(s):                               2
L3 cache:                                512 MiB (16 instances)
NUMA node(s):                            2
```

Memory: {'MemTotal': '527965448 kB', 'MemAvailable': '407517664 kB', 'HugePages_Total': '0', 'Hugepagesize': '2048 kB'}

```
Filesystem                               Size  Used Avail Use% Mounted on
storage.cs.mcgill.ca:/mnt/pool0/cs/2023  1.7T  1.3T  403G  77% /home/2023
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  166G   22G  89% /
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  166G   22G  89% /
tmpfs                                    252G   93G  160G  37% /dev/shm
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  166G   22G  89% /

quota: (no quota on this node)
```

| interface | state | speed Mb/s | MTU | driver |
| --- | --- | --- | --- | --- |
| bond0 | up | 10000 | 1500 | None |
| bonding_masters | (NotADirectoryError) | (NotADirectoryError) | (NotADirectoryError) | None |
| enp97s0f0np0 | up | 10000 | 1500 | bnxt_en |
| enp97s0f1np1 | up | 10000 | 1500 | bnxt_en |
| ens238f0 | down | -1 | 1500 | None |
| ens238f0np0 | down | (OSError) | 1500 | i40e |
| ens238f1 | down | -1 | 1500 | None |
| ens238f1np1 | down | (OSError) | 1500 | i40e |
| enxbe3af2b6059f | down | (OSError) | 1500 | rndis_host |

RDMA devices: bnxt_re0, bnxt_re1

## Slurm limits

```
associations (account|partition|qos|grptres|maxtres|maxjobs|maxwall):
energy-efficiency-comp597||comp597||||
oana2026||research12h||||

qos (name|maxtresperuser|maxjobsperuser|maxwall|grptres|maxtrespu):
normal||1|04:00:00||
research12h|gres/gpu=4|2|12:00:00||gres/gpu=4
comp579-1gpu-12h|gres/gpu=1|1|12:00:00||gres/gpu=1
comp579-0gpu-4cpu-72h|gres/gpu=0|1|3-00:00:00||gres/gpu=0
gpu_1_mem_32_cpu_3|cpu=4,gres/gpu=1,mem=32G||||cpu=4,gres/gpu=1,mem=32G
comp551|cpu=4,gres/gpu=2,mem=16G|2|12:00:00||cpu=4,gres/gpu=2,mem=16G
comp545|cpu=4,gres/gpu=2,mem=16G|2|12:00:00||cpu=4,gres/gpu=2,mem=16G
comp597|cpu=4,gres/gpu=2,mem=16G|2|06:00:00||cpu=4,gres/gpu=2,mem=16G
disabled||0|||
li-qos||10|||
comp767|cpu=8,gres/gpu=2,mem=32G|2|1-00:00:00||cpu=8,gres/gpu=2,mem=32G
high-memcpu|cpu=24,gres/gpu=2,mem=128G|2|3-00:00:00||cpu=24,gres/gpu=2,mem=128G
comp310|cpu=24,gres/gpu=2,mem=128G|2|14-00:00:00||cpu=24,gres/gpu=2,mem=128G
```

## Software

- python3: Python 3.12.3
- python3.12: Python 3.12.14
- gcc: gcc (Ubuntu 13.3.0-6ubuntu2~24.04.1) 13.3.0
- nvcc: (no nvcc on this node)
- cmake: cmake version 3.28.3
- git: git version 2.43.0
- KV_HOME venv: (not set up)
- KV_HOME venv_vllm: (not set up)
