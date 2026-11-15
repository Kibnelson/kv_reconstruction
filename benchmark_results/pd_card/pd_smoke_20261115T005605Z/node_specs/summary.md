# Node specs: gpu-grad-01.cs.mcgill.ca

2026-11-14 20:10:23 EST; job 30163 (all / oana2026 / research12h), GPUs 0,1,2,3, 16 CPUs, job memory limit 206158430208 bytes, kernel 6.8.0-124-generic.

## GPUs

| GPU | name | memory MiB | bus | NUMA | PCIe now / max | power W | ECC | driver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA RTX A5000 | 24564 | 00000000:07:00.0 | 0 | Gen2 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 1 | NVIDIA RTX A5000 | 24564 | 00000000:C3:00.0 | 1 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 2 | NVIDIA RTX A5000 | 24564 | 00000000:C4:00.0 | 1 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
| 3 | NVIDIA RTX A5000 | 24564 | 00000000:C5:00.0 | 1 | Gen1 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |

PCIe 'now' is read while the GPUs idle; they drop the link speed to save power, so 'max' is what a transfer gets.

```
[4mGPU0	GPU1	GPU2	GPU3	CPU Affinity	NUMA Affinity	GPU NUMA ID[0m
GPU0	 X 	NODE	NODE	NODE	48-55,176-183	0		N/A
GPU1	NODE	 X 	PIX	PIX		1		N/A
GPU2	NODE	PIX	 X 	PIX		1		N/A
GPU3	NODE	PIX	PIX	 X 		1		N/A

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

## Bandwidth, one transfer at a time (896 MiB)

| GPU | to pinned | from pinned | to pageable | from pageable |
| --- | --- | --- | --- | --- |
| 0 | 27.1 GB/s | 27.0 GB/s | 21.4 GB/s | 24.9 GB/s |
| 1 | 27.1 GB/s | 27.0 GB/s | 21.1 GB/s | 24.5 GB/s |
| 2 | 27.1 GB/s | 27.0 GB/s | 20.7 GB/s | 23.7 GB/s |
| 3 | 27.1 GB/s | 27.0 GB/s | 21.4 GB/s | 24.7 GB/s |

| pair | peer | direct | through host | through host, pipelined |
| --- | --- | --- | --- | --- |
| 0 -> 1 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 0 -> 2 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 0 -> 3 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 1 -> 0 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 1 -> 2 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 1 -> 3 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 2 -> 0 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 2 -> 1 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 2 -> 3 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 3 -> 0 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 3 -> 1 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |
| 3 -> 2 | True | 26.4 GB/s | 13.5 GB/s | 26.0 GB/s |

## Bandwidth, the planned layouts (every leg at once, 896 MiB each)

| pattern | path | legs | aggregate | per leg GB/s | correct |
| --- | --- | --- | --- | --- | --- |
| 1p3d, prefill GPU 0 | direct | 0->1, 0->2, 0->3 | 26.4 GB/s (211 Gb/s) | 26.4 / 13.2 / 8.8 | yes |
| 1p3d, prefill GPU 0 | host | 0->1, 0->2, 0->3 | 26.3 GB/s (210 Gb/s) | 26.0 / 13.1 / 8.8 | yes |
| 1p3d, prefill GPU 1 | direct | 1->0, 1->2, 1->3 | 26.4 GB/s (211 Gb/s) | 26.3 / 13.2 / 8.8 | yes |
| 1p3d, prefill GPU 1 | host | 1->0, 1->2, 1->3 | 26.3 GB/s (210 Gb/s) | 25.4 / 13.1 / 8.8 | yes |
| 1p3d, prefill GPU 2 | direct | 2->0, 2->1, 2->3 | 26.4 GB/s (211 Gb/s) | 26.3 / 13.2 / 8.8 | yes |
| 1p3d, prefill GPU 2 | host | 2->0, 2->1, 2->3 | 26.2 GB/s (210 Gb/s) | 25.4 / 12.9 / 8.7 | yes |
| 1p3d, prefill GPU 3 | direct | 3->0, 3->1, 3->2 | 26.4 GB/s (211 Gb/s) | 26.4 / 13.2 / 8.8 | yes |
| 1p3d, prefill GPU 3 | host | 3->0, 3->1, 3->2 | 26.3 GB/s (210 Gb/s) | 26.0 / 13.1 / 8.8 | yes |
| 2p2d, prefill 0,1 | direct | 0->2, 1->3 | 47.3 GB/s (379 Gb/s) | 23.7 / 23.7 | yes |
| 2p2d, prefill 0,1 | host | 0->2, 1->3 | 45.7 GB/s (366 Gb/s) | 23.4 / 22.8 | yes |
| 2p2d, prefill 0,1 | direct | 0->3, 1->2 | 47.3 GB/s (378 Gb/s) | 23.7 / 23.7 | yes |
| 2p2d, prefill 0,1 | host | 0->3, 1->2 | 45.6 GB/s (365 Gb/s) | 23.4 / 22.8 | yes |
| 2p2d, prefill 0,2 | direct | 0->1, 2->3 | 47.3 GB/s (378 Gb/s) | 23.7 / 23.7 | yes |
| 2p2d, prefill 0,2 | host | 0->1, 2->3 | 45.7 GB/s (366 Gb/s) | 23.4 / 22.9 | yes |
| 2p2d, prefill 0,2 | direct | 0->3, 2->1 | 46.0 GB/s (368 Gb/s) | 23.0 / 23.0 | yes |
| 2p2d, prefill 0,2 | host | 0->3, 2->1 | 45.6 GB/s (365 Gb/s) | 23.5 / 22.8 | yes |
| 2p2d, prefill 0,3 | direct | 0->1, 3->2 | 46.2 GB/s (369 Gb/s) | 23.1 / 23.1 | yes |
| 2p2d, prefill 0,3 | host | 0->1, 3->2 | 45.8 GB/s (366 Gb/s) | 23.4 / 22.9 | yes |
| 2p2d, prefill 0,3 | direct | 0->2, 3->1 | 47.4 GB/s (379 Gb/s) | 23.7 / 23.7 | yes |
| 2p2d, prefill 0,3 | host | 0->2, 3->1 | 45.5 GB/s (364 Gb/s) | 23.4 / 22.7 | yes |
| 2p2d, prefill 1,2 | direct | 1->0, 2->3 | 52.6 GB/s (421 Gb/s) | 26.3 / 26.3 | yes |
| 2p2d, prefill 1,2 | host | 1->0, 2->3 | 33.6 GB/s (269 Gb/s) | 25.0 / 16.8 | yes |
| 2p2d, prefill 1,2 | direct | 1->3, 2->0 | 52.6 GB/s (421 Gb/s) | 26.3 / 26.3 | yes |
| 2p2d, prefill 1,2 | host | 1->3, 2->0 | 34.0 GB/s (272 Gb/s) | 17.0 / 17.0 | yes |
| 2p2d, prefill 1,3 | direct | 1->0, 3->2 | 52.7 GB/s (422 Gb/s) | 26.4 / 26.3 | yes |
| 2p2d, prefill 1,3 | host | 1->0, 3->2 | 33.7 GB/s (269 Gb/s) | 25.0 / 16.8 | yes |
| 2p2d, prefill 1,3 | direct | 1->2, 3->0 | 52.7 GB/s (422 Gb/s) | 26.4 / 26.3 | yes |
| 2p2d, prefill 1,3 | host | 1->2, 3->0 | 34.1 GB/s (272 Gb/s) | 17.0 / 17.0 | yes |
| 2p2d, prefill 2,3 | direct | 2->0, 3->1 | 52.5 GB/s (420 Gb/s) | 26.4 / 26.3 | yes |
| 2p2d, prefill 2,3 | host | 2->0, 3->1 | 33.7 GB/s (269 Gb/s) | 25.0 / 16.8 | yes |
| 2p2d, prefill 2,3 | direct | 2->1, 3->0 | 52.6 GB/s (421 Gb/s) | 26.4 / 26.3 | yes |
| 2p2d, prefill 2,3 | host | 2->1, 3->0 | 34.0 GB/s (272 Gb/s) | 17.0 / 17.0 | yes |
| cache reload to 1 GPU(s) | direct | host->0 | 27.0 GB/s (216 Gb/s) | 27.0 | yes |
| cache reload to 2 GPU(s) | direct | host->0, host->1 | 54.0 GB/s (432 Gb/s) | 27.0 / 27.0 | yes |
| cache reload to 3 GPU(s) | direct | host->0, host->1, host->2 | 75.4 GB/s (603 Gb/s) | 27.0 / 25.1 / 25.1 | yes |
| cache reload to 4 GPU(s) | direct | host->0, host->1, host->2, host->3 | 66.6 GB/s (533 Gb/s) | 27.0 / 16.7 / 16.7 / 16.6 | yes |
| cache store from 4 GPU(s) | direct | 0->host, 1->host, 2->host, 3->host | 70.9 GB/s (568 Gb/s) | 26.3 / 17.7 / 17.7 / 17.7 | yes |

Best placement per layout and path:

- 1p3d direct: 1p3d, prefill GPU 3 (3->0, 3->1, 3->2), 26.4 GB/s aggregate
- 1p3d host: 1p3d, prefill GPU 0 (0->1, 0->2, 0->3), 26.3 GB/s aggregate
- 2p2d direct: 2p2d, prefill 1,3 (1->0, 3->2), 52.7 GB/s aggregate
- 2p2d host: 2p2d, prefill 0,3 (0->1, 3->2), 45.8 GB/s aggregate

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

Memory: {'MemTotal': '527965448 kB', 'MemAvailable': '433022828 kB', 'HugePages_Total': '0', 'Hugepagesize': '2048 kB'}

```
Filesystem                               Size  Used Avail Use% Mounted on
storage.cs.mcgill.ca:/mnt/pool0/cs/2023  1.7T  1.3T  408G  77% /home/2023
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  157G   31G  84% /
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  157G   31G  84% /
tmpfs                                    252G  622M  252G   1% /dev/shm
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  157G   31G  84% /

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
- KV_HOME venv: 3.12.14, torch 2.11.0+cu128, triton 3.6.0, vllm -, nixl -, lmcache -, nvidia.nvcomp 5.3.0, ibp -
- KV_HOME venv_vllm: 3.12.14, torch 2.13.0+cu130, triton 3.7.1, vllm 0.32.0, nixl -, lmcache -, nvidia.nvcomp 6.0.0, ibp -
