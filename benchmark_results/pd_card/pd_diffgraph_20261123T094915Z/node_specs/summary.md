# Node specs: gpu-grad-01.cs.mcgill.ca

2026-11-23 05:02:14 EST; job 30583 (all / oana2026 / research12h), GPUs 0,1,2,3, 16 CPUs, job memory limit 206158430208 bytes, kernel 6.8.0-124-generic.

## GPUs

| GPU | name | memory MiB | bus | NUMA | PCIe now / max | power W | ECC | driver |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | NVIDIA RTX A5000 | 24564 | 00000000:06:00.0 | 0 | Gen4 x16 / Gen4 x16 | 230.00 | Disabled | 595.71.05 |
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

## Bandwidth, one transfer at a time (896 MiB)

| GPU | to pinned | from pinned | to pageable | from pageable |
| --- | --- | --- | --- | --- |
| 0 | 26.4 GB/s | 27.0 GB/s | 19.6 GB/s | 24.9 GB/s |
| 1 | 26.4 GB/s | 27.0 GB/s | 20.5 GB/s | 25.0 GB/s |
| 2 | 26.4 GB/s | 27.0 GB/s | 23.0 GB/s | 25.0 GB/s |
| 3 | 26.4 GB/s | 27.0 GB/s | 22.7 GB/s | 25.0 GB/s |

| pair | peer | direct | through host | through host, pipelined |
| --- | --- | --- | --- | --- |
| 0 -> 1 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 0 -> 2 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 0 -> 3 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 1 -> 0 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 1 -> 2 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 1 -> 3 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 2 -> 0 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 2 -> 1 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 2 -> 3 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 3 -> 0 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 3 -> 1 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |
| 3 -> 2 | True | 9.5 GB/s | 13.3 GB/s | 25.4 GB/s |

## Bandwidth, the planned layouts (every leg at once, 896 MiB each)

| pattern | path | legs | aggregate | per leg GB/s | correct |
| --- | --- | --- | --- | --- | --- |
| 1p3d, prefill GPU 0 | direct | 0->1, 0->2, 0->3 | 9.5 GB/s (76 Gb/s) | 9.5 / 4.8 / 3.2 | NO |
| 1p3d, prefill GPU 0 | host | 0->1, 0->2, 0->3 | 26.0 GB/s (208 Gb/s) | 25.4 / 12.9 / 8.7 | yes |
| 1p3d, prefill GPU 1 | direct | 1->0, 1->2, 1->3 | 9.5 GB/s (76 Gb/s) | 9.5 / 4.8 / 3.2 | NO |
| 1p3d, prefill GPU 1 | host | 1->0, 1->2, 1->3 | 26.0 GB/s (208 Gb/s) | 25.4 / 12.9 / 8.7 | yes |
| 1p3d, prefill GPU 2 | direct | 2->0, 2->1, 2->3 | 9.5 GB/s (76 Gb/s) | 9.5 / 4.8 / 3.2 | NO |
| 1p3d, prefill GPU 2 | host | 2->0, 2->1, 2->3 | 26.0 GB/s (208 Gb/s) | 25.4 / 12.9 / 8.7 | yes |
| 1p3d, prefill GPU 3 | direct | 3->0, 3->1, 3->2 | 9.5 GB/s (76 Gb/s) | 9.5 / 4.8 / 3.2 | NO |
| 1p3d, prefill GPU 3 | host | 3->0, 3->1, 3->2 | 26.0 GB/s (208 Gb/s) | 25.4 / 12.9 / 8.7 | yes |
| 2p2d, prefill 0,1 | direct | 0->2, 1->3 | 18.9 GB/s (151 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 0,1 | host | 0->2, 1->3 | 47.5 GB/s (380 Gb/s) | 24.5 / 23.7 | yes |
| 2p2d, prefill 0,1 | direct | 0->3, 1->2 | 19.1 GB/s (153 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 0,1 | host | 0->3, 1->2 | 47.7 GB/s (381 Gb/s) | 24.4 / 23.8 | yes |
| 2p2d, prefill 0,2 | direct | 0->1, 2->3 | 19.0 GB/s (152 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 0,2 | host | 0->1, 2->3 | 47.5 GB/s (380 Gb/s) | 24.4 / 23.8 | yes |
| 2p2d, prefill 0,2 | direct | 0->3, 2->1 | 19.1 GB/s (152 Gb/s) | 10.4 / 9.5 | NO |
| 2p2d, prefill 0,2 | host | 0->3, 2->1 | 47.6 GB/s (380 Gb/s) | 24.4 / 23.8 | yes |
| 2p2d, prefill 0,3 | direct | 0->1, 3->2 | 19.0 GB/s (152 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 0,3 | host | 0->1, 3->2 | 47.7 GB/s (381 Gb/s) | 24.4 / 23.8 | yes |
| 2p2d, prefill 0,3 | direct | 0->2, 3->1 | 19.0 GB/s (152 Gb/s) | 10.4 / 9.5 | NO |
| 2p2d, prefill 0,3 | host | 0->2, 3->1 | 47.5 GB/s (380 Gb/s) | 24.4 / 23.7 | yes |
| 2p2d, prefill 1,2 | direct | 1->0, 2->3 | 18.6 GB/s (149 Gb/s) | 9.5 / 9.3 | NO |
| 2p2d, prefill 1,2 | host | 1->0, 2->3 | 33.4 GB/s (267 Gb/s) | 24.6 / 16.7 | yes |
| 2p2d, prefill 1,2 | direct | 1->3, 2->0 | 19.1 GB/s (152 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 1,2 | host | 1->3, 2->0 | 33.7 GB/s (269 Gb/s) | 16.8 / 16.8 | yes |
| 2p2d, prefill 1,3 | direct | 1->0, 3->2 | 19.0 GB/s (152 Gb/s) | 11.1 / 9.5 | NO |
| 2p2d, prefill 1,3 | host | 1->0, 3->2 | 33.5 GB/s (268 Gb/s) | 24.6 / 16.7 | yes |
| 2p2d, prefill 1,3 | direct | 1->2, 3->0 | 19.0 GB/s (152 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 1,3 | host | 1->2, 3->0 | 33.8 GB/s (270 Gb/s) | 16.9 / 16.9 | yes |
| 2p2d, prefill 2,3 | direct | 2->0, 3->1 | 19.0 GB/s (152 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 2,3 | host | 2->0, 3->1 | 33.4 GB/s (267 Gb/s) | 24.6 / 16.7 | yes |
| 2p2d, prefill 2,3 | direct | 2->1, 3->0 | 18.9 GB/s (151 Gb/s) | 9.5 / 9.5 | NO |
| 2p2d, prefill 2,3 | host | 2->1, 3->0 | 33.8 GB/s (270 Gb/s) | 16.9 / 16.9 | yes |
| cache reload to 1 GPU(s) | direct | host->0 | 27.0 GB/s (216 Gb/s) | 27.0 | yes |
| cache reload to 2 GPU(s) | direct | host->0, host->1 | 54.0 GB/s (432 Gb/s) | 27.0 / 27.0 | yes |
| cache reload to 3 GPU(s) | direct | host->0, host->1, host->2 | 79.6 GB/s (637 Gb/s) | 27.0 / 26.5 / 26.5 | yes |
| cache reload to 4 GPU(s) | direct | host->0, host->1, host->2, host->3 | 71.0 GB/s (568 Gb/s) | 27.0 / 17.7 / 17.7 / 17.7 | yes |
| cache store from 4 GPU(s) | direct | 0->host, 1->host, 2->host, 3->host | 70.4 GB/s (563 Gb/s) | 26.4 / 17.6 / 17.6 / 17.6 | yes |

Best placement per layout and path:

- 1p3d direct: 1p3d, prefill GPU 1 (1->0, 1->2, 1->3), 9.5 GB/s aggregate
- 1p3d host: 1p3d, prefill GPU 0 (0->1, 0->2, 0->3), 26.0 GB/s aggregate
- 2p2d direct: 2p2d, prefill 0,1 (0->3, 1->2), 19.1 GB/s aggregate
- 2p2d host: 2p2d, prefill 0,1 (0->3, 1->2), 47.7 GB/s aggregate

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

Memory: {'MemTotal': '527965448 kB', 'MemAvailable': '482126656 kB', 'HugePages_Total': '0', 'Hugepagesize': '2048 kB'}

```
Filesystem                               Size  Used Avail Use% Mounted on
storage.cs.mcgill.ca:/mnt/pool0/cs/2023  1.7T  1.3T  410G  76% /home/2023
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  162G   26G  87% /
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  162G   26G  87% /
tmpfs                                    252G   23G  230G   9% /dev/shm
/dev/mapper/ubuntu--vg-ubuntu--lv        197G  162G   26G  87% /

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
- KV_HOME venv_vllm: 3.12.14, torch 2.13.0+cu130, triton 3.7.1, vllm 0.32.0, nixl ?, lmcache -, nvidia.nvcomp 6.0.0, ibp -
