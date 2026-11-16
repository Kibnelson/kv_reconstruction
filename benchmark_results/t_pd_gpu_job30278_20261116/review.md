# Four-GPU IOMMU follow-up: job 30278

The supplied stdout records a Slurm allocation on `gpu-grad-01` with
`GPUS=4`. All four inspected GPU IOMMU groups report `DMA-FQ`, and
`systemd-detect-virt` reports `none`.

This is user-supplied configuration evidence, received November 16, 2026.
The command was published at `9816bf3`, but it printed no Git pin or capture
timestamp. It is a separate allocation from failed-copy job 30224; it did
not run a new copy benchmark and does not retroactively assign its settings
or GPU identities to that earlier job.

## Observed GPU group configuration

| PCI bus ID | IOMMU group | Reported default-domain type |
|---|---:|---|
| 0000:06:00.0 | 55 | DMA-FQ |
| 0000:45:00.0 | 30 | DMA-FQ |
| 0000:46:00.0 | 31 | DMA-FQ |
| 0000:47:00.0 | 32 | DMA-FQ |

Linux defines DMA-FQ as translated DMA with deferred invalidation.
NVIDIA documents translated IOMMU mode as a concern for PCIe GPU peer
transfers on Linux bare-metal systems, including cases where capability
queries report OK. With no virtualization detected, the observed
configuration matches that documented limitation. It is a strong candidate
explanation for the failed peer transfers, not a causal repair-and-retest
proof. No host setting was changed by our checks.

Job 30224 ended with all 64 MiB of the destination still equal to its initial
inverse-reference pattern in the four tested directions. Successful peer
enable and API responses did not establish a correct copy. Those directions
were 0→1, 1→0, 1→2 and 2→1: every tested pair included logical CUDA device 1.
We have not measured every GPU pair, and the logical CUDA-to-PCI mapping
was not included in the supplied stdout.

## Reported GPU identities

| nvidia-smi index | UUID in job 30278 |
|---|---|
| 0 | GPU-256b70c0-c8bc-c483-c767-3a5c27ed7286 |
| 1 | GPU-1593dd02-947c-388d-92af-e2c750110f35 |
| 2 | GPU-2680c039-8dff-cb93-2c6a-9e89e46586d7 |
| 3 | GPU-06931f50-f453-a006-ef25-da5c555c0b34 |

The [published job 30196 node record](https://github.com/Kibnelson/kv_reconstruction/blob/90012974da8fe60ae67f20d4cb4b1f47760cb835/benchmark_results/pd_card/pd_smoke_tmatched_20261115T112147Z_594875/node_specs/specs.json)
contains `GPU-683e95a6-81fe-c256-8c35-a91c6e3d38ff` at nvidia-smi index 1.
The other three UUIDs match. This establishes a difference in reported GPU
identity sets between jobs 30196 and 30278, not the cause or date of that
difference. The job 30224 UUID list was not supplied, so no identity change
between jobs 30224 and 30278 is asserted. A physical replacement or a
particular faulty GPU is not established.

## Administrator follow-up and acceptance

The next configuration action belongs with the node administrator under
cluster maintenance controls. Provide the failed-copy report and this
follow-up, and request review of translated GPU IOMMU domains, the PCIe/ACS
configuration and the driver. Have the administrator establish a supported
peer-transfer configuration and retain before/after settings and hardware
identities. NVIDIA's `simpleP2P` is an independent data-integrity check
recommended in its troubleshooting guidance.

Use the complete pair sweep below to establish the affected links on the
current reported GPU set before maintenance, and repeat it after a
configuration change to validate the repair. Every check stays inside our
four-GPU allocation. The command selects every ordered pair, including pairs
that exclude logical device 1. It produces 48 route cases: 12 ordered pairs times the
three original routes and the independent CUDA-runtime route. A pass still
requires all requested copies and controls to succeed. The subsequent full
node survey also tests concurrent patterns before a serving campaign can
pass its gate.

```bash
cd "$HOME/kv_t_pd_independent" &&
git fetch --no-tags origin experiment/t-pd-matched-campaign-v1 &&
git merge --ff-only FETCH_HEAD &&
T_PD_DIAG_PAIRS=0-1,1-0,0-2,2-0,0-3,3-0,1-2,2-1,1-3,3-1,2-3,3-2 \
bash scripts/run_t_pd_diagnostics.sh
```

A configuration change is not accepted as a fix until the exact GPU
retest passes. The full matched serving comparison remains pending.

Future diagnostic reports now collect each GPU group's domain type and the
virtualization command output within the GPU allocation. The GPU inventory
also includes PCI bus IDs alongside UUIDs; runtime-only case records already
contain logical CUDA bus-ID lists. This allows explicit mapping rather than
assuming enumeration orders agree. Only metadata capture was changed;
copy algorithms, gates, package pins and allocation sizes are unchanged.
The metadata regression expectations were updated, but no CPU test suite
or new GPU job was executed by the assistant for this change.

## Primary references

- [Linux IOMMU sysfs ABI](https://github.com/torvalds/linux/blob/v6.8/Documentation/ABI/testing/sysfs-kernel-iommu_groups).
- [NVIDIA GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).

