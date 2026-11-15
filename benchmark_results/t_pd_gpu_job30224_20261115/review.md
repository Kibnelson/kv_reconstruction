# GPU peer-copy failure: job 30224

Source pin: `470cc96ef20b3a00ac30f5f574942d91ab3b9cd5`  
Attempt: `links_20261115T170655Z_720208`  
Node: `gpu-grad-01`; kernel: `6.8.0-124-generic`.

This review transcribes the user's console output. The original
`share.zip` has not been independently retrieved. Its home location is
`~/.kvpd_t_matched/diagnostics/links_20261115T170655Z_720208/share.zip`.
The review is not a replacement for that measured evidence.

## Observations

| Direction | Payload bytes | Mismatched bytes | Bytes still equal to initial poison | Source unchanged |
|---|---:|---:|---:|---|
| 0→1 | 67,108,864 | 67,108,864 | 67,108,864 | True |
| 1→0 | 67,108,864 | 67,108,864 | 67,108,864 | True |
| 1→2 | 67,108,864 | 67,108,864 | 67,108,864 | True |
| 2→1 | 67,108,864 | 67,108,864 | 67,108,864 | True |

Both directions of peer enabling reported `enabled` for each case.
The read/write capability matrices reported OK for every distinct GPU pair.
Nevertheless, after `cudaMemcpyPeer` and device synchronization, each tested
destination retained the complete initial inverse-reference pattern. The
API response and advertised capability did not establish a correct transfer.

At this source pin, the runtime worker imports no Torch and checks both
selected GPUs' H2D, local D2D and D2H round trips before a peer trial. Reaching
the reported trial implies those local controls succeeded. This observation
comes from the pinned worker's control flow; the supplied summary does not
reprint the complete local-control JSON. Earlier shared logs also show failed
Torch direct copies and successful host-staged cases.

This narrows the investigation to the peer-transfer path and its runtime,
driver or platform conditions. It does not establish a physical GPU defect,
an IOMMU fault, an ACS fault or a particular software defect. No T/T2 codec
is invoked by these byte-copy tests, and they provide no serving speed result.

## IOMMU evidence and remaining uncertainty

| Reported PCI row order | PCI bus ID | Reported IOMMU group | Domain type |
|---|---|---:|---|
| 0 | 0000:06:00.0 | 55 | Not yet read |
| 1 | 0000:45:00.0 | 30 | Not yet read |
| 2 | 0000:46:00.0 | 31 | Not yet read |
| 3 | 0000:47:00.0 | 32 | Not yet read |

These PCI rows are in the reported nvidia-smi order. The supplied excerpt
does not establish their mapping to the logical CUDA indexes in the pair
tests; that requires the worker's bus-ID list and visible-device mapping.

The system reported 122 groups and no explicit relevant IOMMU boot flags.
That does not identify the GPUs' default domain types. The Linux ABI
exposes the group `type`: `DMA`/`DMA-FQ` translate DMA addresses, while
`identity` does not. NVIDIA documents limitations for Linux bare-metal
PCIe peer transfers under IOMMU translation, and notes that a capability
matrix can show OK while transfers still fail. These are reasons to inspect
the active configuration, not proof of causality.

The existing full campaign requires a successful full hardware/bandwidth
survey and will stop before the serving matrix if that survey fails. No
scope was silently reduced and no gate, codec, transport or system setting
was changed by this review.

## Read-only follow-up

Run from the login host, outside an existing Slurm allocation. This requests
one CPU, 1 GiB RAM and no GPU resources for up to five minutes. It reads
the current configuration, which is a follow-up allocation rather than a
measurement from job 30224. The PCI IDs are resolved to their current groups
instead of assuming that group numbers survived a reboot.

```bash
bash <<'READ_DOMAIN'
set -Eeuo pipefail
test -z "${SLURM_JOB_ID:-}"
if type module >/dev/null 2>&1; then module load slurm; fi

srun -p all -A oana2026 -q research12h \
    -N 1 -w gpu-grad-01 -n 1 -c 1 --mem=1G --gres=none -t 00:05:00 \
    --chdir=/tmp --export=ALL --propagate=NONE bash -s <<'NODE'
set -Eeuo pipefail
test "$(hostname -s)" = gpu-grad-01
printf 'HOST=%s\n' "$(hostname -s)"
for bdf in 0000:06:00.0 0000:45:00.0 0000:46:00.0 0000:47:00.0; do
    group_path=$(readlink -e "/sys/bus/pci/devices/$bdf/iommu_group" || true)
    domain=unavailable
    if [ -n "$group_path" ] && [ -r "$group_path/type" ]; then
        domain=$(cat "$group_path/type")
    fi
    printf '%s group=%s type=%s\n' "$bdf" "${group_path##*/}" "$domain"
done
if command -v systemd-detect-virt >/dev/null 2>&1; then
    printf 'VIRTUALIZATION='
    systemd-detect-virt || true
fi
NODE
READ_DOMAIN
```

`DMA` or `DMA-FQ` would provide a concrete translated-domain condition for
the node administrator to investigate. `identity` would not establish this
specific explanation; driver/PCIe/ACS and independent sample checks remain
possible follow-ups. A permission or missing-file result remains unknown.
Only a controlled repair followed by exact GPU retests can establish that
a configuration change resolves the failure.

## Primary references

- [Linux 6.8 IOMMU sysfs ABI](https://github.com/torvalds/linux/blob/v6.8/Documentation/ABI/testing/sysfs-kernel-iommu_groups).
- [NVIDIA GPU troubleshooting](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/troubleshooting/gpu_troubleshooting.html).
- [CUDA 12.8 IOMMU and peer-copy documentation](https://docs.nvidia.com/cuda/archive/12.8.2/cuda-c-programming-guide/index.html#iommu-on-linux).

