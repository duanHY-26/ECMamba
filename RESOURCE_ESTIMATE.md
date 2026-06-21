# Computational resource estimate

## Status

All ECMamba values in this document are analytical planning ranges, not measured benchmark results.

## Reference workload

- 100,000 proteins and 20 training epochs
- precomputed ESM-2 representations
- one NVIDIA A100 80 GB accelerator
- automatic mixed precision
- batch size 64
- ESM token length capped at 1000
- MSA depth capped at 64 and alignment length capped at 512
- ESM projection dimension 256
- MSA hidden dimension 128 with two axial blocks

## Projected ranges

| Quantity | Projected range |
|---|---:|
| Peak accelerator memory | 24-36 GB |
| Training wall time | 18-30 accelerator-hours |
| Inference for 10,000 proteins | 3-8 min |

The ranges reflect uncertainty in realized sequence lengths, MSA depths, storage throughput, CUDA kernels, label-vocabulary size, and padding efficiency. They exclude the upstream cost of ESM-2 representation generation and MSA database search.

## Reporting rule

Do not present these projections as measured performance. A release intended to support efficiency claims should report the exact accelerator, CUDA and PyTorch versions, batch size, length distributions, trainable parameter count, peak allocated and reserved memory, examples per second, preprocessing cost, warm-up procedure, and repeated-run variation.

