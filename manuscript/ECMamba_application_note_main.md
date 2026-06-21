# ECMamba: gated fusion of protein language model and MSA-aware selective state-space representations for enzyme function annotation

[Author names to be inserted]

[Affiliations to be inserted]

\*To whom correspondence should be addressed: [institutional email]

## Abstract

**Summary:** Reliable Enzyme Commission annotation remains difficult for proteins with weakly characterized homologs and for long-tail EC classes. ECMamba combines pretrained protein language model features with explicit evolutionary context from multiple sequence alignments through an asymmetric dual-branch architecture. The language-model branch remains lightweight, whereas the MSA branch alternates row-wise Mamba sequence mixing and column-wise homolog attention before feature-wise gated fusion. On the supplied NEW-392 and Price-149 benchmarks, ECMamba achieved weighted precision/recall/F1 scores of 0.662/0.634/0.601 and 0.603/0.592/0.570, respectively. Relative to the strongest sequence-only neural baseline, the F1 gain was 0.009 on NEW-392 and 0.043 on Price-149, with the largest benefit arising from recall.

**Availability and implementation:** ECMamba is implemented in Python. Source code, documentation, configuration examples and manuscript materials are available at `https://github.com/duanHY-26/ECMamba`.

**Contact:** [institutional email]

**Supplementary information:** Supplementary methods, extended benchmark tables, projected computational profile and reproducibility notes are provided in the accompanying supplementary material.

## 1 Introduction

Enzyme annotation becomes unreliable as sequence identity decreases, yet many practical predictors still depend primarily on either homology search or single-sequence representations. Protein language models provide broad sequence semantics, but they do not directly encode conservation and residue agreement across homologs. Multiple sequence alignments expose that complementary signal, although dense all-cell attention can be expensive on long or deep MSAs.

ECMamba addresses this gap by assigning different computational roles to the two input sources. Precomputed language-model features are normalized, projected and pooled without applying another heavy token backbone. Evolutionary modeling is concentrated in the MSA branch, where row-wise Mamba follows residue order within each homolog and column-wise attention exchanges information across homologs at aligned positions. A learned feature-wise gate then combines both branches before multi-label classification.

## 2 Implementation

The implementation accepts precomputed ESM representations together with one MSA per protein. The current workflow trains on the split100 partition and evaluates on NEW and Price, while excluding the HARD split. MSA inputs are normalized from common alignment formats, deduplicated by row and truncated to fixed depth and length caps, and the query sequence can be used as a depth-one fallback when no MSA is available.

The language-model stream produces a compact pooled representation through LayerNorm, linear projection, dropout and attentive, mean and max pooling. The MSA stream embeds aligned residues, alternates row-wise Mamba and column-wise multi-head attention, and pools the updated query row. A feature-wise gate uses the two branch outputs, their absolute difference and their element-wise product to form the fused representation used for EC prediction. The released code also records label coverage, threshold sweeps and run configurations for reproducibility.

![Figure 1. ECMamba framework, training flow and inference flow.](e:/研究/ECMamba/ECMamba.png)

## 3 Results

The supplied comparison benchmark shows that ECMamba achieves the best recall and F1 on both displayed datasets. On NEW-392, F1 increased from 0.592 to 0.601 relative to the strongest sequence-only multilayer perceptron baseline. On Price-149, the gain was larger, with recall increasing from 0.520 to 0.592 and F1 from 0.527 to 0.570. This pattern indicates that explicit evolutionary context is particularly helpful under stronger distribution shift, where sensitivity improves without a large loss of precision.

![Figure 2. Benchmark comparison on NEW-392 and Price-149.](e:/研究/ECMamba/model_comparison_full.png)

## 4 Discussion

ECMamba is best viewed as a targeted extension of a strong sequence baseline rather than as a wholesale replacement for all existing EC annotation workflows. Its main advantage is that it introduces explicit evolutionary evidence without adding another heavy sequence backbone to the large pretrained language-model features. This keeps the method conceptually simple while improving recall on both evaluation datasets and showing a clearer gain on the shifted Price-149 benchmark.

The current evidence package still has clear limits. Resource numbers are projected rather than measured, and the supplied benchmark material does not yet include multi-seed uncertainty, ablation controls or gate-value analyses. These additional details, together with the full machine-learning dataset description, are therefore provided or summarized in the supplementary material, which is the appropriate location for Application Notes of this type.

## Funding

[Funding information to be inserted.]

## Conflict of interest

The authors declare no competing interests.

## References

1. Lin,Z. *et al.* (2023) Evolutionary-scale prediction of atomic-level protein structure with a language model. *Science*, **379**, 1123-1130.
2. Rao,R. *et al.* (2021) MSA Transformer. *Proceedings of the 38th International Conference on Machine Learning*, 8844-8856.
3. Gu,A. and Dao,T. (2024) Mamba: Linear-Time Sequence Modeling with Selective State Spaces. *Proceedings of the First Conference on Language Modeling*.
4. Yu,T. *et al.* (2023) Enzyme function prediction using contrastive learning. *Science*, **379**, 1358-1363.
5. Sanderson,T. *et al.* (2023) ProteInfer: deep networks for protein functional inference. *eLife*, **12**, e80942.
6. Ryu,J.Y. *et al.* (2019) Deep learning enables high-quality and high-throughput prediction of enzyme commission numbers. *Proc. Natl. Acad. Sci. USA*, **116**, 13996-14001.
7. Li,Y. *et al.* (2018) DEEPre: sequence-based enzyme EC number prediction by deep learning. *Bioinformatics*, **34**, 760-769.
8. Dalkiran,A. *et al.* (2018) ECPred: a tool for the prediction of the enzymatic functions of protein sequences based on the EC nomenclature. *BMC Bioinformatics*, **19**, 334.
