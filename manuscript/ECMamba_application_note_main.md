# ECMamba: gated fusion of protein language model and MSA-aware selective state-space representations for enzyme function annotation

Hongyu Duan^1^,^†^, Mianzhi Dai^2^,^†^, Tianlai Huang^3^, Jing Li^4^, Fanjie Wei^5^, Zhong Li^6^*

^1^ Department of Statistics and Financial Mathematics, School of Mathematics, South China University of Technology, Guangzhou, 510640, China  
^2^ School of Physics, Sun Yat-sen University, Guangzhou, 510275, China  
^3^ Intelligent Speech Interaction Department, Ping An Technology, Shenzhen, China  
^4^ Department of Operations and Medical Insurance Management, The Sixth Affiliated Hospital, Sun Yat-sen University, Guangzhou, China  
^5^ Department of Software Engineering, School of Software Engineering, South China University of Technology, Guangzhou, 510006, China  
^6^ Department of Neurology, The Sixth Affiliated Hospital, Sun Yat-sen University, Guangzhou, China; Key Laboratory of Human Microbiome and Chronic Diseases, Sun Yat-sen University, Ministry of Education, China; Biomedical Innovation Center, The Sixth Affiliated Hospital, Sun Yat-sen University, Guangzhou, China; Guangdong Provincial Key Laboratory of Brain Function and Disease, Guangzhou, 510080, China

\*To whom correspondence should be addressed: lzhong@mail.sysu.edu.cn  
^†^These authors contributed equally to this work.

## Abstract

**Summary:** Reliable Enzyme Commission annotation remains difficult for proteins with weakly characterized homologs and long-tail EC classes. Homology search depends strongly on close matches, sequence-only neural predictors miss explicit family-level conservation, and dense MSA attention models can be expensive on long or deep alignments. ECMamba addresses this gap by combining pretrained protein language model features with explicit evolutionary context through an asymmetric dual-branch architecture. The language-model branch remains lightweight, whereas the MSA branch alternates row-wise Mamba sequence mixing and column-wise homolog attention before feature-wise gated fusion. On the supplied NEW-392 and Price-149 benchmarks, ECMamba achieved weighted precision/recall/F1 scores of 0.662/0.634/0.601 and 0.603/0.592/0.570, respectively. Relative to the strongest sequence-only neural baseline, the F1 gain was 0.009 on NEW-392 and 0.043 on Price-149, with the largest benefit arising from recall.

**Availability and implementation:** ECMamba is implemented in Python. Source code, documentation, configuration examples and manuscript materials are available at `https://github.com/duanHY-26/ECMamba`.

**Contact:** [institutional email]

**Supplementary information:** Supplementary methods, extended benchmark tables, projected computational profile and reproducibility notes are provided in the accompanying supplementary material.

## 1 Introduction

Enzyme annotation becomes unreliable as sequence identity decreases, yet many practical predictors still depend primarily on one information source at a time. Homology-search methods remain effective when close annotated enzymes exist, but they degrade when informative neighbors are sparse. Sequence-only deep models and protein language models improve generalization, yet they still lack direct access to residue agreement across homologs. Conversely, full MSA attention captures evolutionary structure but can be unnecessarily heavy for an application-focused workflow.

ECMamba addresses this gap by assigning different computational roles to the two input sources. Precomputed language-model features are normalized, projected and pooled without applying another heavy token backbone. Evolutionary modeling is concentrated in the MSA branch, where row-wise Mamba follows residue order within each homolog and column-wise attention exchanges information across homologs at aligned positions. A learned feature-wise gate then combines both branches before multi-label classification. The central idea is therefore not merely to add another encoder, but to fuse broad sequence semantics with explicit evolutionary evidence in a computationally selective way.

## 2 Implementation

The implementation accepts precomputed ESM representations together with one MSA per protein. The current workflow trains on the split100 partition and evaluates on NEW and Price, while excluding the HARD split. MSA inputs are normalized from common alignment formats, deduplicated by row and truncated to fixed depth and length caps, and the query sequence can be used as a depth-one fallback when no MSA is available.

The language-model stream produces a compact pooled representation through LayerNorm, linear projection, dropout and attentive, mean and max pooling. The MSA stream embeds aligned residues, alternates row-wise Mamba and column-wise multi-head attention, and pools the updated query row. A feature-wise gate uses the two branch outputs, their absolute difference and their element-wise product to form the fused representation used for EC prediction. This design contributes three practical innovations: linear-time sequence mixing inside each homolog, targeted cross-homolog interaction at aligned columns, and adaptive branch weighting instead of fixed fusion. The released code also records label coverage, threshold sweeps and run configurations for reproducibility, which makes the workflow easier to inspect and reuse.

![Figure 1. ECMamba framework, training flow and inference flow.](e:/研究/ECMamba/ECMamba.png)

## 3 Results

The supplied comparison benchmark shows that ECMamba achieves the best recall and F1 on both displayed datasets. On NEW-392, F1 increased from 0.592 to 0.601 relative to the strongest sequence-only multilayer perceptron baseline. On Price-149, the gain was larger, with recall increasing from 0.520 to 0.592 and F1 from 0.527 to 0.570. This pattern indicates that explicit evolutionary context is particularly helpful under stronger distribution shift, where sensitivity improves without a large loss of precision. In practical terms, the method is most useful when users need a stronger application model than PLM-only baselines, but do not want the complexity of a full dense-attention MSA stack.

![Figure 2. Benchmark comparison on NEW-392 and Price-149.](e:/研究/ECMamba/model_comparison_full.png)

## 4 Discussion

ECMamba is best viewed as a targeted extension of a strong sequence baseline rather than as a wholesale replacement for all existing EC annotation workflows. Its main advantage over mainstream alternatives is that it keeps the strong transferability of pretrained protein representations, adds explicit evolutionary evidence from MSAs, and avoids the cost of applying dense attention everywhere. The most important algorithmic novelty is the asymmetric fusion strategy: a lightweight pooled PLM branch, an MSA branch with row-wise selective state-space modeling and column-wise homolog attention, and a gate that learns when each source should dominate.

The current evidence package still has clear limits. Resource numbers are projected rather than measured, and the supplied benchmark material does not yet include multi-seed uncertainty, ablation controls or gate-value analyses. Even so, the released implementation is easy to use in practice because it works from precomputed embeddings and common MSA formats, provides explicit output files for coverage and threshold analysis, and exposes a compact repository structure rather than a complicated service stack. The remaining methodological details, together with the full machine-learning dataset description, are therefore provided or summarized in the supplementary material, which is the appropriate location for Application Notes of this type.

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
