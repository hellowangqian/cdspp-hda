# Cross Domain Structure Preserving Projection for Heterogeneous Domain Adaptation

_"Heterogeneous Domain Adaptation (HDA) addresses the transfer learning problems where data from the source and target domains are of different modalities (e.g., texts and images) or feature dimensions (e.g., features extracted with different methods). It is useful for multi-modal data analysis. Traditional domain adaptation algorithms assume that the representations of source and target samples reside in the same feature space, hence are likely to fail in solving the heterogeneous domain adaptation problem. Contemporary state-of-the-art HDA approaches are usually composed of complex optimization objectives for favourable performance and are therefore computationally expensive and less generalizable. To address these issues, we propose a novel Cross-Domain Structure Preserving Projection (CDSPP) algorithm for HDA. As an extension of the classic LPP to heterogeneous domains, CDSPP aims to learn domain-specific projections to map sample features from source and target domains into a common subspace such that the class consistency is preserved and data distributions are sufficiently aligned. CDSPP is simple and has deterministic solutions by solving a generalized eigenvalue problem. It is naturally suitable for supervised HDA but has also been extended for semi-supervised HDA where the unlabelled target domain samples are available. Extensive experiments have been conducted on commonly used benchmark datasets (i.e. Office-Caltech, Multilingual Reuters Collection, NUS-WIDE-ImageNet) for HDA as well as the Office-Home dataset firstly introduced for HDA by ourselves due to its significantly larger number of classes than the existing ones (65 vs 10, 6 and 8). The experimental results of both supervised and semi-supervised HDA demonstrate the superior performance of our proposed method against contemporary state-of-the-art methods."_

[[Wang, Breckon, _Pattern Recognition_, Vol 123, March 2022](https://breckon.org/toby/publications/papers/wang22crossdomain.pdf)]


## Dataset

You can download extracted features (Office-Caltech, Office-Home) used in our experiments from [Durham Collections](https://collections.durham.ac.uk):

**Cross-Domain Structure Preserving Projection for Heterogeneous Domain Adaptation - Supporting Dataset
[ [DOI](http://doi.org/10.15128/r2jw827b67n) ]

(or alternatively from  [Dropbox](https://www.dropbox.com/sh/293h2sij1oirn3y/AAD_J8ZReGHglzw84RSs6sb8a?dl=0) or [BaiduPan](https://pan.baidu.com/s/1tLfPuOj8745bme4omzAcNg) - code: 57ar)

## Reference

If making use of this code or collated dataset in your own work please cite:

```
@article{wang22crossdomain,
 author = {Wang, Q. and Breckon, T.P.},
 title = {Cross-Domain Structure Preserving Projection for Heterogeneous Domain Adaptation},
 journal = {Pattern Recognition},
 year = {2022},
 volume = {123},
 month = {March},
 publisher = {Elsevier},
 keywords = {heterogeneous domain adaptation, cross-domain projection, image classification, text classification},
 url = {https://breckon.org/toby/publications/papers/wang22crossdomain.pdf},
 doi = {https://doi.org/10.1016/j.patcog.2021.108362},
 arxiv = {https://arxiv.org/abs/2004.12427},
}

```

## Contact

Qian Wang - qian.wang173@hotmail.com
