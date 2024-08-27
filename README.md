## Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic

This is the official implementation for the algorithm Localize-and-Stitch in the paper ["Localize-and-Stitch: Efficient Model Merging via Sparse Task Arithmetic"](https://arxiv.org/abs/2408.13656).

![alt text](localize_and_stitch.png "Localize-and-Stitch")


# Abstract
Model merging offers an effective strategy to combine the strengths of multiple finetuned models into a unified model that preserves the specialized capabilities of each. Existing methods merge models in a global manner, performing arithmetic operations across all model parameters. However, such global merging often leads to task interference, degrading the performance of the merged model. In this work, we introduce Localize-and-Stitch, a novel approach that merges models in a localized way. Our algorithm works in two steps: i) Localization: identify tiny (1% of the total parameters) localized regions in the finetuned models containing essential skills for the downstream tasks, and ii) Stitching: reintegrate only these essential regions back into the pretrained model for task synergy. We demonstrate that our approach effectively locates sparse regions responsible for finetuned performance, and the localized regions could be treated as compact and interpretable representations of the finetuned models (tasks). Empirically, we evaluate our method on various vision and language benchmarks, showing that it outperforms existing model merging methods under different data availability scenarios. Beyond strong empirical performance, our algorithm also facilitates model compression and preserves pretrained knowledge, enabling flexible and continual skill composition from multiple finetuned models with minimal storage and computational overhead.


# Experiments
Both the vision and language experiments in the paper can be reproduced in the corresponding folders. Please check the readme in the folders for details on the finetuned checkpoints, data and running the code.

# Citation
