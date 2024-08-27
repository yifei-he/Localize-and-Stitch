
## Checkpoints

You can download the fine-tuned checkpoints from the authors of [AdaMerging](https://github.com/EnnengYang/AdaMerging) with the following link: [task_vectors_checkpoints](https://drive.google.com/drive/folders/1u_Tva6x0p6oxu5Eo0ZZsf-520Cc_3MKw).

## Data
You can also download the data processed by the authors of AdaMerging from [HuggingFace](https://huggingface.co/collections/tanganke/image-classification-datasets-662abda7d75efe6b0e6b43da).



## Code

First enter the root directory of the source code.
> cd root_path/Localize-and-Stitch/vision

Export python path
> export PYTHONPATH="$PYTHONPATH:$PWD"

Run Localize-and-Stitch
> python main_localize_stitch.py

Run Dataless Localize-and-Stitch
> python main_dataless_localize_stitch.py


## Acknowledgement
The codebase is largely built upon [AdaMerging](https://github.com/EnnengYang/AdaMerging), which also references code from [Task Arithmetic](https://github.com/mlfoundations/task_vectors), [TIES-MERGING](https://github.com/prateeky2806/ties-merging) and [Model Soups](https://github.com/mlfoundations/model-soups). 
