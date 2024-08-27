## Checkpoints
We provide the finetuned checkpoints at [Google drive](https://drive.google.com/drive/folders/1liy0pWMrLzpa9rZ3TBRkfOw-NtMoksLg?usp=sharing). Alternatively, you can finetune with your own configuration following [Skill-Localization-by-grafting](https://github.com/abhishekpanigrahi1996/Skill-Localization-by-grafting).

## Data
Please follow the repository [LM-BFF-main](https://github.com/princeton-nlp/LM-BFF#prepare-the-data) to download all data. In the remaining codes, we assume that there is a "data" folder containing all the necessary datasets.

## Code
First enter the root directory of the source code.
> cd root_path/Localize-and-Stitch/language/

Run Localize-and-Stitch
> python main_localize_stitch.py

Run Dataless Localize-and-Stitch
> python main_dataless_localize_stitch.py

## Acknowledgement
The codebase is built upon [Skill-Localization-by-grafting](https://github.com/abhishekpanigrahi1996/Skill-Localization-by-grafting).