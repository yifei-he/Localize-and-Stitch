import os
import argparse
from transformers import GlueDataTrainingArguments as DataTrainingArguments
import torch
from dataclasses import dataclass, field

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('~/data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--eval-datasets",
        default=None,
        type=lambda x: x.split(","),
        help="Which datasets to use for evaluation. Split by comma, e.g. MNIST,EuroSAT. "
    )
    parser.add_argument(
        "--train-dataset",
        default=None,
        type=lambda x: x.split(","),
        help="Which dataset(s) to patch on.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="Name of the experiment, for organization purposes only."
    )
    parser.add_argument(
        "--results-db",
        type=str,
        default=None,
        help="Where to store the results, else does not store",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="The type of model (e.g. RN50, ViT-B-32).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.1,
        help="Weight decay"
    )
    parser.add_argument(
        "--ls",
        type=float,
        default=0.0,
        help="Label smoothing."
    )
    parser.add_argument(
        "--warmup_length",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--load",
        type=lambda x: x.split(","),
        default=None,
        help="Optionally load _classifiers_, e.g. a zero shot classifier or probe or ensemble both.",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optionally save a _classifier_, e.g. a zero shot classifier or probe.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory for caching features and encoder",
    )
    parser.add_argument(
        "--openclip-cachedir",
        type=str,
        # default='/gscratch/efml/gamaga/.cache/open_clip',
        default='/data/common/task-arithmetic/.cache/open_clip',
        help='Directory for caching models from OpenCLIP'
    )
    parsed_args = parser.parse_args()
    parsed_args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if parsed_args.load is not None and len(parsed_args.load) == 1:
        parsed_args.load = parsed_args.load[0]
    return parsed_args


def parse_data_arguments(dataset_name):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--num-k",
        type=int,
        default=64,
        help="Number of training instances per class",
    )
    
    parser.add_argument(
        "--num-sample",
        type=int,
        default=16,
        help="Number of samples (for inference) in fine-tuning with demonstrations",
    )

    parser.add_argument(
        "--num-demo",
        type=int,
        default=1,
        help="Number of demonstrations from each class",
    )

    # convert everything below to be parser like above
    parser.add_argument(
        "--mapping",  
        type=str,
        default=None,
        help="Label word mapping",
    )

    parser.add_argument(
        "--template",
        type=str,
        default=None,
        help="Template",
    )

    parser.add_argument(
        "--template_path",
        type=str,
        default=None,
        help="Path to a txt file that stores all the templates, one per line. Do not set this when prompt_path is used",
    )

    parser.add_argument(
        "--mapping_path",
        type=str,
        default=None,
        help="Path to a txt file that stores all the label word mappings, one per line. Do not set this when prompt_path is used",
    )

    parser.add_argument(
        "--prompt_path",
        type=str,
        default=None,
        help="Path to a txt file that stores all the prompts (templates and mappings), one per line",
    )

    parser.add_argument(
        "--template_id",
        type=int,
        default=None,
        help="Template id if using template_path",
    )

    parser.add_argument(
        "--mapping_id",
        type=int,
        default=None,
        help="Mapping id if using template_path",
    )

    parser.add_argument(
        "--prompt_id",
        type=int,
        default=None,
        help="Prompt id if using prompt_path",
    )

    parser.add_argument(
        "--top_n_template",
        type=int,
        default=None,
        help="Use top-n template in the template path",
    )

    parser.add_argument(
        "--tag",
        type=str,
        default='',
        help="Set the tag and find the result easier in the log.",
    )

    parser.add_argument(
        "--demo_filter",
        type=bool,
        default=False,
        help="Only use similar instances in demonstrations",
    )

    parser.add_argument(
        "--demo_filter_rate",
        type=float,
        default=0.5,
        help="Only use top-x\% similar instances in demonstrations",
    )

    parser.add_argument(
        "--demo_filter_model",
        type=str,
        default=None,
        help="Model name for demonstration filter embeddings. Will load embeddings based on the model name.",
    )

    parser.add_argument(
        "--debug_mode",
        type=bool,
        default=False,
        help="Debug mode",
    )

    parser.add_argument(
        "--double_demo",
        type=bool,
        default=False,
        help="Use double length for using demonstrations",
    )

    parser.add_argument(
        "--first_sent_limit",
        type=int,
        default=None,
        help="Limit the length of the first sentence (i.e., sent_0)",
    )

    parser.add_argument(
        "--other_sent_limit",
        type=int,
        default=None,
        help="Limit the length of sentences other than the first sentence",
    )

    parser.add_argument(
        "--use_full_length",
        type=bool,
        default=None,
        help="Use the full length (512)",
    )

    parser.add_argument(
        "--template_list",
        type=list,
        default=None,
        help="(DO NOT List of templates (only initialized after the program starts.",
    )

    parsed_args = parser.parse_args()

    parsed_args.prompt = True
    parsed_args.max_seq_len = 256
    parsed_args.task_name = dataset_name.lower()
    parsed_args.data_dir = f"/data/common/lm-bff/k-shot/{dataset_name}/64-0"
    parsed_args.overwrite_cache = False
    parsed_args.truncate_head = True
    parsed_args.auto_demo = True
    parsed_args.max_length_per_example = 128

    if dataset_name == "SST-2":
        parsed_args.template = "*cls**sent_0*_It_was*mask*.*sep+*"
        parsed_args.mapping = "{'0':'terrible','1':'great'}"
    elif dataset_name == "CoLA":
        parsed_args.template = "*cls**sent_0*_This_is*mask*.*sep+*"
        parsed_args.mapping = "{'0':'incorrect','1':'correct'}"
    elif dataset_name == "MRPC":
        parsed_args.template = "*cls**sent_0**mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'0':'No','1':'Yes'}"
    elif dataset_name == "QQP":
        parsed_args.template = "*cls**sent_0**mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'0':'No','1':'Yes'}"
        parsed_args.num_sample = 4
    elif dataset_name == "MNLI":
        parsed_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        parsed_args.first_sent_limit = 110
        parsed_args.max_seq_len = 256
        parsed_args.num_sample = 4
    elif dataset_name == "SNLI":
        parsed_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'contradiction':'No','entailment':'Yes','neutral':'Maybe'}"
        parsed_args.num_sample = 4
        parsed_args.max_seq_len = 256
    elif dataset_name == "QNLI":
        parsed_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'not_entailment':'No','entailment':'Yes'}"
    elif dataset_name == "RTE":
        parsed_args.template = "*cls**sent-_0*?*mask*,*+sentl_1**sep+*"
        parsed_args.mapping = "{'not_entailment':'No','entailment':'Yes'}"
        parsed_args.max_seq_len = 256
        parsed_args.first_sent_limit = 240
    elif dataset_name == "mr":
        parsed_args.template = "*cls**sent_0*_It_was*mask*.*sep+*"
        parsed_args.mapping = "{0:'terrible',1:'great'}"
        parsed_args.double_demo = True
        parsed_args.first_sent_limit = 110
        parsed_args.other_sent_limit = 50
    elif dataset_name == "subj":
        parsed_args.template = "*cls**sent_0*_This_is*mask*.*sep+*"
        parsed_args.mapping = "{0:'subjective',1:'objective'}"
        parsed_args.double_demo = True
        parsed_args.first_sent_limit = 110
        parsed_args.other_sent_limit = 50
    elif dataset_name == "trec":
        parsed_args.template = "*cls**mask*:*+sent_0**sep+*"
        parsed_args.mapping = "{0:'Description',1:'Entity',2:'Expression',3:'Human',4:'Location',5:'Number'}"
        parsed_args.double_demo = True
        parsed_args.first_sent_limit = 110
    elif dataset_name == "cr":
        parsed_args.template = "*cls**sent_0*_It_was*mask*.*sep+*"
        parsed_args.mapping = "{0:'terrible',1:'great'}"
        parsed_args.double_demo = True
        parsed_args.first_sent_limit = 110
        parsed_args.other_sent_limit = 50
    elif dataset_name == "mpqa":
        parsed_args.template = "*cls**sent_0*_It_was*mask*.*sep+*"
        parsed_args.mapping = "{0:'terrible',1:'great'}"
        parsed_args.double_demo = True
        parsed_args.first_sent_limit = 110
    parsed_args.max_seq_length = parsed_args.max_seq_len
    parsed_args.tag = 'exp'
    
    return parsed_args