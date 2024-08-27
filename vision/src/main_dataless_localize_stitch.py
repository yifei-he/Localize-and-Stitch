import os
import torch

import numpy as np
from pathlib import Path

import sys
import os
sys.path.append(os.path.abspath('../../'))
from localize_utils import *

import time
import sys
# TODO: change to your checkpoint folders
root = '/data/common/task-arithmetic'
sys.path.append(root)

from eval import eval_single_dataset
from args import parse_arguments
import pickle

def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD'] 

model = 'ViT-B-16' 
args = parse_arguments()
args.data_location = root + '/data'
args.model = model
args.save = root + '/task_vectors_checkpoints/' + model
args.log = False
pretrained_checkpoint = root+'/task_vectors_checkpoints/'+model+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

graft_args = parse_arguments()
graft_args.checkpoint_location = root+'/ckpt'
graft_args.sparsity_level = 0.1
graft_args.sigmoid_bias = 5
args.logs_path = '../logs/'+model+'/'

if args.log:
    log = create_log_dir(args.logs_path, 'log_dataless_localize_stitch_{}.txt'.format(str_time_))

# start training masks
final_model = torch.load(pretrained_checkpoint)
pretrained_model = torch.load(pretrained_checkpoint)
pretrained_model_dic = pretrained_model.state_dict()

trainable_params = {}
frozen = ["model.positional_embedding", "model.text_projection", "model.logit_scale", "model.token_embedding.weight", "model.ln_final.weight", "model.ln_final.bias"]
for k, v in pretrained_model_dic.items():
    if k not in frozen:
        trainable_params[k] = v

start_time = time.time()
masks, finetuned_models, proportions, tests = [], [], [], []
for dataset_name in exam_datasets:
    finetuned_checkpoint = root+'/task_vectors_checkpoints/'+model+'/'+dataset_name+'/finetuned.pt'
    try:
        finetuned_model = torch.load(finetuned_checkpoint)
    except:
        finetuned_model = pickle.load(open(finetuned_checkpoint, 'rb'))

    localizer = Localizer(trainable_params, final_model, pretrained_model, finetuned_model, dataset_name, args, graft_args, model_type='vit')
    mask, proportion = localizer.interpolate_model(round_=True, return_mask=True)
    test = eval_single_dataset(localizer.model, dataset_name, args)["top1"]
            
    masks.append(mask)
    finetuned_models.append(finetuned_model)
    proportions.append(proportion.cpu().item())
    tests.append(test)

localize_time = time.time() - start_time
print(localize_time)
model = torch.load(pretrained_checkpoint)
stitcher = Stitcher(trainable_params, model, pretrained_model, finetuned_models, masks)
image_encoder = stitcher.interpolate_models()
stitch_time = time.time() - start_time - localize_time

print(stitch_time)

accs = []
for i in range(len(exam_datasets)):
    dataset = exam_datasets[i]
    metrics = eval_single_dataset(image_encoder, dataset, args)
    accs.append(metrics.get('top1'))
    if args.log:
        log.info(str(dataset)+','+str(tests[i])+','+str(proportions[i])+','+str(metrics.get('top1')))
if args.log:
    log.info('Avg'+','+str(np.mean(tests))+','+str(np.mean(proportions))+','+str(np.mean(accs)))
    log.info('Localize time: '+str(localize_time))
    log.info('Stitch time: '+str(stitch_time))
