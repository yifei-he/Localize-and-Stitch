import os
import torch
import numpy as np
from pathlib import Path
import time
import sys
from eval import eval_single_dataset
from args import parse_arguments
from vision_datasets.registry import get_dataset
import pickle
from vision_datasets.registry import split_train_into_train_val, create_k_shot_dataset

import sys
import os
sys.path.append(os.path.abspath('../../'))
from localize_utils import *

# TODO: change to your checkpoint folders
root = '/data/common/task-arithmetic'
sys.path.append(root)

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

args = parse_arguments()

model_name = 'ViT-B-16'

train_mask = True
args.data_location = root + '/data'
args.model_name = model_name
args.save = root + '/task_vectors_checkpoints/' + model_name
args.log = False
args.save_mask = False
n_shot = 64
pretrained_checkpoint = root+'/task_vectors_checkpoints/'+model_name+'/zeroshot.pt'

str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

graft_args = parse_arguments()
graft_args.checkpoint_location = root+'/ckpt'
graft_args.sigmoid_bias = 5
args.batch_size = 128
graft_args.l1_strength = 1
graft_args.learning_rate = 1e7
graft_args.num_train_epochs = 10
graft_args.sparsity = 1e-5

folder_name = model_name+'/'+str(graft_args.sparsity)+'_'+str(graft_args.sigmoid_bias)+'_'+str(graft_args.l1_strength)+'_'+str(graft_args.num_train_epochs)


mask_folder = root+f'/masks/{n_shot}shot/'+folder_name+'/'
args.logs_path = f'../logs/{n_shot}shot/'+folder_name

if args.log:
    log = create_log_dir(args.logs_path, 'log_{}_localize_stitch.txt'.format(str_time_))

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
    graft_args.sparsity_level = graft_args.sparsity
    finetuned_checkpoint = root+'/task_vectors_checkpoints/'+model_name+'/'+dataset_name+'/finetuned.pt'
    try:
        finetuned_model = torch.load(finetuned_checkpoint)
    except:
        finetuned_model = pickle.load(open(finetuned_checkpoint, 'rb'))

    if train_mask:
        base_dataset = get_dataset(dataset_name, final_model.val_preprocess, location=args.data_location, batch_size=args.batch_size)

        valset = create_k_shot_dataset(base_dataset, num_shots=n_shot)
        print("Total samples used for mask computation: ", len(valset))
        
        val_dataloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        graft_args.gradient_accumulation_steps = len(valset) // args.batch_size + 1

        localizer = Localizer(trainable_params, final_model, pretrained_model, finetuned_model, dataset_name, args, graft_args, model_type='vit')
        mask, proportion, test = localizer.train_graft(val_dataloader, dataset_name)
        if args.save_mask:
            Path(mask_folder).mkdir(parents=True, exist_ok=True)  
            torch.save(mask, mask_folder+dataset_name+'_mask.pt')
    else:
        localizer = Localizer(trainable_params, final_model, pretrained_model, finetuned_model, dataset_name, args, graft_args, model_type='vit')
        mask = torch.load(mask_folder+dataset_name+'_mask.pt')
        localizer.mask = mask
        _, proportion = localizer.interpolate_model(return_mask=True)
        test = eval_single_dataset(localizer.model, dataset_name, args)["top1"]
    
    masks.append(mask)
    finetuned_models.append(finetuned_model)
    proportions.append(proportion.cpu().item())
    tests.append(test)

localize_time = time.time() - start_time
final_model = torch.load(pretrained_checkpoint)
stitcher = Stitcher(trainable_params, final_model, pretrained_model, finetuned_models, masks)
image_encoder = stitcher.interpolate_models()
stitch_time = time.time() - start_time - localize_time

accs = []
for i in range(len(exam_datasets)):
    dataset = exam_datasets[i]
    metrics = eval_single_dataset(image_encoder, dataset, args)
    accs.append(metrics.get('top1'))
    if args.log:
        log.info(str(dataset)+','+str(tests[i])+','+str(proportions[i])+','+str(metrics.get('top1')))
if args.log:
    log.info('Avg'+','+str(np.mean(tests))+','+str(np.mean(proportions))+','+str(np.mean(accs)))
    log.info('sparsity_level'+','+str(graft_args.sparsity_level))
    log.info('Localize time: '+str(localize_time))
    log.info('Stitch time: '+str(stitch_time))
