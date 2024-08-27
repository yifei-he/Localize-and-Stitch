import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from vision.src.vision_datasets.common import maybe_dictionarize
from vision.src.eval import eval_single_dataset
from vision.src.modeling import ImageClassifier
from vision.src.heads import get_classification_head
from vision.src.utils import get_logits
import evaluate 


class Localizer(nn.Module):
    def __init__(self, trainable_params, model, pretrained_model, finetuned_model, dataset_name, args, graft_args, model_type="roberta"):
        super(Localizer, self).__init__()

        self.params = trainable_params
        self.model = model
        self.pretrained_model = pretrained_model
        self.finetuned_model = finetuned_model
        self.args = args
        self.graft_args = graft_args
        self.model_type = model_type

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") 
        self.model.to(self.device)
        if self.model_type == "vit":
            self.classifier_head = get_classification_head(self.args, dataset_name)
            self.classifier_head.to(self.device)
        self.pretrained_model.to("cpu")
        self.finetuned_model.to("cpu")
        self.model.eval()
        self.finetuned_model.eval()
        self.pretrained_model.eval()
        for param in self.pretrained_model.parameters():
            param.requires_grad = False   
        for param in self.finetuned_model.parameters():
            param.requires_grad = False

        self.create_binary_masks()
        self.mask = self.create_basepatch()


    def create_binary_masks(self):
        
        self.trainable_name = []
        self.trainable_parameters = []
       
        for n in self.params: 
            self.trainable_name += [n]
            p = self.params[n]
            self.trainable_parameters += [ torch.rand_like( p.data, device=self.device, requires_grad=False) ] 
        
        self.num_params = sum([p.numel() for p in self.trainable_parameters])  

        self.task_vectors = []
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p

            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: finetensor = fine_p        
                    
            self.task_vectors += [ (finetensor - pretensor).detach() ]    


    def reset_model(self):
        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: pretensor = pre_p.to(self.device)

            with torch.no_grad():   
                for n, p in self.model.named_parameters():    
                    if n == self.trainable_name[counter]: 
                        p += ( pretensor - p )    
    

    def create_basepatch(self):
        threshold = int(self.graft_args.sparsity * self.num_params)  

        abs_tv = []
        for p in self.task_vectors:
            abs_tv.append(torch.abs(p).view(-1))

        abs_tv = torch.cat(abs_tv)
        k = int(self.graft_args.sparsity * abs_tv.numel())  # 1% of the total number of elements

        # Get the k largest values; returns values and their indices
        values, indices = torch.topk(abs_tv.view(-1), k)
        threshold = values.min()

        basepatch = [torch.zeros_like(p, requires_grad=False) for p in self.trainable_parameters]

        for p, q in zip(self.task_vectors, basepatch):
            q[torch.absolute(p) > threshold] = self.graft_args.sigmoid_bias
            q[torch.absolute(p) <= threshold] = -self.graft_args.sigmoid_bias

        print ('Total parameters in my stitch: ', sum([ torch.sum(torch.round(torch.nn.Sigmoid()(p))*torch.round(torch.nn.Sigmoid()(p))) / (1. * self.num_params) for p in basepatch ]))
            
        return basepatch


    def interpolate_model(self, round_, return_mask=False):  
        sigmoid = torch.nn.Sigmoid()

        n_graft_params, n_total_params = 0, 0

        binary_mask = []

        for counter in range(len(self.trainable_name)):
            for pre_n, pre_p in self.pretrained_model.named_parameters():
                if pre_n == self.trainable_name[counter]: 
                    pretensor = pre_p.to(self.device)

            for fine_n, fine_p in self.finetuned_model.named_parameters():
                if fine_n == self.trainable_name[counter]: 
                    finetensor = fine_p.to(self.device)

            with torch.no_grad():            
                for n, p in self.model.named_parameters():  
                    n_total_params += p.numel()
                    if n == self.trainable_name[counter]: 
                        frac = sigmoid(self.mask[counter])
                        if round_:
                            frac = torch.round(frac)
                            binary_mask.append(frac)
                        n_graft_params += torch.sum(frac)
                        frac = frac.to(self.device)
                        p += frac * ( finetensor - pretensor ) 
        
        if round_:
            print ('Proportion in my graft: ', n_graft_params / self.num_params)
        
        if return_mask:
            return binary_mask, n_graft_params / self.num_params
    

    def evaluate_vision(self, dataloader, dataset_name):
        classification_head = get_classification_head(self.args, dataset_name)
        model = ImageClassifier(self.model, classification_head)

        model.eval()

        with torch.no_grad():
            top1, correct, n = 0., 0., 0.
            for i, data in enumerate(tqdm(dataloader)):
                data = maybe_dictionarize(data)
                x = data['images'].to(self.device)
                y = data['labels'].to(self.device)

                logits = get_logits(x, model)

                pred = logits.argmax(dim=1, keepdim=True).to(self.device)

                correct += pred.eq(y.view_as(pred)).sum().item()
                
                n += y.size(0)

            top1 = correct / n

        metrics = {'top1': top1}
        print(f'Grafting on {dataset_name}. Accuracy: {100*top1:.2f}%')
    
        return metrics

    def evaluate_nlp(self, dataloader, task_name, mode='dev'):
        if task_name.lower() not in [ 'qqp', 'mrpc' ]: 
            metric = evaluate.load("accuracy")
        else:
            metric = evaluate.load("f1")
            
        self.model.eval()
        device = self.device
        for batch in dataloader:
            with torch.no_grad():
                loss, outputs = self.model(input_ids=batch['input_ids'].to(device), attention_mask=batch['attention_mask'].to(device), mask_pos=batch["mask_pos"].to(device), labels=batch["labels"].to(device))

            predictions = torch.argmax(outputs, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            
        return metric


    def train_graft(self, dataloader, dataset_name):
        
        loss_fct = torch.nn.CrossEntropyLoss()
        first_batch = 0
        sigmoid = torch.nn.Sigmoid()
        
        device = self.device
        lr = self.graft_args.learning_rate
        
        for epoch in tqdm(range(self.graft_args.num_train_epochs), 'Training the mask'):
            print("Epoch: ", epoch)
            total_grad = []
            
            first_batch = 0
            self.interpolate_model(round_=False)

            # for i, data in enumerate(tqdm(dataloader)):
            for data in dataloader:
                if self.model_type == 'vit':
                    data = maybe_dictionarize(data)
                    x = data['images'].to(self.device)
                    y = data['labels'].to(self.device)
                    features = self.model(x)
                    outputs = self.classifier_head(features)
                    loss = loss_fct(outputs, y)
                elif self.model_type == 'roberta':
                    loss, outputs = self.model(input_ids=data['input_ids'].to(device), attention_mask=data['attention_mask'].to(device), mask_pos=data["mask_pos"].to(device), labels=data["labels"].to(device))
                    
                loss.backward()
                
                for n, p in self.model.named_parameters() :
                    if n in self.trainable_name :
                        if p.grad is None: print (n)

                grad = [p.grad.detach().clone() for n, p in self.model.named_parameters() if n in self.trainable_name]
                self.model.zero_grad()
                grad = [ g * p.to(device) for (g, p) in zip(grad, self.task_vectors) ]

                if first_batch == 0:
                    total_grad = [lr * g for g in grad]
                    first_batch += 1 
                else:
                    total_grad = [ p + lr * g for (p, g) in zip( total_grad, grad ) ]

            total_grad = [ p / (1. * first_batch) for p in total_grad ]    
            self.reset_model()
       
            #Take the gradient step
            with torch.no_grad():
                for p, g in zip(self.mask, total_grad):
                    derivative = sigmoid(p) * (1 - sigmoid(p))
                    reg_term = self.graft_args.l1_strength * torch.where(p>0, derivative, -derivative)
                    p -= g * derivative - reg_term
                    # p -= g * sigmoid(p) * (1 - sigmoid(p)) + self.graft_args.l1_strength * sigmoid(p) * (1 - sigmoid(p))
         
            ######## Evaluation of current mask ###########
            if (epoch+1) % 5 == 0 or epoch == self.graft_args.num_train_epochs - 1:
                mask, proportion = self.interpolate_model(round_=True, return_mask=True)
                if self.model_type == "vit":
                    val = self.evaluate_vision(dataloader, dataset_name)
                elif self.model_type == 'roberta':
                    if dataset_name.lower() not in [ 'qqp', 'mrpc' ]: key = "accuracy"
                    else: key = "f1"
                    val  = self.evaluate_nlp(dataloader, dataset_name, mode='train').compute()[key]
                print("Accuracy for the grafted model:", val)
                self.reset_model()   
        
        mask, proportion = self.interpolate_model(round_=True, return_mask=True)
        self.reset_model()

        if self.graft_args.num_train_epochs == 0: val = -1

        return mask, proportion, val


class Stitcher(nn.Module):
    def __init__(self, trainable_params, model, pretrained_model, finetuned_models, masks):
        super(Stitcher, self).__init__()
        self.params = trainable_params
        self.pretrained_model = pretrained_model
        self.finetuned_models = finetuned_models
        self.model = model

        self.masks = masks
        self.masks = self.get_average_masks()

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    

    def get_average_masks(self):
            
        def reciprocal_with_zero(tensor):
            mask = tensor == 0
            reciprocal = torch.reciprocal(tensor)
            reciprocal = reciprocal.masked_fill(mask, 0)
            return reciprocal

        output_masks = []
        for i in range(len(self.masks)):
            output_mask = self.masks[i].copy()
            # every other mask
            for j in range(len(self.masks)):
                if i == j: continue
                # every layer
                for k in range(len(self.masks[i])):
                    intersect = torch.logical_and(self.masks[i][k], self.masks[j][k])
                    output_mask[k] = output_mask[k] + intersect
            
            for k in range(len(self.masks[i])):
                output_mask[k] = reciprocal_with_zero(output_mask[k])
            output_masks.append(output_mask)

        return output_masks


    def interpolate_models(self):
        trainable_name = list(self.params.keys())

        self.pretrained_model.to(self.device)
        self.model.to(self.device)
        for finetuned_model, mask in zip(self.finetuned_models, self.masks):
            finetuned_model.to(self.device)
            for counter in range(len(trainable_name)):
                for pre_n, pre_p in self.pretrained_model.named_parameters():
                    if pre_n == trainable_name[counter]: 
                        pretensor = pre_p.to(self.device)

                for fine_n, fine_p in finetuned_model.named_parameters():
                    if fine_n == trainable_name[counter]: 
                        finetensor = fine_p.to(self.device)

                with torch.no_grad():            
                    for n, p in self.model.named_parameters():  
                        if n == trainable_name[counter]: 
                            mask[counter].to(self.device)
                            p += mask[counter] * ( finetensor - pretensor ) 
        
        return self.model
                        
