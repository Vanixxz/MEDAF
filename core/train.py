import time
import torch.nn.functional as nn
import torch.nn.functional as F

from misc.util import *


def attnDiv(cams):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    orthogonal_loss = 0
    bs = cams.shape[0]
    num_part = cams.shape[1]
    cams = cams.view(bs, num_part, -1)
    cams = F.normalize(cams, p=2, dim=-1)
    mean = cams.mean(dim=-1).view(bs, num_part, -1).expand(size=[bs, num_part, cams.shape[-1]])
    cams = F.relu(cams-mean)
    
    for i in range(cams.shape[1]):
        for j in range(i+1, cams.shape[1]):
            orthogonal_loss += cos(cams[:,i,:].view(bs,1,-1), cams[:,j,:].view(bs,1,-1)).mean()
    return orthogonal_loss/(i*(i-1)/2)

    
def train(train_loader, model, criterion, optimizer, args):
    model.train()
    
    loss_keys = args['loss_keys']
    acc_keys  = args['acc_keys']
    loss_meter = {p: AverageMeter() for p in loss_keys}
    acc_meter  = {p: AverageMeter() for p in acc_keys}
    time_start = time.time()
    
    for i, data in enumerate(train_loader):
        inputs = data[0].cuda()
        target = data[1].cuda()
        
        output_dict = model(inputs, target)
        logits      = output_dict['logits']
        branch_cams = output_dict['cams']
        loss_values = [criterion['entropy'](logit.float(), target.long()) for logit in logits]
        loss_values.append(attnDiv(branch_cams))
        loss_values.append(args['loss_wgts'][0] * sum(loss_values[:3]) + args['loss_wgts'][1] * loss_values[-2] + args['loss_wgts'][2] * loss_values[-1])

        multi_loss = {loss_keys[k]: loss_values[k] for k in range(len(loss_keys))}
        acc_values = [accuracy(logit, target, topk = (1,))[0] for logit in logits]
        train_accs = {acc_keys[k] : acc_values[k] for k in range(len(acc_keys))}
        update_meter(loss_meter, multi_loss, inputs.size(0))
        update_meter(acc_meter, train_accs, inputs.size(0))

        tmp_str = "< Training Loss >\n"
        for k, v in loss_meter.items(): 
            tmp_str = tmp_str + f"{k}:{v.value:.4f} "
        tmp_str = tmp_str + "\n< Training Accuracy >\n"
        for k, v in acc_meter.items():
            tmp_str = tmp_str + f"{k}:{v.value:.1f} "
        optimizer.zero_grad()
        loss_values[-1].backward()
        optimizer.step()
    
    time_eclapse = time.time() - time_start
    print(tmp_str + f"t:{time_eclapse:.1f}s")
    
    return loss_meter[loss_keys[-1]].value
