# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import torch
import numpy as np

from misc.util import *
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

MAX_NUM = 999999

def compute_score(logit_list, softmax_list, score_wgts, branch_opt, fts=None):
    msp  = softmax_list[branch_opt].max(1)[0]
    mls  = logit_list[branch_opt].max(1)[0]
    if score_wgts[2] != 0:
        ftl = fts.mean(dim = [2,3]).norm(dim = 1, p = 2)
        temp = (score_wgts[0]*msp + score_wgts[1]*mls + score_wgts[2]*ftl)
    else:
        temp = (score_wgts[0]*msp + score_wgts[1]*mls)
    return temp


def evaluation(model, test_loader, out_loader, **options):

    model.eval()
    torch.cuda.empty_cache()
    
    correct = 0
    total = 0
    n = 0
    
    pred_close = []
    pred_open = []
    labels_close = []
    labels_open = []
    score_close = []
    score_open = []
    
    open_labels = torch.zeros(MAX_NUM)
    probs = torch.zeros(MAX_NUM)
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.cuda(), labels.cuda()
            batch_size = labels.size(0)
            with torch.set_grad_enabled(False):
                output_dict = model(data, return_ft=True)
                logits_list = output_dict['logits']
                softmax_list = torch.stack(logits_list)
                softmax_list = torch.softmax(softmax_list / options['lgs_temp'], dim=2)
                if options['score_wgts'][2] != 0:
                    fts = output_dict['fts']
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'], fts=fts)
                else:
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])
                score_close.append(score_temp.data.cpu().numpy())
                for b in range(batch_size):
                    probs[n] = score_temp[b]
                    open_labels[n] = 1
                    n += 1                    
                pred_label = softmax_list[options['branch_opt']].data.max(1)[1]
                total += labels.size(0)
                correct += (pred_label == labels.data).sum()
                pred_close.append(softmax_list[options['branch_opt']].data.cpu().numpy())
                labels_close.append(labels.data.cpu().numpy())

        for batch_idx, (data, labels) in enumerate(out_loader):
            data, labels = data.cuda(), labels.cuda()
            batch_size = labels.size(0)
            ood_label = torch.zeros_like(labels) - 1
            
            with torch.set_grad_enabled(False):
                output_dict = model(data, return_ft=True)
                logits_list = output_dict['logits']
                softmax_list = torch.stack(logits_list)
                softmax_list = torch.softmax(softmax_list / options['lgs_temp'], dim=2)
                if options['score_wgts'][2] != 0:
                    fts = output_dict['fts']
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'], fts=fts)
                else:
                    score_temp = compute_score(logits_list, softmax_list, options['score_wgts'], options['branch_opt'])
                score_open.append(score_temp.data.cpu().numpy())
                for b in range(batch_size):
                    probs[n] = score_temp[b]
                    open_labels[n] = 0
                    n += 1
                pred_open.append(softmax_list[options['branch_opt']].data.cpu().numpy())
                labels_open.append(ood_label.data.cpu().numpy())

    acc = float(correct) * 100. / float(total)

    pred_close = np.concatenate(pred_close, 0)
    pred_open = np.concatenate(pred_open, 0)
    labels_close = np.concatenate(labels_close, 0)
    labels_open = np.concatenate(labels_open, 0)
    score_close = np.concatenate(score_close, 0)
    score_open = np.concatenate(score_open, 0)
    pred1, pred2 = np.argmax(pred_close, axis=1), np.argmax(pred_open, axis=1)
    total_pred_label = np.concatenate([pred1, pred2], axis=0)
    total_label = np.concatenate([labels_close, labels_open], axis=0)
    total_pred = np.concatenate([score_close, score_open], axis=0)

    
    open_labels = open_labels[:n].cpu().numpy()
    prob = probs[:n].reshape(-1, 1)
    fpr, tpr, thresholds  =  roc_curve(open_labels, prob)
    auroc = auc(fpr, tpr)
    
    thresh_idx = np.abs(np.array(tpr) - 0.95).argmin()
    threshold = thresholds[thresh_idx]
    open_pred = (total_pred > threshold).astype(np.float32)
    macro_f1 = f1_score(total_label, ((total_pred_label + 1) * open_pred) - 1, average='macro')
    
    precision, recall, _ = precision_recall_curve(open_labels, prob)
    aupr_in = auc(recall,precision)
    precision, recall, _ = precision_recall_curve(np.bitwise_not((open_labels).astype(bool)), -prob)
    aupr_out = auc(recall,precision)

    print('Accuracy (%): {:.3f}, '  .format(acc), 
          'AUROC: {:.5f}, '         .format(auroc), 
          'AUPR_IN: {:.5f}, '       .format(aupr_in), 
          'AUPR_OUT: {:.5f}, '      .format(aupr_out),
          'Macro F1-score: {:.5f}'  .format(macro_f1), 
        )
    
    return acc, auroc, aupr_in, aupr_out, macro_f1
