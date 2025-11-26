import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
from sklearn.mixture import GaussianMixture
import numpy as np
 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_bgm, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)

    loss_bgm = (label_hat*loss_bgm).sum()
    loss_tse = (label_hat*loss_tse).sum()
    
    if loss_type in ['TAL','TRL']:
        return loss_bgm, loss_tse
    else:
        return loss_bgm/label_hat.sum(), loss_tse/label_hat.sum() # mean

def SCA(image_features, text_features, logit_scale_cm=10.0, logit_scale_im=5.0, gamma=1, device='cuda'):
    # normalize features
    image_norm = F.normalize(image_features, dim=-1)
    text_norm = F.normalize(text_features, dim=-1)
    sim_img_txt = logit_scale_cm * (image_norm @ text_norm.T)

    # intra-modal: use mean-pooling (already global features here)
    intra_image = image_features
    intra_text = text_features

    # Step 1: Cross-modal label y_cm
    sim_diag = torch.diag(sim_img_txt)
    sim_img_sum = torch.exp(sim_img_txt).sum(dim=1)
    sim_txt_sum = torch.exp(sim_img_txt.T).sum(dim=1)
    p_img = torch.exp(sim_diag) / sim_img_sum
    p_txt = torch.exp(sim_diag) / sim_txt_sum
    y_cm = 0.5 * (p_img + p_txt)

    # Step 2: Intra-modal label y_im
    cosine_sim = F.cosine_similarity(intra_image, intra_text, dim=1)
    cosine_np = cosine_sim.detach().cpu().numpy().reshape(-1, 1)
    # gmm = GaussianMixture(n_components=2, max_iter=20, random_state=0)
    gmm = GaussianMixture(n_components=2, max_iter=100,tol=1e-3, n_init=5,random_state=0)
    gmm.fit(cosine_np)
    probs = gmm.predict_proba(cosine_np)
    clean_idx = gmm.means_.argmax()
    y_im = torch.tensor(probs[:, clean_idx], dtype=torch.float32, device=device)

    # Step 3: Final soft label
    y_soft = torch.minimum(y_cm, y_im)

    # Step 4: Cross-modal loss
    loss_i = F.cross_entropy(sim_img_txt, torch.arange(sim_img_txt.shape[0], device=device), reduction='none')
    loss_t = F.cross_entropy(sim_img_txt.T, torch.arange(sim_img_txt.shape[0], device=device), reduction='none')
    loss_cm = 0.5 * (loss_i + loss_t) * y_soft
    loss_cm = loss_cm.mean()

    # Step 5: Intra-modal loss
    intra_image_norm = F.normalize(intra_image, dim=-1)
    intra_text_norm = F.normalize(intra_text, dim=-1)
    logits_topo = logit_scale_im * (intra_image_norm @ intra_text_norm.T)
    pos_topo = torch.exp(torch.diag(logits_topo))
    neg_topo = torch.exp(logits_topo).sum(dim=1)
    loss_im = -torch.log(pos_topo / (neg_topo + 1e-8)) * y_soft
    loss_im = loss_im.mean()

    return (loss_cm + gamma * loss_im)

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    elif 'SCA' in loss_type:
        per_loss = SCA(image_features, text_features, logit_scale_cm=logit_scale, logit_scale_im=logit_scale, gamma=1.0)
    else:
        exit()

    return per_loss, scores.diag()



