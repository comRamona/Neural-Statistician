import torch

from math import log, pi
#import pdb
import torch.nn.functional as F

def gaussian_log_likelihood(x, mean, logvar, clip=True):
    if clip:
        logvar = torch.clamp(logvar, min=-4, max=3)
    a = log(2*pi)
    b = logvar
    c = (x - mean)**2 / torch.exp(logvar)
    return -0.5 * torch.sum(a + b + c)


def bernoulli_log_likelihood(x, p, clip=True, eps=1e-6):
    if clip:
        p = torch.clamp(p, min=eps, max=1 - eps)
    return torch.sum((x * torch.log(p)) + ((1 - x) * torch.log(1 - p)))


def kl_diagnormal_stdnormal(mean, logvar):
    a = mean**2
    b = torch.exp(logvar)
    c = -1
    d = -logvar
    return 0.5 * torch.sum(a + b + c + d)

# Wasserstein metric
def pytorch_wass(m1, C1, m2, C2):
    m1 = m1.reshape(-1)
    m2 = m2.reshape(-1)
    C1 = torch.exp(C1.reshape(-1))
    C2 = torch.exp(C2.reshape(-1))
    dist = (m1 - m2).dot(m1-m2)
    Cr1 = torch.sqrt(C1)
    Cr2 = torch.sqrt(C2)
    Csum = torch.sum(C1+ C2 - 2 * Cr2 * Cr1 * Cr2)
    return 1 - torch.sqrt(dist + Csum)

def kl_diagnormal_diagnormal(q_mean, q_logvar, p_mean, p_logvar):
    # Ensure correct shapes since no numpy broadcasting yet
    p_mean = p_mean.expand_as(q_mean)
    p_logvar = p_logvar.expand_as(q_logvar)
    a = p_logvar
    b = - 1
    c = - q_logvar
    d = ((q_mean - p_mean)**2 + torch.exp(q_logvar)) / torch.exp(p_logvar)
    e = 0.5 * torch.sum(a + b + c + d)
    return e
    
def cosine_sim(q_mean, q_logvar, p_mean, p_logvar):
    return 1 - F.cosine_similarity(q_mean, p_mean)