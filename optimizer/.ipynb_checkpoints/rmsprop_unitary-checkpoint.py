# -*- coding: utf-8 -*-
import torch
from torch.optim.optimizer import Optimizer
from .utils import step_unitary


class RMSprop_Unitary(Optimizer):
    """Implements RMSprop gradient descent for unitary matrix.
        
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        
    .. note::
        This is the vanilla version of the gradient descent for unitary matrix, 
        i.e. formula (6) in H. D. Tagare. Notes on optimization on Stiefel manifolds. 
        Technical report, Yale University, 2011, and formula (6) in Scott Wisdom, 
        Thomas Powers, John Hershey, Jonathan Le Roux, and Les Atlas. Full-capacity 
        unitary recurrent neural networks. In NIPS 2016. 
        .. math::
                  A = G^H*W - W^H*G \\
                  W_new = (I+lr/2 * A)^(-1)*(I-lr/2 * A)*W

        where W, G and lr denote the parameters, gradient
        and learning rate respectively.
    """
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(RMSprop_Unitary, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RMSprop_Unitary, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('momentum', 0)
            group.setdefault('centered', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RMSprop does not support sparse gradients')
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['square_avg'] = torch.zeros_like(p.data)
                    if group['momentum'] > 0:
                        state['momentum_buffer'] = torch.zeros_like(p.data)
                    if group['centered']:
                        state['grad_avg'] = torch.zeros_like(p.data)
                        
                        
                square_avg = state['square_avg']
                alpha = group['alpha']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                square_avg.mul_(alpha).addcmul_(1 - alpha, grad, grad)
                
                if group['centered']:
                    grad_avg = state['grad_avg']
                    grad_avg.mul_(alpha).add_(1 - alpha, grad)
                    avg = square_avg.addcmul(-1, grad_avg, grad_avg).sqrt().add_(group['eps'])
                else:
                    avg = square_avg.sqrt().add_(group['eps'])

                if group['momentum'] > 0:
                    buf = state['momentum_buffer']
                    buf.mul_(group['momentum']).addcdiv_(grad, avg)
                    grad = buf
#                    p.data.add_(-group['lr'], buf)
                else:
                    grad = torch.zeros_like(p.data).addcdiv(grad, avg)
#                    p.data.addcdiv_(-group['lr'], grad, avg)
                
#                print(state['square_avg'] - square_avg) 
                #print(grad-p.grad.data)
                #print(grad)
                
                lr = group['lr']
                
#                 G_r = grad[:,:,0]
#                 G_i = grad[:,:,1]
#                 W_r = p.data[:,:,0]
#                 W_i = p.data[:,:,1]
                G = grad[:,:,:]
                W = p.data[:,:,:]
                p.data = _step_unitary(G, W,lr)
        return loss

    

def u_matrix(matrix,reg_type="l1"):
    
    r_i=torch.view_as_complex(matrix)
    conj=r_i.transpose(0,1).conj()
    
    a=r_i.real
    b=r_i.imag
    c=conj.real
    d=conj.imag
    
    real = a @ c - b @ d
    imag = a @ d + b @ c
    
    _I=torch.view_as_complex(torch.stack([real,imag],dim = -1))
    
    contr=torch.eye(matrix.size()[0]).cuda()
    res_matix=_I-contr
    
    if reg_type== "l1":
        return res_matix.abs().sum()/res_matix.size()[0]*res_matix.size()[0]
    
    res_r=res_matix.real
    res_i=res_matix.imag
    return  (res_r.pow(2)+res_matix.pow(2)).sum()/res_matix.size()[0]*res_matix.size()[0]


def calComplexmm(R,L):
    """
        calculate the complex ,mm opration
    """
    a=R.real
    b=R.imag
    c=L.real
    d=L.imag
    real = a @ c - b @ d
    imag = a @ d + b @ c
    GHW = torch.view_as_complex(torch.stack([real,imag],dim = -1))
    return GHW

def _step_unitary(G, W, lr):

    # A = G^H W - W^H G
    # A_r = (G_r^T W_r)+ (G_i^T W_i)- (W_r^T G_r) - (W_i^T G_i)
    # A_i = (G_r^T W_i)- (G_i^T W_r)- (W_r^T G_i)+ (W_i^T G_r)
    G_i=torch.view_as_complex(G)
    conj_G=G_i.transpose(0,1).conj()#get conj

    W_i=torch.view_as_complex(W)
    conj_W=W_i.transpose(0,1).conj()#get conj
    
    GHW=calComplexmm(conj_G,W_i)
    WHG=calComplexmm(conj_W,G_i)
    
    A=GHW-WHG
    
    i_=torch.eye(G_i.size()[0]).cuda()

    lr_A_l=i_+(lr/2)*A
    
    ff=lr_A_l.detach().cpu()

    _lr_A_l=ff.inverse().cuda()
    
    lr_A_r=i_-(lr/2)*A

    gain = calComplexmm(_lr_A_l,lr_A_r)
    res = calComplexmm(gain,W_i)
    return torch.stack([res.real,res.imag],dim = -1)    
    
                
    