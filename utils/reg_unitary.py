import torch
import torch.nn as nn

def reg_unitary(param):
    reg_loss=[]
    for p in param:
        ll=u_matrix(p)
        reg_loss.append(ll)
    reg = torch.stack(reg_loss,dim=0)# cat all dialogue's loss
    #cost = torch.cat(reg,dim=0)# cat all dialogue's loss
    cost = torch.mean(reg)
    #reg = torch.sum(torch.tensor(reg_loss,requires_grad=True))/len(param)
    return cost
        
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
    
    ###l2 ，这个有问题，没调试
    res_r=res_matix.real
    res_i=res_matix.imag
    l_res=  (res_r.pow(2)+res_matix.pow(2)).sum()/res_matix.size()[0]*res_matix.size()[0]    
    return l_res