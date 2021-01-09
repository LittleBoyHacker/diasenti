# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import copy
from utils.evaluation import evaluate
import time
import pickle
from optimizer import RMSprop_Unitary
from models.DialogueRNN import MaskedNLLLoss
import  mylogging 
import utils.reg_unitary as  reg


def train(params, model):
    criterion_emo = get_criterion(params,p_type="emotion")
    criterion_act = get_criterion(params,p_type="act")

    train_type=params.train_type##
    
    
    log_file_name=r"result"
    log_file_path=r"/home/sunsi/android/Quantum/diasenti/eval/"+str(params.train_type)+str(params.network_type)+"1."+"txt"
    log_result=mylogging.log(log_file_name,log_file_path)
    
    
    if hasattr(model,'get_params'):
        unitary_params, remaining_params = model.get_params()
    else:
        remaining_params = model.parameters()
        unitary_params = []
        
#     if len(unitary_params)>0:
#         unitary_optimizer = RMSprop_Unitary(unitary_params,lr = params.unitary_lr)
        
#     print("model parameters:",model.parameters())
#     print("unitary parameters",unitary_params)
#     print("remaining_params parameters",remaining_params)

    optimizer = torch.optim.RMSprop(model.parameters(),lr = params.lr)
    
    # Temp file for storing the best model 
    temp_file_name = str(int(np.random.rand()*int(time.time())))
    params.best_model_file = os.path.join('tmp',temp_file_name)

    best_f1_acc= 0.0
#    best_val_loss = -1.0
    for i in range(params.epochs):
        print('epoch: ', i)
        model.train()
        with tqdm(total = params.train_sample_num) as pbar:
            time.sleep(0.05)            
            for _i,data in enumerate(params.reader.get_data(iterable = True, shuffle = True, split='train'),0):

                b_inputs = [inp.to(params.device) for inp in data[:-2]]
                b_targets_e = data[-2].to(params.device)
                b_targets_a = data[-1].to(params.device)
                # Does not train if batch_size is 1, because batch normalization will crash
                if b_inputs[0].shape[0] == 1:
                    continue
                outputs_e,outputs_a = model(b_inputs)
                b_targets_e, outputs_e, loss_e = get_loss(params, criterion_emo, outputs_e, b_targets_e, b_inputs[-1],p_type="emotion")
                b_targets_a, outputs_a, loss_a = get_loss(params, criterion_act, outputs_a, b_targets_a, b_inputs[-1],p_type="act")
                optimizer.zero_grad()
#                 if len(unitary_params)>0:
#                     unitary_optimizer.zero_grad()
                    
                if train_type=="act":
                    loss=loss_a
                elif train_type=="emotion":
                    loss=loss_e
                elif train_type=="joint":
                    loss=2*loss_a+loss_e
                else:
                    raise Exception("trainning type  not supported: {}")
                    
                if np.isnan(loss.item()):
                    torch.save(model,params.best_model_file)
                    raise Exception('loss value overflow!')
                    
                loss=loss+0.1*reg.reg_unitary(unitary_params)
                
#                 print("unitary_loss:",reg.reg_unitary(unitary_params))

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), params.clip)
                optimizer.step()

                # Compute Training Accuracy                                 
                n_total = len(outputs_e)   
                n_correct_a = (outputs_e.argmax(dim = -1) == b_targets_e).sum().item()
                train_acc_a = n_correct_a/n_total 

                #Update Progress Bar
                pbar.update(params.batch_size)
                ordered_dict={' acc': train_acc_a, ' loss':loss.item()}        
                pbar.set_postfix(ordered_dict=ordered_dict)
        
        model.eval()
        
        #################### Compute Validation Performance##################
#         val_output,val_target, val_mask = get_predictions(model, params, split = 'dev',p_type="emotion")
#         val_target, val_output, val_loss = get_loss(params, criterion,val_output,val_target, val_mask,p_type="emotion")
        if train_type=="act":
            val_output,val_target, val_mask = get_predictions(model, params, split = 'dev',p_type="act")
            val_target, val_output, val_loss = get_loss(params, criterion_act,val_output,val_target, val_mask,p_type="act")
            
            print('validation performance:')
            performances = evaluate(params,val_output,val_target)    
            print('act acc {},f1 {}, act val_loss = {}'.format(performances['acc'], performances['f1'],val_loss))
            f1_acc=performances['acc']+performances['f1']
            log_result.log_info("******epoch:{}****** act dev acc:{} f1:{} ".format(i,performances['acc'],performances['f1']))


        if train_type=="emotion":
            val_output,val_target, val_mask = get_predictions(model, params, split = 'dev',p_type="emotion")
            val_target, val_output, val_loss = get_loss(params, criterion_emo,val_output,val_target, val_mask,p_type="emotion")
            print('emotion validation performance:')
            performances = evaluate(params,val_output,val_target)    
            print('emotion acc {},emotion f1 {}, val_loss = {}'.format(performances['acc'], performances['f1'],val_loss))
            f1_acc=performances['acc']+performances['f1']
            log_result.log_info("******epoch:{}****** emo dev acc:{} f1:{} ".format(i,performances['acc'],performances['f1']))


            
        if train_type=="joint":
            val_output,val_target, val_mask = get_predictions(model, params, split = 'dev',p_type="act")
            val_target, val_output, val_loss = get_loss(params, criterion_act,val_output,val_target, val_mask,p_type="act")
            print('act validation performance:')
            performances = evaluate(params,val_output,val_target)    
            print('act acc {},act f1 {}, val_loss = {}'.format(performances['acc'], performances['f1'],val_loss))
            act_f1_acc=performances['acc']+performances['f1']
            log_result.log_info("******epoch:{}****** act dev acc:{} f1:{} ".format(i,performances['acc'],performances['f1']))

            
            val_output,val_target, val_mask = get_predictions(model, params, split = 'dev',p_type="emotion")
            val_target, val_output, val_loss = get_loss(params, criterion_emo,val_output,val_target, val_mask,p_type="emotion")
            print('emotion validation performance:')
            performances = evaluate(params,val_output,val_target)    
            print('emotion acc {},emotion f1 {}, val_loss = {}'.format(performances['acc'], performances['f1'],val_loss))
            emo_f1_acc=performances['acc']+performances['f1']

            f1_acc=act_f1_acc+emo_f1_acc
            log_result.log_info("******epoch:{}****** emo dev acc:{} f1:{} ".format(i,performances['acc'],performances['f1']))


    
        if params.train_type=="joint":
            performance_dict_emo, performance_dict_act= test(model, params,p_type=params.train_type)

            print("action classification:")
            performance_str_act = print_performance(performance_dict_act, params)
            print("emotion classification:")
            performance_str_emo = print_performance(performance_dict_emo, params)


            log_result.log_info("******epoch:{}****** act test acc:{} f1:{} ".format(i,performance_dict_act["acc"],performance_dict_act["f1"]))
            log_result.log_info("******epoch:{}****** emo test acc:{} f1:{} ".format(i,performance_dict_emo["acc"],performance_dict_emo["f1"]))

        else:
            performance_dict= test(model, params,p_type=params.train_type)
            performance_str_dict = print_performance(performance_dict, params)

            log_result.log_info("******epoch:{}****** {} test acc:{} f1:{} ".format(i,params.train_type,performance_dict["acc"],performance_dict["f1"]))

        
        ##################################################################
        if f1_acc > best_f1_acc:
            print("model_file:",params.best_model_file)
            torch.save(model,params.best_model_file)
            print('The best model up till now. Saved to File.')
            best_f1_acc = f1_acc

            
            
            



    
    
def get_criterion(params,p_type="emotion"):
    # For ICON, CMN, NLLLoss is used
    if params.dialogue_context:      
        criterion = nn.NLLLoss()
    # For BC-LSTM, DialogueRNN and DialogueGCN, MaskedNLLLoss is used
    else:
        if p_type=="emotion":
            criterion = MaskedNLLLoss(params.loss_weights_emotion.to(params.device))
        else:
            criterion = MaskedNLLLoss(params.loss_weights_act.to(params.device))
    return criterion

def get_loss(params, criterion, outputs, b_targets, mask,p_type="emotion"):
    
    # b_targets: (batch_size, dialogue_length, output_dim)
    # outputs: (batch_size, dialogue_length, output_dim)    
    if p_type=="emotion":
        b_targets = b_targets.reshape(-1, params.output_dim_emo).argmax(dim=-1)
        outputs = outputs.reshape(-1, params.output_dim_emo)
    if p_type=="act":
        b_targets = b_targets.reshape(-1, params.output_dim_act).argmax(dim=-1)
        outputs = outputs.reshape(-1, params.output_dim_act)
    
    if params.dialogue_context:
        loss = criterion(outputs,b_targets)
    else:
        loss = criterion(outputs,b_targets,mask)
        nonzero_idx = mask.view(-1).nonzero()[:,0]
        outputs = outputs[nonzero_idx]
        b_targets = b_targets[nonzero_idx]
            
    return b_targets, outputs, loss



def test(model,params,p_type="emotion"):
    model.eval()
    if p_type=="joint":
        test_output_act,test_target_act, test_mask_act = get_predictions(model, params, split = 'test',p_type="act")
        test_target_act = torch.argmax(test_target_act.reshape(-1, params.output_dim_act),-1)
        test_output_act = test_output_act.reshape(-1, params.output_dim_act)
        if not params.dialogue_context:
            nonzero_idx = test_mask_act.view(-1).nonzero()[:,0]
            test_output_act = test_output_act[nonzero_idx]
            test_target_act = test_target_act[nonzero_idx]
        performances_act = evaluate(params,test_output_act,test_target_act)


        test_output_emo,test_target_emo, test_mask_emo = get_predictions(model, params, split = 'test',p_type="emotion")
        test_target_emo = torch.argmax(test_target_emo.reshape(-1, params.output_dim_emo),-1)
        test_output_emo = test_output_emo.reshape(-1, params.output_dim_emo)
        if not params.dialogue_context:
            nonzero_idx = test_mask_emo.view(-1).nonzero()[:,0]
            test_output_emo = test_output_emo[nonzero_idx]
            test_target_emo = test_target_emo[nonzero_idx]
        performances_emo = evaluate(params,test_output_emo,test_target_emo)
        
        return performances_emo,performances_act
    else:

        test_output,test_target, test_mask = get_predictions(model, params, split = 'test',p_type=p_type) 
        if p_type=="act":
            test_target = torch.argmax(test_target.reshape(-1, params.output_dim_act),-1)
            test_output = test_output.reshape(-1, params.output_dim_act)
        else:
            test_target = torch.argmax(test_target.reshape(-1, params.output_dim_emo),-1)
            test_output = test_output.reshape(-1, params.output_dim_emo)
        if not params.dialogue_context:
            nonzero_idx = test_mask.view(-1).nonzero()[:,0]
            test_output = test_output[nonzero_idx]
            test_target = test_target[nonzero_idx]
        performances = evaluate(params,test_output,test_target)
        return performances

def print_performance(performance_dict, params):
    performance_str = ''
    for key, value in performance_dict.items():
        performance_str = performance_str+ '{} = {} '.format(key,value)
    print(performance_str)
    return performance_str

def get_predictions(model, params, split ='dev',p_type="emotion"):
    outputs = []
    targets = []
    masks = []
    iterator = params.reader.get_data(iterable =True, shuffle = False, split = split)
        
    for _ii,data in enumerate(iterator,0):  
        data_x = [inp.to(params.device) for inp in data[:-2]]
        if p_type == "emotion":
            data_t = data[-2].to(params.device)
        else:
            data_t= data[-1].to(params.device)
        pred_emo,pred_act = model(data_x)
        
        data_o=pred_emo if p_type=="emotion" else pred_act

        if not params.dialogue_context:
            masks.append(data_x[-1])
                        
        outputs.append(data_o.detach())
        targets.append(data_t.detach())
            
    outputs = torch.cat(outputs)
    targets = torch.cat(targets)
    if not params.dialogue_context:   
        masks = torch.cat(masks)
        
    return outputs, targets, masks




def save_model(model,params,s):
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    params.dir_name = str(round(time.time()))
    dir_path = os.path.join('tmp',params.dir_name)
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    torch.save(model.state_dict(),os.path.join(dir_path,'model'))
#    copyfile(params.config_file, os.path.join(dir_path,'config.ini'))
    params.export_to_config(os.path.join(dir_path,'config.ini'))
    params_2 = copy.deepcopy(params)
    if 'lookup_table' in params_2.__dict__:
        del params_2.lookup_table
    if 'sentiment_dic' in params_2.__dict__:
        del params_2.sentiment_dic
    del params_2.reader
    pickle.dump(params_2, open(os.path.join(dir_path,'config.pkl'),'wb'))
    
    del params_2
    if 'save_phases' in params.__dict__ and params.save_phases:
        print('Saving Phases.')
        phase_dict = model.get_phases()
        for key in phase_dict:
            file_path = os.path.join(dir_path,'{}_phases.pkl'.format(key))
            pickle.dump(phase_dict[key],open(file_path,'wb'))
    eval_path = os.path.join(dir_path,'eval')
    with open(eval_path,'w') as f:
        f.write(s)
    
def save_performance(params, performance_dict,task="emotion"):
    df = pd.DataFrame()
    output_dic = {'dataset' : params.dataset_name,
                    'modality' : params.features,
                    'network' : params.network_type,
                    'model_dir_name': params.dir_name}
    output_dic.update(performance_dict)
    df = df.append(output_dic, ignore_index = True)

    params.output_file = 'eval/{}_{}_{}.csv'.format(params.dataset_name, params.network_type,task)
    df.to_csv(params.output_file, encoding='utf-8', index=True)
