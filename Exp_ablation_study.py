# 用于跑 自组织秩序的对比实验

# name = "full-model"
# name = "no-rec"
# name = "no-tran"
# name = "rec-no-noise"
# name = "tran-no-noise"
# name = "pre-noise"
# name = "no-noise"

import argparse
import os
import re
import time
from itertools import chain
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
from tqdm import tqdm
import pickle
from datetime import datetime 
import random

from BatchIter import BatchIter
from DataPair import DataPair
from AutoEvaluationF import *
from Model_Lstm2 import TransLstm2 as Trans

parser = argparse.ArgumentParser(description='DGST')
parser.add_argument('-m', '--max-text-length', type=int, default=None,
                    help='Define the maximum length of the text. If the text length exceeds the maximum length, the text is truncated.')
parser.add_argument('-pg', '--show-progress',  action='store_true',
                    help='Show the progress bar during training.')
args = parser.parse_args()

print("== Settings ==")
if args.max_text_length is None:
    print("-- Do not set the maximum text length")
else:
    print("-- Maximum text length:", args.max_text_length)
print("-- Show progress bar:", args.show_progress)

time.sleep(1)
versions = ['full-model','no-rec','no-tran','rec-no-noise','tran-no-noise','pre-noise']
print("Please specify the type of the ablation experiment:")
print("For details please see the paper: https://www.aclweb.org/anthology/2020.emnlp-main.578/")
for i,v in enumerate(versions):
    print(f'  {i}. {v}')
args.type = versions[int(input("Please input the id of the type: "))]
print('Ablation type:', args.type)


model_name = "Ablation_Study-"+args.type
dataset_base = "Dataset/"
corpus_name = "Yelp"
model_save = './model_save/'
output_dir = './outputs/'
batch_size = 256
lr = 1e-3
embedding_size = 256

print("\ninit / load fasttext ....")
init_fasttext(dataset_base, corpus_name, model_save)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("loading datasets ....")


train_data = DataPair(f"{dataset_base}{corpus_name}/train.0",f"{dataset_base}{corpus_name}/train.1", device=device)
test_data = DataPair(f"{dataset_base}{corpus_name}/test.0",f"{dataset_base}{corpus_name}/test.1", base_corpus=train_data, device=device)
gt_data = DataPair(f"{dataset_base}{corpus_name}/reference.0",f"{dataset_base}{corpus_name}/reference.1", base_corpus=train_data, device=device)
print(len(train_data))

vocabulary_size = train_data.vocab_size
trainloader = BatchIter(train_data, batch_size, shuffle=True, batch_first=True, force_length=16)
testloader = BatchIter(test_data, batch_size, shuffle=False, batch_first=True, force_length=16)
gtloader = BatchIter(gt_data, batch_size, shuffle=False, batch_first=True, force_length=30)


print("===========", "model name:", model_name, "==========")


modelOptType = optim.Adam

best_score = 0
T_to1 = Trans(vocabulary_size,embedding_size).to(device)
T_to0 = Trans(vocabulary_size,embedding_size).to(device)
optimizer = modelOptType(list(T_to1.parameters())+list(T_to0.parameters()), lr=lr, weight_decay=1e-5)


def random_replace(data, p):
    # data=data.clone()
    shape = data.shape
    for _ in range(int(shape[0]*shape[1]*p)):
        data[random.randint(0,shape[0]-1),random.randint(0,shape[1]-1)] = random.randint(3,vocabulary_size-1)
    return data

p1 = 0.3
p2 = 0.3

def conbine_fix(data, p=0.4):
    data1 = data.clone()
    data1 = random_replace(data1, p)
    return data1

def train(ep):
    a=datetime.now() 
    T_to1.train()
    T_to0.train()
    L_total_t, L_total_c, total = 0.0, 0.0, 0.0
    Dr,Df = [],[]
    if args.show_progress:
        totaL_ct = len(trainloader)
        dataprovider = tqdm(trainloader, total=totaL_ct,bar_format='{desc}{percentage:3.0f}%|{bar:30}{r_bar}')
    else:
        dataprovider = trainloader

    for i, (data0, data1) in enumerate(dataprovider):
        b_size = data0.shape[0]
        data0 = data0.to(device)
        data1 = data1.to(device)
        
        data0_ = conbine_fix(data0, p1)
        data1_ = conbine_fix(data1, p1)

        optimizer.zero_grad()
        if args.type == "pre-noise":
            prod_0to1 = T_to1(data0_)
        else:
            prod_0to1 = T_to1(data0)
        if args.type == "tran-no-noise" or args.type == "pre-noise" or args.type == "no-noise":
            prod_0to1 = T_to1.argmax(prod_0to1)
        else:
            prod_0to1 = conbine_fix(T_to1.argmax(prod_0to1), p2)
        prod_0to1to0 = T_to0(prod_0to1)

        if args.type == "pre-noise":
            prod_1to0 = T_to0(data1_)
        else:
            prod_1to0 = T_to0(data1)
        if args.type == "tran-no-noise" or args.type == "pre-noise" or args.type == "no-noise":
            prod_1to0 = T_to0.argmax(prod_1to0)
        else:
            prod_1to0 = conbine_fix(T_to0.argmax(prod_1to0), p2)
        prod_1to0to1 = T_to1(prod_1to0)

        L_t = T_to0.loss(data0, prod_0to1to0) + T_to1.loss(data1, prod_1to0to1)

        if args.type == "rec-no-noise"  or args.type == "no-noise":
            prod_0to0 = T_to0(data0)
            prod_1to1 = T_to1(data1)
        else:
            prod_0to0 = T_to0(data0_)
            prod_1to1 = T_to1(data1_)
        L_c = T_to0.loss(data0, prod_0to0) + T_to1.loss(data1, prod_1to1)


        if args.type == "no-rec":
            L_t.backward()
        elif args.type == "no-tran":
            L_c.backward()
        else:
            (L_t + L_c).backward()

        torch.nn.utils.clip_grad_norm_(list(T_to1.parameters())+list(T_to0.parameters()),1)
        optimizer.step()

        L_total_t += L_t.item()
        L_total_c += L_c.item()

        total+=b_size
        if args.show_progress:
            dataprovider.set_description(f"=== ep: {ep} === Loss(t): {L_total_t/total:.2f}, Loss(c): {L_total_c/total:.2f}")
    if not args.show_progress:
        print(f"=== ep: {ep} === Loss(t): {L_total_t/total:.2f}, Loss(c): {L_total_c/total:.2f} | time cost: {(datetime.now()-a).seconds}")

def clean_text(txt):
    txt = re.sub(r"<pad>|<sos>|<eos>","",txt)
    txt = re.sub(r"\s+"," ",txt)
    return txt.strip()

def test(ep):
    input_1 = []
    input_0 = []
    outputs_1to0 = []
    outputs_0to1 = []
    ref_1 = []
    ref_0 = []
    T_to0.eval()
    T_to1.eval()
    with torch.no_grad():
        for data0, data1 in testloader:
            b_size = data0.shape[0]
            data0 = data0.to(device)
            data1 = data1.to(device)

            prod_0to1 = T_to1.argmax(T_to1(data0)).tolist()
            prod_1to0 = T_to1.argmax(T_to0(data1)).tolist()

            data0 = data0.tolist()
            data1 = data1.tolist()
            for i in range(b_size):
                input_1.append(clean_text(train_data.totext(data1[i])))
                input_0.append(clean_text(train_data.totext(data0[i])))
                outputs_1to0.append(clean_text(train_data.totext(prod_1to0[i])))
                outputs_0to1.append(clean_text(train_data.totext(prod_0to1[i])))
        for ref0, ref1 in gtloader:
            ref0 = ref0.tolist()
            ref1 = ref1.tolist()
            for i in range(len(ref0)):
                ref_1.append(clean_text(train_data.totext(ref1[i])))
                ref_0.append(clean_text(train_data.totext(ref0[i])))

    gb1,bleu41,cbleu41,transfer1 = evaluate(input_1, outputs_1to0, ref_1, 1)
    gb2,bleu42,cbleu42,transfer2 = evaluate(input_0, outputs_0to1, ref_0, 0)

    with open(f"{output_dir}{model_name}-stest.txt", "w") as f:
        texts = []
        for (in0, in1), (o01, o10) in zip(zip(input_0,input_1),zip(outputs_0to1,outputs_1to0)):
            texts.append(" in - neg: "+ in0)
            texts.append("out - pos: "+ o01)
            texts.append("")
            texts.append(" in - pos: "+ in1)
            texts.append("out - neg: "+ o10)
            texts.append("")
        f.write(f"=== ep: {ep} ===\n\n"+"\n".join(texts))

    # with open(f"{output_dir}{model_name}-eval.txt", "w") as f:
    #     texts = []
    #     for ((in0, in1), (o01, o10)), (r1,r0) in zip(zip(zip(input_0,input_1),zip(outputs_0to1,outputs_1to0)), zip(ref_1,ref_0)):
    #         texts.append(f"label-input-output-reference|neg->pos|{in0}|{o01}|{r0}")
    #         texts.append(f"label-input-output-reference|pos->net|{in1}|{o10}|{r1}")
    #     f.write("\n".join(texts))
    
    with open(f"{output_dir}{model_name}-numbers.txt", "a") as f:
        if ep==1:
            f.write("\n\n ========== New Start: "+model_name+" ==========\n")
            f.write("EP\tG.Blue\tS.Blue\tAcc\tp1\tp2\n")
        if ep==0:
            f.write("---------------------------\n")
        if ep>0:
            nums = [ep, (gb1+gb2)/2, (bleu41+bleu42)/2, (transfer1+transfer2)/2, p1, p2]
            nums = [str(i) for i in nums]
            f.write("\t".join(nums)+"\n")
            
    print(f"----------------------------------------------------------------")
    return (transfer1+transfer2)/2


print("starting training")

test(0)

for ep in range(1,100):
    train(ep)
    score = test(ep)
    if score >= best_score and ep>5:
        best_score = score
        torch.save({
            "T_to1": T_to1.state_dict(),
            "T_to0": T_to0.state_dict(),
        }, f'{model_save}/{model_name}.mod.best.tch')

    
    