import numpy as np
from numpy import random
import csv
from numba import jit
import pandas as pd
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation
from tqdm import trange,tqdm
import jieba
from settings import *
from rouge import Rouge
rouge=Rouge()

#########################################################################################################
"""
rouge计算相关函数
"""

def compute_rouge(source, target, unit='word'):
    """计算rouge-1、rouge-2、rouge-l
    """
    if unit == 'word':
        source = jieba.cut(source, HMM=False)
        target = jieba.cut(target, HMM=False)
    source, target = ' '.join(source), ' '.join(target)
    try:
        scores = rouge.get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }

def compute_metrics(source, target, unit='word'):
    """计算所有metrics
    """
    metrics = compute_rouge(source, target, unit)
    metrics['main'] = (
        metrics['rouge-1'] * 0.2 + metrics['rouge-2'] * 0.4 +
        metrics['rouge-l'] * 0.4
    )
    return metrics

#########################################################################################################
"""
主要函数是QUERY()，其作用为
根据query(原文)返回检索式摘要
"""

@jit
def encoding(sentence,model):
    return model.encode(sentence)

@jit
def cos(a,b):
    return a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))


@jit
def QUERY(query,datas,model):  
    """

    """
    ret=""
    query_embedding = encoding(str(query),model).astype(np.double)
    #datas[i]["_sim"]代表了query和answer的关系
    maxindex=0
    maxsimilarity=-1
    for i in datas:
        passage_embedding = datas[i]["embedding"]
        similarity=cos(query_embedding, passage_embedding)
        if similarity>maxsimilarity:
            maxindex=i
            maxsimilarity=similarity
    ret=datas[maxindex]['answer'] 
    return ret

#########################################################################################################
print("""读取问答数据并形成main_datas,数据结构为{id:{"query":..,"answer":...,"embeddings":...,},}""")
"""
这一段只用了20000个数据，剩下的是用来交叉验证看效果的。
onlycustomer的具体生成方法看"process_of_onlycustomer.py"，
主要是把常见无意义词去掉了，
且仅保留了customer的部分内容，service的都去掉了，这样减小了embedding的复杂度。
这是因为摘要主要集中在客户的反映上，当然有一些也有service的反馈，
不过需求上速度>精度。
"""
main_datas={}
with open(train_path,'r+',encoding='utf-8') as file:
    A=file.readline()
    A=file.readline()
    for i in range(train_data_used):
        A=A.split('|')
        id=int(A[0])
        main_datas.update({id:{"query":A[1],"answer":A[2],"tag":0}})
        A=file.readline()
     
#########################################################################################################
print("""预训练对话-摘要相关性匹配模型""")

model = SentenceTransformer(pretrained_model_name)
#device_ids = [0, 1]
#model = torch.nn.DataParallel(model, device_ids=device_ids)

"""
上面这个可以用SentenceTransformer的各种预训练库，一开始用的是paraphrase-multilingual-MiniLM-L12-v2
不知道为什么DataParallel用不了（多个gpu同时工作的一个指令），之后可能会改一改

下面主要是生成训练集、测试集的部分
思路是每个query对应一个answer的正例，n个其他query的answer的负例。
正例的标签是1，负例的标签是两个query对应answer的rouge值
为了防止正负例不够均匀，设置了positive_rate/negative_rate两个参数，可以自己调整
预训练的超参数主要有warmup_steps,epochs,evaluator之类的，也可以自己调整
"""

 
train_examples=[]
for i in tqdm(list(main_datas.keys())[:train_data_used]):
    for j in range(positive_rate):
        train_examples.append(InputExample(texts=[main_datas[i]["query"],main_datas[i]["answer"]],label=1.0))   
    for j in range(negative_rate):
        false_index=random.randint(len(main_datas))
        while false_index==i:
            false_index=random.randint(len(main_datas))
        txtA=main_datas[i]
        txtB=main_datas[false_index]
        sim=compute_metrics(txtA['answer'], txtB['answer'], unit='word')['main']
        train_examples.append(InputExample(texts=[txtA['query'],txtB['answer']],label=sim))

#生成训练集
shuffle_num=int(len(train_examples)*train_rate)
shuffled_train_examples=[]
shufflelist=random.permutation(len(train_examples))
for i in range(len(train_examples)):
    shuffled_train_examples.append(train_examples[shufflelist[i]])
train_dataloader = DataLoader(shuffled_train_examples[:shuffle_num], shuffle=True, batch_size=batch_size)
train_loss = losses.CosineSimilarityLoss(model)
#生成测试集
sentences1 = []
sentences2 = []
scores = []
for pair in shuffled_train_examples[shuffle_num:]:
    sentences1.append(pair.texts[0])
    sentences2.append(pair.texts[1])
    scores.append(pair.label)
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores,write_csv=True)
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=epochs, evaluator=evaluator)

#########################################################################################################
"""
测试模型预测query和answer匹配程度的能力
"""
print("测试模型的相关性预测性能：")
relation_score=model.evaluate(evaluator=evaluator)
print("相关性预测得分为：",relation_score)



#########################################################################################################
"""
把候选的embedding写入csv文件，
方便每次为query搜索answer的时候都可以直接使用候选已经给出的embedding，而不用重新计算
"""

with open(candidate_representation_path,"w+",encoding="utf-8",newline="") as file:
    write=csv.writer(file)
    write.writerow(["id"]+list(range(384)))
    for i in list(main_datas.keys())[:candidate_num]:
        embedding=encoding(main_datas[i]["query"],model)
        write.writerow([i]+list(embedding))

df=pd.read_csv(candidate_representation_path,header=0,encoding="utf-8")
for i in main_datas.keys()[:candidate_num]:
    main_datas[i].update({"embedding":df[df["id"]==i].values[0][1:].astype(np.double)})


#########################################################################################################
"""
作出预测并写入文件
"""

with open(test_path,"r+",encoding='utf-8') as oldfile,\
open(testout_path,"w+",encoding='utf-8') as newfile:
    question=oldfile.readline()
    question=oldfile.readline()
    newfile.write('id|ret\n')
    while question!='':
        question=question.split('|')
        answer=QUERY(question[1],main_datas,model)
        newfile.write(question[0]+'|'+answer)
        question=oldfile.readline()
    

