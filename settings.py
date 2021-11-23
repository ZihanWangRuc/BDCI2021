pretrained_model_name='PRETRAINED_MODEL'#所采用的与训练模型的名字（这个是我们自己训练过的）

train_path='onlycustomer_train_dataset.csv'#训练集地址
candidate_representation_path="dataset_representation_last_2.csv"#候选表示输出地址
test_path="onlycustomer_test_dataset.csv"#测试集地址
testout_path="onlycustomer_test_out.csv"#测试集输出地址

train_data_used=20000#赛题给出的train_data中用来训练的数目（剩下的用来交叉验证）
positive_rate=5#每条QA_data对应正例的比值
negative_rate=4#每条QA_data对应负例的比值
batch_size=64#训练时dataloader的超参数
train_rate=0.9#训练集的占比
epochs=1#训练轮数

candidate_num=20000#候选数

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2, 3'

