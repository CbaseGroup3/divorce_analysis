#coding: UTF-8

import numpy as np
import re 
from scipy import stats
import json, jieba, wordcloud
import matplotlib.pyplot as plt
from imageio import imread

from util import util

#%matplotlib inline
plt.rc('figure', figsize=(15, 15))
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号



filename = 'E:/project_agg/divorce_analysis/divorce.json'
divorce = []
with open(filename, 'r', encoding='utf-8') as f:
    for line in f:
        divorce += [json.loads(line)]


#先寻找labels所代表的内容
#可以做的更加工程化：按照每个label分析热词之类的，从而分析出每个label是啥玩意

# target_label = 'DV9'
# count = 0
# for lines in divorce:
#     for every_dict in lines:
#         if target_label in every_dict['labels']:
#             #print(True if '债' in every_dict['sentence'] else every_dict)
#             print(every_dict)

#建立每个label的字典
label_dict = {}
for lines in divorce:
    for every_dict in lines:
        if len(every_dict['labels']) != 0:
            for dv in every_dict['labels']:
                if dv in label_dict.keys():
                    label_dict[dv] += [every_dict['sentence']]
                else:
                    label_dict[dv] = [every_dict['sentence']]
#print(label_dict['DV10'])
#停用词字典
stop_word = ['年','月', '日', '元', '万元', '原告', '被告', '原被告', '向', '的', '由', '在', '为']

label_dict_seg = {}
for labels in label_dict.keys():
    tmp = label_dict[labels]
    label_dict_seg[labels] = ''
    for word in tmp:
        word = word.replace(' ', '')
        word = re.sub(r'[，。、；（）：“”]*', '', word)
        tmp_stop_word = jieba.cut(word, cut_all = False)
        for seg in tmp_stop_word:
            if seg not in stop_word:
                label_dict_seg[labels] += ' ' + seg 
        #label_dict_seg[labels] += ' ' + ' '.join(jieba.cut(word, cut_all = False))

#print(label_dict_seg['DV1'])


##做词云
# 引入字体
font=r"C:/Windows/Fonts/simhei.ttf"
#读取背景图片,生成矩阵
color_mask = imread("E:/project_agg/divorce_analysis/love.jpg")
# 生成词云对象，设置参数
cloud = wordcloud.WordCloud( font_path=font,#设置字体
           background_color="black", #背景颜色
           max_words=2000,# 词云显示的最大词数
           mask=color_mask,#设置背景图片
           max_font_size=100, #字体最大值
           random_state=42)
# 绘制词云图
mywc = cloud.generate(label_dict_seg['DV1'])
plt.imshow(mywc)
# mywc2 = cloud.generate(label_dict_seg['DV9'])
# plt.imshow(mywc2)
plt.show()

#之后可以做一些描述统计的内容，比如说判例中离婚了多少，没离婚的多少。
#进一步的描述统计，比如某个词在总共中出现了多少次
total = len(divorce)
count_divorce = 0
for lines in divorce:
    sig = 0
    for every_dict in lines:
        if 'DV9' in every_dict['labels']:
            sig = 1
    if sig == 1:
        count_divorce += 1
        #every_dict += 

print(count_divorce)
print(total)
#正负样本数量相当：185 vs 215，因此不用考虑增加负样本的算法



#可能做一下归因分析？或者说相关度分析，离婚未离婚都是由于啥造成的，做一个tree model，寻找一些因子，去训练一下树？



#离不离婚作为y，有01之分




