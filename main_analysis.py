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

filename = 'D:/project_agg/divorce_demo/divorce.json'
stop_word = ['年','月', '日', '元', '万元', '原告', '被告', '原被告', '向', '的', '由', '在', '为', '与', '之', '了']
color_mask = "D:/project_agg/divorce_demo/love.jpg"


divorce = util().read_json(filename)
# label_dict = util().dict_every_label(divorce)

# util().draw_cloudmap(label_dict, stop_word, color_mask, 'DV9')

# print(util().count_divorce(divorce))

#out = util()._pre_process(divorce, stop_word)
# model = util().w2v(divorce, stop_word)
# print('词向量维度：', model.wv.vectors.shape[1])
# print(model.wv.vectors)
#print(out[:10])
# print(model.wv.similarity('债务', '离婚'))

#组成的数据集应该如下，embedding情况，和dv的count（表明这个案子判决过程中相关内容提及的数量）

# print(util().count_dv(divorce[0]))

dataset = util().data_prepare(divorce, stop_word)
#print(dataset)
print('data is well prepared!')
model = util().classify_logis(dataset)
model2 = util().classify_cart(dataset)