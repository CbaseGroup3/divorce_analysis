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
stop_word = ['年','月', '日', '元', '万元', '原告', '被告', '原被告', '向', '的', '由', '在', '为']
color_mask = "E:/project_agg/divorce_analysis/love.jpg"


divorce = util().read_json(filename)
# label_dict = util().dict_every_label(divorce)

# util().draw_cloudmap(label_dict, stop_word, color_mask, 'DV9')

# print(util().count_divorce(divorce))

out = util().pre_process(divorce, stop_word)
print(out[:100])