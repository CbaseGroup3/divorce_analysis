#coding: UTF-8
import numpy as np
import re 
from scipy import stats
import json, jieba, wordcloud
import matplotlib.pyplot as plt
from imageio import imread

class util:
    def __init__(self):
        return 
    
    #将源数据导入，json数据表现如下：
    #每一行为一个完整的判例，有若干个字典，字典包含一个label和具体判决书
    def read_json(self, root_json):
        divorce = []
        with open(root_json, 'r', encoding='utf-8') as f:
            for line in f:
                divorce += [json.loads(line)]
        return divorce
    
    #生成每个label的对应的字典，归纳每个label对应的特征
    def dict_every_label(self, divorce):
        label_dict = {}
        for lines in divorce:
            for every_dict in lines:
                if len(every_dict['labels']) != 0:
                    for dv in every_dict['labels']:
                        if dv in label_dict.keys():
                            label_dict[dv] += [every_dict['sentence']]
                        else:
                            label_dict[dv] = [every_dict['sentence']]
        return label_dict

    def draw_cloudmap(self, label_dict, stop_word, back_pic, DV):
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
        
        font=r"C:/Windows/Fonts/simhei.ttf"
        #读取背景图片,生成矩阵
        color_mask = imread(back_pic)
        # 生成词云对象，设置参数
        cloud = wordcloud.WordCloud( font_path=font,#设置字体
                background_color="black", #背景颜色
                max_words=2000,# 词云显示的最大词数
                mask=color_mask,#设置背景图片
                max_font_size=100, #字体最大值
                random_state=42)
        # 绘制词云图
        mywc = cloud.generate(label_dict_seg[DV])
        plt.imshow(mywc)
        # mywc2 = cloud.generate(label_dict_seg['DV9'])
        # plt.imshow(mywc2)
        plt.show()

    #对是否含有dv9计数，dv9表明离婚成功
    def count_divorce(self, divorce):
        total = len(divorce)
        count_divorce = 0
        for lines in divorce:
            sig = 0
            for every_dict in lines:
                if 'DV9' in every_dict['labels']:
                    sig = 1
            if sig == 1:
                count_divorce += 1
        return [total, count_divorce]

    #训练词向量，把所有的句子都放到一起
    def pre_process(self, divorce, stop_word):
        out = []
        for lines in divorce:
            for every_dict in lines:
                tmp = every_dict['sentence']
                tmp = re.sub(r'[、（）“”]*', '', tmp)
                tmp = tmp.replace(' ', '')
                tmp = re.split(r'[，。；：]', tmp)
                
                for tmp_new in tmp:
                    tmp_stop_word = jieba.cut(tmp_new, cut_all = False)
                    tmp_1 = ''
                    for seg in tmp_stop_word:
                        if seg not in stop_word and seg != '':
                            tmp_1 += ' ' + seg 
                    if tmp_1 != '':
                        out += [[tmp_1]]
        
        return out 


