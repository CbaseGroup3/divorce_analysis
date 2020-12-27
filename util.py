#coding: UTF-8
import numpy as np
import pandas as pd 
import re 
from scipy import stats
import json, jieba, wordcloud
import matplotlib.pyplot as plt
from imageio import imread
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
    
    def isdivorce(self, div_line):
        #判断这个案子最终是否离婚
        sig = 0
        for every_dict in div_line:
            if 'DV9' in every_dict['labels']:
                sig = 1
                break 
        return True if sig == 1 else False 


    def count_dv(self, div_line):
        #本函数用于计算这一行各个dv的数量
        #从dv1到dv20
        dv_dict = {}
        for i in range(1, 21):
            if i != 9:
                label = 'DV' + str(i)
                dv_dict[label] = 0
        
        #将这一行中dv_count存到dv_dict中去
        for every_dict in div_line:
            if len(every_dict['labels']) != 0:
                for dv in every_dict['labels']:
                    if dv != 'DV9':
                        dv_dict[dv] += 1
        
        return dv_dict


    #训练词向量，把所有的句子都放到一起
    def _pre_process(self, divorce, stop_word):
        out = []
        for lines in divorce:
            for every_dict in lines:
                tmp_1 = []
                tmp = every_dict['sentence']
                tmp = re.sub(r'[、（）“”，。；：]*', '', tmp)
                tmp = tmp.replace(' ', '')
                tmp_stop_word = jieba.cut(tmp, cut_all = False)
                
                for seg in tmp_stop_word:
                    if seg not in stop_word and seg != '':
                        tmp_1 += [seg] 
                if tmp_1 != '':
                    out += [tmp_1]
        
        return out 

    def w2v(self, divorce, stop_word):
        sentence_agg = self._pre_process(divorce, stop_word)
        model = Word2Vec(sentence_agg, size = 200, window = 5, min_count = 3)
        return model

    #可以尝试更多的word embedding方法
    # tf-idf, 



    #需要做sentence embedding，然后训练分类模型
    def data_prepare(self, divorce, stop_word):
        model1 = self.w2v(divorce, stop_word)
        n = model1.wv.vectors.shape[1]

        # sentence embedding 先尝试加权平均
        ##先产出label
        label = list(self.count_dv(divorce[0]).keys())
        for i in range(n):
            tmp = 'c' + str(i)
            label += [tmp]
        label += ['isdivorce']

        out = []
        for lines in divorce:
            tmp = self._pre_process([lines], stop_word) #jieba.cut
            sen_embedding = [0] * n
            count = 0

            for sen in tmp:
                for wd in sen:
                    #word必须在word2vec的词列表中
                    if wd in model1.wv.index2word:
                        vec = model1.wv.vectors[model1.wv.vocab[wd].index]
                        sen_embedding = [(sen_embedding[i] + vec[i]) for i in range(n)] 
                        count += 1
            
            sen_embedding = [(sen_embedding[i]/count) for i in range(n)]
            dv = self.count_dv(lines)
            y = self.isdivorce(lines)
            out += [list(dv.values()) + sen_embedding + [1 if y else 0]]
        
        df = pd.DataFrame.from_records(out, columns = label)

        return df 
            
    def classify_logis(self, dataset):
        X = dataset.iloc[:, :219]
        y = dataset['isdivorce']
        #print(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=33)
        model = LogisticRegression(max_iter=10000)
        model.fit(X_train, y_train)
        print(model.score(X_test, y_test))

        return model 
    

            








# word = '欧几里得'
# vec = model.wv.vectors[model.wv.vocab[word].index]

# print('词向量长度：', vec.shape)
# print('词向量：\n', vec)


