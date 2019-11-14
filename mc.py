from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras_bert import get_model,compile_model
import numpy as np
from bert4keras.utils import Tokenizer
import os
import tensorflow as tf



class Mc_infer:
    def __init__(self,tokenizer):
        self.tokenizer = tokenizer
        self.model_path = os.path.join(os.path.dirname(__file__), "models/mc_weights_new.hf")
        self.graph = tf.Graph()
        with self.graph.as_default(), tf.device("/gpu:1"):
            self.model = self.load_bert_mc()

    def load_bert_mc(self):
        b_model = get_model(token_num=21128, )##21128是词典大小
        compile_model(b_model)
        bert_model = Model(
            inputs=b_model.input[:2],
            outputs=b_model.get_layer('Encoder-12-FeedForward-Norm').output
        )
        x1_in = Input(shape=(None,)) # 问题和资料的拼接句子输入
        x2_in = Input(shape=(None,)) # 问题和资料的拼接句子输入
        s1_in = Input(shape=(None,)) #答案的左边界（标签）
        s2_in = Input(shape=(None,)) #答案的右边界（标签）
        x1, x2, s1, s2 = x1_in, x2_in, s1_in, s2_in
        x_mask = Lambda(lambda x: K.cast(K.greater(K.expand_dims(x, 2), 0), 'float32'))(x1)
        x = bert_model([x1, x2])
        ps1 = Dense(1, use_bias=False)(x)
        ###[[0.1],[0.2],[0.3]..] -> [0.1,0.2,0.3,...]
        ###[0.1,0.2,0.3,...] - [0,0,0,0,1,1,1,1]*1e10
        ps1 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps1, x_mask])
        # ps1 = Lambda(lambda x: x[0]*x[1])([ps1, x_mask])
        ps2 = Dense(1, use_bias=False)(x)
        ps2 = Lambda(lambda x: x[0][..., 0] - (1 - x[1][..., 0]) * 1e10)([ps2, x_mask])
        # ps2 = Lambda(lambda x:x[0]*x[1])([ps2, x_mask])
        model = Model([x1_in, x2_in], [ps1, ps2])
        model.load_weights(self.model_path)
        return  model

    def softmax(self,x):
        x = x - np.max(x)
        x = np.exp(x)
        return x / np.sum(x)

    def generate_ans(self,text_in, c_in):
        text_in = u'___%s___%s' % (c_in, text_in)
        text_in = text_in[:510]
        _tokens = self.tokenizer.tokenize(text_in)
        _x1, _x2 = self.tokenizer.encode(text_in)
        _x1, _x2 = np.array([_x1]), np.array([_x2])
        with  self.graph.as_default():
            _ps1, _ps2  = self.model.predict([_x1, _x2])
        _ps1, _ps2 = self.softmax(_ps1[0]), self.softmax(_ps2[0])
        start = _ps1.argmax()
        # print(start)
        end = _ps2[start:].argmax() + start
        # print(end)
        ans = text_in[start-1: end]
        # print(ans)
        return ans


def get_token_dict(token_file):
    with open(token_file,"r") as f:
        token_list = f.readlines()
        token_dict = {word.strip():id_ for id_,word in enumerate(token_list)}
    return token_dict

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R



if __name__ =="__main__":
    dict_path = '/opt/developer/wp/wzcq/roberta_wwm/vocab.txt'
    token_dict = get_token_dict(dict_path)
    tokenizer = OurTokenizer(token_dict)
    mc_model = Mc_infer(tokenizer)
    mc_model.generate_ans('''吴恩达（1976-，英文名：Andrew Ng），华裔美国人，是斯坦福大学计算机科学系和电子工程系副教授，人工智能实验室主任。吴恩达是人工智能和机器学习领域国际上最权威的学者之一。吴恩达也是在线教育平台Coursera的联合创始人（with Daphne Koller）。
        2014年5月16日，吴恩达加入百度，担任百度公司首席科学家，负责百度研究院的领导工作，尤其是Baidu Brain计划。 [1]
        2017年10月，吴恩达将出任Woebot公司新任董事长，该公司拥有一款同名聊天机器人。''',
                   "吴恩达什么时候加入百度？")
