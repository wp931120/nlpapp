from bert4keras.bert import build_bert_model
import numpy as np
import os
from bert4keras.utils import Tokenizer
import tensorflow as tf


class ci_infer:
    def __init__(self,tokenizer,token_dict):
         self.tokenizer = tokenizer
         self.token_dict = token_dict
         self.config_path = '/opt/developer/wp/wzcq/roberta_wwm/bert_config.json'
         self.model_path = os.path.join(os.path.dirname(__file__),"models/best_model.weights")
         self.graph = tf.Graph()
         with self.graph.as_default(), tf.device("/gpu:3"):
            self.model = self.load_bert_seq2seq()

    def load_bert_seq2seq(self):
        model = build_bert_model(
            self.config_path,
            application='seq2seq',
        )
        model.load_weights(self.model_path)
        return model


    def gen_sent(self,s,
                 topk=10,
                 max_input_len = 8,
                 max_output_len=100,):
        """beam search解码
        每次只保留topk个最优候选结果；如果topk=1，那么就是贪心搜索
        """
        token_ids, segment_ids =self.tokenizer.encode(s[:max_input_len])
        target_ids = [[] for _ in range(topk)]  # 候选答案id
        target_scores = [0] * topk  # 候选答案分数
        for i in range(max_output_len):  # 强制要求输出不超过max_output_len字
            _target_ids = [token_ids + t for t in target_ids]
            _segment_ids = [segment_ids + [1] * len(t) for t in target_ids]
            with self.graph.as_default():
                _probas = self.model.predict([_target_ids, _segment_ids
                                         ])[:, -1, 3:]  # 直接忽略[PAD], [UNK], [CLS]
            _log_probas = np.log(_probas + 1e-6)  # 取对数，方便计算
            _topk_arg = _log_probas.argsort(axis=1)[:, -topk:]  # 每一项选出topk
            _candidate_ids, _candidate_scores = [], []
            for j, (ids, sco) in enumerate(zip(target_ids, target_scores)):
                # 预测第一个字的时候，输入的topk事实上都是同一个，
                # 所以只需要看第一个，不需要遍历后面的。
                if i == 0 and j > 0:
                    continue
                for k in _topk_arg[j]:
                    _candidate_ids.append(ids + [k + 3])
                    _candidate_scores.append(sco + _log_probas[j][k])
            _topk_arg = np.argsort(_candidate_scores)[-topk:]  # 从中选出新的topk
            target_ids = [_candidate_ids[k] for k in _topk_arg]
            target_scores = [_candidate_scores[k] for k in _topk_arg]
            best_one = np.argmax(target_scores)
            if target_ids[best_one][-1] == self.token_dict.get("[SEP]"):
                return self.tokenizer.decode(target_ids[best_one])
        # 如果max_output_len字都找不到结束符，直接返回
        return self.tokenizer.decode(target_ids[np.argmax(target_scores)])

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
    seq_model = ci_infer(tokenizer, token_dict)
    print(seq_model.gen_sent("菩萨蛮:"))