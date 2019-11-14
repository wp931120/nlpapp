#coding:utf-8
from flask import Flask,request
from flask import  render_template
import json
from mc import *
from ci import *
from bert4keras.utils import Tokenizer
app = Flask(__name__)


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

dict_path = '/opt/developer/wp/wzcq/roberta_wwm/vocab.txt'
token_dict = get_token_dict(dict_path)
tokenizer = OurTokenizer(token_dict)

@app.route('/')
def hello_world():
    data = {}
    return render_template("ci.html", **data)

@app.route('/mc')
def machine_read():
    return render_template('mc.html')

@app.route('/ci')
def generate_ci():
    return render_template('ci.html')

@app.route('/gen_ans',methods=["POST"])
def gen_ans():
    result = {}
    doc = request.form['doc']
    qry = request.form['qry']
    mc_model = Mc_infer(tokenizer)
    ans = mc_model.generate_ans(doc,qry)
    result['content'] = ans
    return json.dumps(result, ensure_ascii=False)

@app.route('/gen_ci',methods=["POST"])
def gen_ci():
    result = {}
    ci_head = request.form['ci_head']
    topk = request.form["topk"]
    seq_model = ci_infer(tokenizer, token_dict)
    ans = seq_model.gen_sent(ci_head,topk=int(topk))
    result['content'] = ans
    return json.dumps(result, ensure_ascii=False)




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8890, debug = True)
