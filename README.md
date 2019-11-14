# NLP WEBAPP

# 项目背景 
+ 由于对基于深度学习的自然语言处理技术较为感兴趣，于是用bert训练了两个有趣的机器人————自动写词机器人和阅读理解机器人
+ 算法部分主要还是借鉴了苏剑林**苏神的项目https://github.com/bojone** 

# 项目技术

### 项目目录
![project](/img/project.png)
+ 其中train目录下是模型训练脚本，包含训练数据
+ stastic和template是前端部分
+ 预训练的bert权重的下载地址是https://github.com/ymcui/Chinese-BERT-wwm
+ 训练好的模型会放在models文件夹中
+ app.py ci.py mc.py 是后端部分

#### 运行项目

+ 由于模型太大无法上传，如果想体验的同学，首先clone项目
+ 然后去https://github.com/ymcui/Chinese-BERT-wwm 这里下载BERT预训练的权重,解压后放到项目中
+ 环境配置 首先 pip install git+https://www.github.com/bojone/bert4keras.git 这个库,然后pip install -r requirements.txt 安装项目依赖
+ 环境配置好之后在运行train里面的两个ipython文件训练生成模型文件
+ 然后按照上述目录的模型的文件名保存到models文件夹中，运行python app.py 即可启动服务 
+ 完成上述过程后你可以以容器化的方式使用docker部署你的webapp ：
      制作镜像命令是docker build -t nlpweb:v1 .
      启动容器采用 docker run -d -p 8890:8890 nlpweb:v1  


### 算法部分
+ 自动写词机器人的算法部分主要借鉴苏剑林大神的这篇博客https://spaces.ac.cn/archives/6933,
+ 阅读理解机器人的的算法部主要借鉴苏剑林大神的这篇博客https://spaces.ac.cn/archives/6736,
+ 建议大家仔细研读一下博客和代码

### 部署部分
+ 前端技术 bootstrap,js,css
+ 后端技术 flask
+ docker容器化部署

# webapp 演示部分

### 自动写词机器人
##### 其中第一个form填**词牌名**，格式是——**“菩萨蛮：”** , 第二个form填beamsearch解码器的**topk值**,不同的topk会生成不同的词。
当beamsearch的 topk = 6时
![generate_ci](/img/ci1.png)
当beamsearch的 topk = 10时
![generate_ci](/img/ci2.png)
### 阅读理解机器人
##### 其中第一个form填入**资料**， 第二个form填入**问题**,机器会自动帮你寻找到合适的答案。
演示1
![geerate_ans](/img/mc1.png)
演示2
![generate_ans](/img/mc2.png)

# TO DO
+ 采用容器化技术docker实现项目的部署
+ 增加更多的NLP相关的机器人

