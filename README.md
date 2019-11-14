# nlpapp
# 项目背景 
+ 由于对基于深度学习的自然语言处理技术较为感兴趣，于是用bert训练了两个有趣的机器人,算法不服主要还是借鉴了苏神的项目
+ 自动写词机器人和阅读理解机器人
# 项目技术
### 算法
+ 自动写词机器人的算法部分主要借鉴苏剑林大神的这篇博客https://spaces.ac.cn/archives/6933,
+ 阅读理解机器人的的算法部主要借鉴苏剑林大神的这篇博客https://spaces.ac.cn/archives/6736,
### 部署
+ 前端技术 bootstrap,js,css
+ 后端技术 flask
# webapp 演示部分
### 自动写词机器人
其中第一个form填**词牌名**，格式是——**“菩萨蛮：”** , 第二个form填beamsearch解码器的**topk值**,不同的topk会生成不同的词。
![generate_ci](/img/ci1.png)
![generate_ci](/img/ci2.png)
### 阅读理解机器人
![generate_ci](/img/mc1.png)
![generate_ci](/img/mc2.png)
