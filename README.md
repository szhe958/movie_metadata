# movie_metadata

# Airbnb Automation

## Table of contents
* [功能](#功能)
* [项目介绍](#项目介绍)
* [使用方式](#使用方式)
* [Techonologies](#Techonologies)
* [TODO](#TODO)


## 功能
* AirbnbAutomation 
* Airbnb_scraper
* Sender
* Airtable_manager
* Prox_pool 
* Airbnb_publisher

## 项目介绍
Airbnb作为最大的短租平台之一，提供了高价值的房源信息和房东信息。为获取相关的数据，拓展销售渠道并扩大潜在客户，本项目实现了Airbnb爬虫、向房主自动发送信息、将Airbnb数据传入Airtable的功能。

为实现以上功能，我们首先创建了用于初始化账号信息的AirbnbAutomation。然后，我们通过Airbnb_Scraper来实现爬虫，Sender来实现批量发送信息，Airtable Manger来实现Airtable的数据存储、更新。

### 1 . AirbnbAutomation 
<br>这是一个用于初始化一个保存有目标城市、Airbnb账号信息、Airtable信息的等参数的类，一般不直接使用；获得一个使用本地账号Cookies自动登录Airbnb网站的driver。

### 2. Airbnb_scraper
<br> 这是一个用于爬取Airbnb房源数据的类。通过在Airbnb检索小区名称，如("珠江新城, 广州市)，爬取房源标示ID后，利用Airbnb的Api获取房源详细信息。因此使用该类之前，需要先使用**Ke**爬取小区数据。

### 3. Sender
 <br> 这是一个通过给定房源id，使用selenium向指定房东发送信息的类。根据房源详细信息（地区、价格等）与房东经营状况（房客评论、入住率、房东回复速度），筛选经营不善的一手房东，向他们发送推广信息。
< img width="250" height="500" alt="Sender流程图" src="https://user-images.githubusercontent.com/39979406/80678064-ba41a280-8aec-11ea-8d2d-275521ad151b.png">
 
### 4. Airtable Manager
<br> 这是一个集合了爬虫操作和Airtable数据存储、更新操作的类。

### 5. Proxy pool 
<br> 这是一个ip代理池的类。通过使用ip代理池，防止ip被禁用的问题的发生。

### 6. Airbnb_publisher
<br> 该类通过一个Airbnb账号自动上传到官网，以期降低成本，助推新签约房产在期初空置期按时上线，提高运营标准化程度。
需要本地指定一个文件夹存储所有房源的信息，以及一个文件夹包含浏览器和登陆cookies.json文件，房源信息会从airtable读取，图片信息会下载到本地的文件夹中，并在上传发布后自动删除。

## Setup
 
1. 环境配置：

* Python : 3.6+

* 运行系统 : Windows / Linux / Mac

* selenium

* [Airbnb api][1]

* [Airtable python wrapper][2]
<br>

2. 配置Selenium相关文件
为使用Selenium，需要下载并安装chromedriver，并配置Chromedriver环境变量

* 下载chromedriver 见[Ciwei部署视频](https://drive.weixin.qq.com/s?k=AAsA-wf4AAo6tZDL9MAEoAVgYqAKo)
* Mac系统下配置chromedriver环境变量 
```
sudo mv chromedriver /usr/bin
```

 * Windows系统下配置chromedriver环境变量，[链接](https://blog.csdn.net/qq_41429288/article/details/80472064)
<br>

3. 收集Airbnb账号信息
为实现Airbnb网站的自动登陆，需要手动保存Airbnb账号的Cookies(自动保存有时会失效)

* 安装chrome插件EditThisCookies
* 手动登陆Airbnb账号后, 导出文本类型的cookies后保存为```Cookies/username.json```
* Cookies文件的命名格式，rangduju@163.com -> rangduju.json
## 使用方式 
1. 各类使用方式详见 AirbnbAutomation.i
