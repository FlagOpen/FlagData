下面详细描述了如何使用分布式能力来进行数据去重
一、Spark standalone集群（1台master2台worker）搭建
1. 安装jdk
1. 下载jdk包
   下载地址 https://www.oracle.com/java/technologies/downloads/
2. 解压
   tar -zxvf ./jdk-8u181-linux-x64.tar.gz
3. 环境配置
   ~/.bashrc文件尾部添加，添加之后source ~/.bashrc生效


```bash
# JAVA_HOME
export JAVA_HOME=/xxx/jdk1.8.0_181
# PATH
export PATH=$PATH:${JAVA_HOME}/bin
```





4. 成功效果
   java -version显示版本号
2. 安装spark集群
1. 下载spark包
   下载地址 https://spark.apache.org/downloads.html
2. 解压并重命名
   tar -zxvf spark-2.3.1-bin-hadoop2.6.tgz
   mv spark-2.3.1-bin-hadoop2.6.tgz  spark-2.3.1
3. 修改配置文件
4. 配置spark-env.sh
   1. cd spark-2.3.1
   2. cp spark-env.sh.template spark-env.sh
   3. vim spark-env.sh
```bash
SPARK_MASTER_PORT=7077          #master 服务端口
SPARK_MASTER_HOST=172.31.32.51  #master 节点ip   ifconifg命令查找，ifconfig命令找不到，需要apt install net-tools 安装再执行ifconfig
JAVA_HOME=/data1/jdk-11.0.15.1  #master 节点jdk地址  echo $JAVA_HOME 查找
PYSPARK_PYTHON=/data1/miniconda3/bin/python   #python环境
PYSPARK_DRIVER_PYTHON=/data1/miniconda3/bin/python
SPARK_MASTER_WEBUI_PORT=50010   #master webUI端口 防止端口冲突lsof -i:50010
SPARK_WORKER_WEBUI_PORT=50011   #worker webUI端口
```


使用which python确定路径
如果没有python环境，推荐conda进行管理：

```bash
在Linux上安装Miniconda：

1. 对于Linux系统，使用以下命令下载Miniconda安装脚本：


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


2. 接下来，运行以下命令来执行安装脚本：


bash Miniconda3-latest-Linux-x86_64.sh


3. 按照安装程序的指示进行安装。按照默认设置进行安装，或者根据需要进行自定义设置。

4. 安装完成后，您可能需要激活Miniconda。可以通过执行以下命令来激活Miniconda：
在vim ~/.bashrc尾部添加
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/root/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

source ~/.bashrc


5. 现在，验证Miniconda是否成功安装，可以尝试执行以下命令来检查Miniconda的版本：
conda --version

如果安装成功，是能够看到安装的Miniconda的版本号。

安装完成后，可以使用Miniconda来管理您的Python环境和安装各种包和依赖项。
```

5. 配置spark-defaults.conf
   1. cp spark-defaults.conf.template spark-defaults.conf
   2. vim spark-defaults.conf
```bash
spark.master                     spark://172.31.32.51:7077       #master节点:端口
spark.driver.host                172.31.32.51                    #master节点ip
spark.eventLog.enabled           true                            #开启日志
spark.eventLog.dir               file:///cwwu/spark_setup/spark/eventLog     #日志地址
# spark.serializer               org.apache.spark.serializer.KryoSerializer
# spark.executor.instances       5
spark.driver.memory              32g  # Spark应用程序的驱动器内存，拿之前的示例给个参考值
# spark.executor.memory 340g
# spark.executor.extraJavaOptions  -XX:+PrintGCDetails -Dkey=value -Dnumbers="one two three"
spark.executor.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true  #Spark执行器配置Netty网络库以提高网络性能
spark.network.timeout   10000000
spark.memory.offHeap.enabled     true
spark.memory.offHeap.size        4g                              #堆外内存
```

6. 配置spark worker节点
   1. 免密
    1. ~/.ssh/config
       几个work配置几个，vim /etc/hosts配置本地hosts，输入hostname检验，# ubuntu 不要写成ubuntu
       vim ~/.ssh/config

```bash
Host spark-worker0
    HostName 172.31.0.149 # ifconifg命令查找
    User root
    Port 22               # 机器登录端口号
Host spark-worker1
    HostName 172.xx.x.xxx 
    User root   # ubuntu 不要写成ubuntu
    Port 22
Host spark-master-langchao
    HostName 172.31.32.51
    User root
    Port 60022
 
Host spark-worker-langchao
    HostName 172.xx.x.xxx
    User root
    Port 22
```
      2. 发送免密
      ssh-copy-id -i ~/.ssh/id_rsa.pub spark-worker1【root用户下执行】
如果master缺少公钥文件。请按照以下步骤检查和生成公钥文件：

```bash

1. 首先，检查是否已经存在名为`id_rsa.pub`的公钥文件。可以执行以下命令检查：
   ls ~/.ssh/id_rsa.pub

2. 如果文件不存在，可以使用`ssh-keygen`命令生成新的SSH密钥对。执行以下命令：
   ssh-keygen

   按照提示输入路径和密码等信息，生成新的SSH密钥对。

3. 在生成的公钥文件`id_rsa.pub`中复制公钥内容。然后执行`ssh-copy-id`命令将公钥复制到目标主机，确保替换`<your_username>`和`<remote_host>`为正确的用户名和远程主机名称：
   ssh-copy-id -i ~/.ssh/id_rsa.pub spark-worker1
```

2. 配置workers
   cp workers.template workers
   vim workers，尾部添加
   spark-master #master节点也可以同时当做worker节点使用
   spark-worker1
7. 将配置完成的spark包分别发送给各worker节点
   1. scp -r spark-2.3.1 root@120.92.85.38:/cwwu/spark-standalone（worker启动的地址必须一致不然不能启动）被访问的文件，必须每一个worker都有（共享数据目录/或者hdfs）
3. 启动spark集群

```bash
启动master
./sbin/start-master.sh
./sbin/stop-master.sh #关闭master
启动work
./sbin/start-workers.sh
./sbin/stop-worker.sh  #关闭work
```

4. 访问web
1. work http://120.92.14.245:50010/
5. 提交spark任务
1.  Standalone提交命令


```bash
cd bin
./spark-submit --master spark://172.31.32.51:7077 --class org.apache.spark.examples.SparkPi ../examples/jars/spark-examples_2.12-3.4.0.jar 10000
./bin/spark-submit --master spark://172.31.32.51:7077 --class org.apache.spark.examples.SparkPi ./examples/jars/spark-examples_2.12-3.4.0.jar 10000 

./bin/spark-submit --master spark://172.31.32.51:7077 /cwwu/kenlm/spark_script.py /cwwu/cn/paper_abstract_processed.jsonl /cwwu/spark_test_1105/paper_abstract_output
```

scp -r /cwwu/cn/paper_abstract_processed.jsonl root@120.92.85.38:/cwwu/cn


./bin/spark-submit --master spark://172.31.67.138:7077 /cwwu/text-process-spark-tool/unified_format/post_processing_unified_format_spark_downfile0.py

./spark-submit /cwwu/kenlm/spark_script.py /cwwu/cn/paper_abstract_processed.jsonl /cwwu/spark_test_1105/paper_abstract_output
***

## 改成去重的例子

***

master(worker)
worker

172.31.32.51:43851
172.31.0.149:27602

502.6 GiB  72cores
61.7 GiB 16 cores

893.8s（14.8min）

小于60s（小于1min）
2kw行（20424772）
87G


二、性能分析
任务名
数据描述
Python脚本
spark集群
中文过滤
paper_abstract_processed.jsonl
100w行
586M
893.8s（14.8min）

小于60s（小于1min）



单台400（96核*384G）


单台200（112核*512G）


三、跳板机
使用需要使用跳板机链接，多台机器通过跳板机，可以通过隧道方式链接

挂载共享数据盘
sudo su 切换到root用户
mkdir /cwwu
