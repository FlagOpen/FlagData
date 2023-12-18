The following describes in detail how to use distributed capabilities for data deduplication

first. Build a Spark standalone cluster (1 master2 worker)
1. Install jdk

a. Download the jdk package

Download address https://www.oracle.com/java/technologies/downloads/

b. Decompression

   tar -zxvf ./jdk-8u181-linux-x64.tar.gz

c. Environment configuration

   ~/.bashrc Add at the end of the file, after adding source ~/.bashrc Take effect


```bash
# JAVA_HOME
export JAVA_HOME=/xxx/jdk1.8.0_181
# PATH
export PATH=$PATH:${JAVA_HOME}/bin
```





d. Successful effect

   java -version Show version number
2. Install spark cluster

a. Download the spark package

Download address https://spark.apache.org/downloads.html

b. Extract and rename

   tar -zxvf spark-2.3.1-bin-hadoop2.6.tgz

   mv spark-2.3.1-bin-hadoop2.6.tgz  spark-2.3.1

c. Modify the configuration file

d. Configure spark-env.sh

   i. cd spark-2.3.1

   ii. cp spark-env.sh.template spark-env.sh

   iii. vim spark-env.sh
```bash
SPARK_MASTER_PORT=7077          #master Service port
SPARK_MASTER_HOST=172.31.32.51  #master node ip   ifconifg command to find，ifconfig The command cannot be found and is required. apt install net-tools Installation and execution ifconfig
JAVA_HOME=/data1/jdk-11.0.15.1  #master Node jdk address  echo $JAVA_HOME Find
PYSPARK_PYTHON=/data1/miniconda3/bin/python   #python Environment
PYSPARK_DRIVER_PYTHON=/data1/miniconda3/bin/python
SPARK_MASTER_WEBUI_PORT=50010   #master webUI Port prevents port conflicts lsof -i:50010
SPARK_WORKER_WEBUI_PORT=50011   #worker webUI Port
```


Use which python to determine the path

If there is no python environment, it is recommended that conda manage:

```bash
Install Miniconda on Linux:

1. For Linux systems, download the Miniconda installation script using the following command:


wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh


2. Next, run the following command to execute the installation script:


bash Miniconda3-latest-Linux-x86_64.sh


3. Follow the instructions of the installer to install. Install according to the default settings, or customize the settings as needed.

4. After the installation is complete, you may need to activate Miniconda. You can activate Miniconda by executing the following command:
Add to the tail of vim ~ / .bashrc
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


5. Now, to verify that Miniconda is installed successfully, try the following command to check the version of Miniconda:
conda --version

If the installation is successful, you can see the version number of the installed Miniconda.

After installation, you can use Miniconda to manage your Python environment and install various packages and dependencies.
```

e. 配置Spark-defaults.conf

   i. cp spark-defaults.conf.template spark-defaults.conf

   ii. vim spark-defaults.conf
```bash
spark.master                     spark://172.31.32.51:7077       #master Nodes: Port
spark.driver.host                172.31.32.51                    #master Nodes ip
spark.eventLog.enabled           true                            #Open the log
spark.eventLog.dir               file://your_path/spark/eventLog     #Log address
# spark.serializer               org.apache.spark.serializer.KryoSerializer
# spark.executor.instances       5
spark.driver.memory              32g  # Spark application drive memory, take the previous example to give a reference value
# spark.executor.memory 340g
# spark.executor.extraJavaOptions  -XX:+PrintGCDetails -Dkey=value -Dnumbers="one two three"
spark.executor.extraJavaOptions -Dio.netty.tryReflectionSetAccessible=true  #Spark Actuator configures Netty Network Library to improve Network performance
spark.network.timeout   10000000
spark.memory.offHeap.enabled     true
spark.memory.offHeap.size        4g                              #Out-of-heap memory
```

f. Configure the spark worker nod

   i. Non-secret

1. ~/.ssh/config
   Several work configurations, vim / etc/hosts configuration local hosts, enter hostname verification, # ubuntu should not be written as ubuntu
       vim ~/.ssh/config

```bash
Host spark-worker0
    HostName 172.31.0.149 # Ifconifg command lookup
    User root
    Port 22               # Machine login port number
Host spark-worker1
    HostName 172.xx.x.xxx 
    User root   # Ubuntu should not be written as ubuntu
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
2. Send secret-free
      ssh-copy-id -i ~/.ssh/id_rsa.pub spark-worker1【execute under root user】
   If master is missing a public key file. Follow these steps to check and generate the public key file:

```bash

1. First, check to see if a public key file named `id rsa.pub` already exists. You can execute the following command to check:
   ls ~/.ssh/id_rsa.pub

2. If the file does not exist, a new SSH key pair can be generated using the `ssh-keygen` command. Execute the following command:
   ssh-keygen

   Follow the prompts to enter information such as path and password to generate a new SSH key pair.

3. Copy the public key content in the generated public key file `public key rsa.pub`. Then execute the `public key id` command to copy the public key to the target host, and make sure to replace `< your username >` and `< remote host >` as the correct user name and remote host name:
   ssh-copy-id -i ~/.ssh/id_rsa.pub spark-worker1
```

ii. Configure workers
   cp workers.template workers
   vim workers，tail addition
```bash
   spark-master #Master nodes can also be used as worker nodes at the same time
   spark-worker1
```
g. Send the configured spark packets to each worker node respectively

scp -r spark-2.3.1 root@120.92.85.38:spark-standalone (the address of worker startup must be the same or it cannot be started) every worker must have (shared data directory / or hdfs) the files being accessed

3. Start the spark cluster

```bash
启动master
./sbin/start-master.sh
./sbin/stop-master.sh #Close master
启动work
./sbin/start-workers.sh
./sbin/stop-worker.sh  #close work
```

4. Visit web

work http://120.92.14.245:50010/

5. Submit spark task

Standalone commit command


```bash
cd bin
./spark-submit --master spark://172.31.32.51:7077 --class org.apache.spark.examples.SparkPi ../examples/jars/spark-examples_2.12-3.4.0.jar 10000
```


