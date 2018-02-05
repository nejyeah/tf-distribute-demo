## Things to do
Problem1: Only worker0 works for all training task, time is 22.32 s.
Solved: the training code should be under the tf.device 

##  1.基本概念：

1. Cluster:  定义一个集群
2. Job:  定义一个Job(ps, worker)
```
cluster = tf.train.ClusterSpec({'ps': ps_spec, 'worker': worker_spec})
```

3. Task: 每个Job可以有多个task，一般一台机器上运行一个task

##  2.运行：


```
	cluster_spec = {
		'ps': '10.240.209.96:8888', # 参数服务器
		'worker': '10.240.209.95:8888,10.240.209.95:9999' #两个worker，每一个有一个task
	}
```

## 3. 测试结果
```
# 单机测试
1个ps 与2个worker在同一台机器上
batch_size: 100
step: 10000
Train time: 11.9 s

# 多机测试
ps 在一个机器上， 2个worker 在另外一个机器上
batch_size: 100
step: 10000
Train time: 66.6 s
```
