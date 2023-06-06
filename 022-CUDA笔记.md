# CUDA笔记



### 1 异构并行计算

CUDA编程模型是一个异构模型，需要CPU和GPU协同工作。在CUDA中，**host**和**device**是两个重要的概念。

我们用host指代CPU及其内存，而用device指代GPU及其内存。

CUDA程序中既包含host程序，又包含device程序，它们分别在CPU和GPU上运行。同时，host与device之间可以进行通信，这样它们之间可以进行数据拷贝。

典型的CUDA程序的执行流程如下：

1. 分配host内存，并进行数据初始化；
2. 分配device内存，并从host将数据拷贝到device上；
3. 调用CUDA的核函数在device上完成指定的运算；
4. 将device上的运算结果拷贝到host上；
5. 释放device和host上分配的内存。

### 2 CUDA编程模型

### 3 CUDA执行模型

### 4 内存

### 5 流与并发

### 6 指令极原语言

### 7 GPU加速库

### 8 多GPU编程

### 9 注意事项
