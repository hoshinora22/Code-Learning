# CUDA笔记



## CUDA基础学习笔记

ref：[人工智能编程 | 谭升的博客 (face2ai.com)](https://face2ai.com/program-blog/#GPU编程（CUDA）)





## 1.0 并行计算与计算机架构

#### 1 并行计算

并行计算其实涉及到两个不同的技术领域：

- 计算机架构（硬件）
- 并行程序设计（软件）

这两个很好理解，一个生产工具，一个用工具产生各种不同应用。

- 硬件主要的目标：为软件提供更快的计算速度，更低的性能功耗比，硬件结构上支持更快的并行。
- 软件的主要目的：使用当前的硬件压榨出最高的性能，给应用提供更稳定快速的计算结果。

##### 并行性

写并行程序主要是分解任务，我们一般把一个程序看成是指令和数据的组合，当然并行也可以分为这两种：

- 指令并行
- 数据并行

我们的任务更加关注**数据并行**，所以我们的主要任务是：分析数据的相关性，哪些可以并行，哪些不能不行。

>
> 如果你对并行不太了解，可以先去学习学习pThread和OpenMP，了解下载多核CPU上是怎么并行的，比如把用openmp把for并行。

我们研究的是大规模数据计算，计算过程比较单一（不同的数据基本用相同的计算过程）但是数据非常多，所以我们主要是数据并行，分析好数据的相关性，决定了我们的程序设计。CUDA非常适合数据并行。

**数据并行**程序设计，首先需要：**把数据依据线程进行划分**

1. 块划分：把一整块数据切成小块，每个小块随机的划分给一个线程，每个块的执行顺序随机；
2. 周期划分：线程按照顺序处理相邻的数据块，每个线程处理多个数据块，比如有五个线程，线程1执行块1，线程2执行块2，...，线程5执行块5，线程1执行块6...

> 不同的数据划分严重影响程序性能，所以针对不同的问题和不同计算机结构，我们要通过和理论和试验共同来决定最终最优的数据划分。



#### 2 计算机架构

##### Flynn’s Taxonomy

划分不同计算机结构的方法有很多，广泛使用的一种被称为佛林分类法Flynn’s Taxonomy，他根据**指令和数据进入CPU的方式**分类，分为以下四类：

- 单指令单数据SISD（传统串行计算机，386）
- 单指令多数据SIMD（并行架构，比如向量机，所有核心指令唯一，但是数据不同，现在CPU基本都有这类的向量指令）
- 多指令单数据MISD（少见，多个指令围殴一个数据）
- 多指令多数据MIMD（并行架构，多核心，多指令，异步处理多个数据流，从而实现空间上的并行，MIMD多数情况下包含SIMD，就是MIMD有很多计算核，计算核支持SIMD）



为了提高并行的计算能力，我们要从架构上实现下面这些性能提升：

- **降低延迟**：**延迟**是指<u>操作从开始到结束所需要的时间</u>，一般用微秒计算，延迟越低越好。
- **提高带宽**：**带宽**是<u>单位时间内处理的数据量</u>，一般用MB/s或者GB/s表示。
- **提高吞吐量**：**吞吐量**是<u>单位时间内成功处理的运算数量</u>，一般用gflops来表示（十亿次浮点计算），吞吐量和延迟有一定关系，都是反应计算速度的，一个是时间除以运算次数，得到的是单位次数用的时间–延迟，一个是运算次数除以时间，得到的是单位时间执行次数–吞吐量。

##### 根据内存划分

计算机架构也可以根据内存进行划分：

1. **分布式内存**的多节点系统
2. **共享内存**的多处理器系统

第一个更大，通常叫做集群，就是一个机房好多机箱，每个机箱都有内存处理器电源等一些列硬件，通过网络互动，这样组成的就是分布式。

第二个是单个主板有多个处理器，他们共享相同的主板上的内存，内存寻址空间相同，通过PCIe和内存互动。多个处理器可以分多片处理器，和单片多核（众核many-core），也就是有些主板上挂了好多片处理器，也有的是一个主板上就一个处理器，但是这个处理器里面有几百个核。

GPU就属于众核系统。当然现在CPU也都是多核的了，但是他们还是有很大区别的：

- CPU适合执行复杂的逻辑，比如多分支，其核心比较重（复杂）
- GPU适合执行简单的逻辑，大量的数据计算，其吞吐量更高，但是核心比较轻（结构简单）



## 1.1 异构计算与CUDA

#### 1 异构计算

##### 异构

不同的计算机架构就是异构，如CPU与GPU的架构不同。

##### 异构架构

1. **CPU**我们可以把它看做一个指挥者，主机端，host；
2. 而完成大量计算的**GPU**是我们的计算设备，device；
3. CPU和GPU之间通过PCIe总线连接，用于**传递指令和数据**，这部分也是后面要讨论的性能瓶颈之一。

一个异构应用包含两种以上架构，所以代码也包括不止一部分：

- 主机代码：在主机端运行，被编译成主机架构的机器码
- 设备代码：在设备上执行，被编译成设备架构的机器码

所以主机端的机器码和设备端的机器码是隔离的，自己执行自己的，没办法交换执行。

主机端代码主要是控制设备，完成数据传输等控制类工作，设备端主要的任务就是计算。

因为当没有GPU的时候CPU也能完成这些计算，只是速度会慢很多，所以可以把GPU看成CPU的一个加速设备。

##### 范例

CPU和GPU相互配合，各有所长，各有所短：

- 低并行、逻辑复杂的程序适合用CPU；
- 高并行、逻辑简单的大数据计算适合GPU；

CPU和GPU线程的区别：

- CPU线程是重量级实体，操作系统交替执行线程，线程上下文切换花销很大；
- GPU线程是轻量级的，GPU应用一般包含成千上万的线程，多数在排队状态，线程之间切换基本没有开销；
- CPU的核被设计用来尽可能减少一个或两个线程运行时间的延迟，而GPU核则是大量线程，最大幅度提高吞吐量；

##### CUDA：一种异构计算平台

CUDA平台不是单单指软件或者硬件，而是建立在Nvidia GPU上的一整套平台，并扩展出多语言支持，CUDA C 是标准ANSI C语言的扩展，扩展出一些语法和关键字来编写设备端代码，而且CUDA库本身提供了大量API来操作设备完成计算。

对于API也有两种不同的层次，一种相对高层，一种相对底层，两种API是互斥的，只能用一个，两者之间的函数不可以混合调用，只能用其中的一个库：

- CUDA驱动API：低级的API，使用相对困难；
- CUDA运行时API：高级API使用简单，其实现基于驱动API。



一个CUDA应用通常可以分解为两部分：

- CPU 主机端代码
- GPU 设备端代码

CUDA nvcc编译器会自动分离你代码里面的不同部分，例如：

- 主机代码用C写成，使用本地的C语言编译器编译；
- 设备端代码，也就是核函数，用CUDA C编写，通过nvcc编译；
- 链接阶段，在内核程序调用或者明显的GPU设备操作时，添加运行时库。

**核函数**是我们后面主要接触的一段代码，也就是设备上执行的程序段。



#### 2 CUDA代码示例

```cpp
/**hello_world.cu*/

#include<stdio.h>
__global__ void hello_world(void)
{
  printf("GPU: Hello world!\n");
}

int main(int argc,char **argv)
{
  printf("CPU: Hello world!\n");
  hello_world<<<1,10>>>();
  // if no this line ,it can not output hello world from gpu
  cudaDeviceReset();
  return 0;
}
```



##### 几个关键字

```cpp
// 是告诉编译器这个是个可以在设备上执行的核函数。
__global__
```



```c++
// 这句话C语言中没有’<<<>>>’是对设备进行配置的参数，也是CUDA扩展出来的部分。
hello_world<<<1,10>>>();
```



```cpp
// 这句话如果没有，则不能正常的运行，因为这句话包含了隐式同步。
cudaDeviceReset();
```

**GPU和CPU执行程序是异步的**，核函数调用后成立刻会到主机线程继续，而不管GPU端核函数是否执行完毕，所以上面的程序就是GPU刚开始执行，CPU已经退出程序了，所以我们要等GPU执行完了，再退出主机线程。



##### CUDA程序一般步骤

1. 分配GPU内存；
2. 拷贝内存到设备；
3. 调用CUDA内核函数来执行计算；
4. 把计算完成数据拷贝回主机端；
5. 内存销毁；

上面的hello world只到第三步，没有内存交换。



#### 3 关于CUDA C

CPU与GPU的编程主要区别在于对GPU架构的熟悉程度，理解机器的结构是对编程效率影响非常大的一部分，了解你的机器，才能写出更优美的代码，而目前计算设备的架构决定了局部性将会严重影响效率，数据局部性分两种：

- 空间局部性
- 时间局部性

这个两个性质告诉我们，当一个数据被使用，其附近的数据将会很快被使用，当一个数据刚被使用，则随着时间继续其被再次使用的可能性降低，数据可能被重复使用。



CUDA中有两个模型是决定性能的：

- 内存层次结构
- 线程层次结构

CUDA C写核函数的时候我们只写一小段串行代码，但是这段代码被成千上万的线程执行，所有线程执行的代码都是相同的，CUDA编程模型提供了一个层次化的组织线程，直接影响GPU上的执行顺序。



CUDA抽象了硬件实现：

1. 线程组的层次结构
2. 内存的层次结构
3. 障碍同步

这些都是我们后面要研究的，**线程**、**内存**是主要研究的对象。





## 2.0 CUDA编程模型概述（一）

#### 1 CUDA编程模型概述

**编程模型**可以理解为：我们要用到的语法，内存结构，线程结构等这些我们写程序时我们自己控制的部分，这些部分控制了异构计算设备的工作模式，都是属于编程模型。

**CUDA编程模型**：应用和硬件设备之间的桥梁。在GPU中分为以下几个关键部分：

- 核函数
- 内存管理
- 线程管理
- 流

GPU架构下特有几个功能：

- 通过组织层次结构在GPU上组织**线程**的方法
- 通过组织层次结构在GPU上组织**内存**的方法

从宏观上我们可以从以下几个环节完成CUDA应用开发：

1. 领域层：在领域层（也就是你所要解决问题的条件）分析数据和函数，以便在并行运行环境中能正确，高效地解决问题；
2. 逻辑层：当分析设计完程序就进入了编程阶段，我们关注点应转向如何组织并发进程。CUDA模型主要的一个功能就是**线程层结构抽象的概念，以允许控制线程行为**。这个抽象为并行变成提供了良好的可扩展性（一个CUDA程序可以在不同的GPU机器上运行，即使计算能力不同）；
3. 硬件层：通过理解线程如何映射到机器上，能充分帮助我们提高性能。



#### 2 CUDA编程结构

一个异构环境，通常有多个CPU多个GPU，他们都通过PCIe总线相互通信，也是通过PCIe总线分隔开的。所以我们要区分一下两种设备的内存：

- 主机：CPU及其内存
- 设备：GPU及其内存

注意这两个内存从硬件到软件都是隔离的（CUDA6.0 以后支持统一寻址）

从host的串行到调用核函数（核函数被调用后控制马上归还主机线程，也就是在第一个并行代码执行时，很有可能第二段host代码已经开始同步执行了）。



#### 3 内存管理

内存管理在传统串行程序是非常常见的：

- 寄存器空间，栈空间内的内存由机器自己管理；
- 堆空间由用户控制分配和释放；

CUDA程序同样，只是CUDA提供的API可以分配管理设备上的内存，当然也可以用CDUA管理主机上的内存，主机上的传统标准库也能完成主机内存管理。

| 标准C函数 | CUDA C 函数 |   说明   |
| :-------: | :---------: | :------: |
|  malloc   | cudaMalloc  | 内存分配 |
|  memcpy   | cudaMemcpy  | 内存复制 |
|  memset   | cudaMemset  | 内存设置 |
|   free    |  cudaFree   | 释放内存 |



最关键的一步，这一步要走总线的

```cpp
cudaError_t cudaMemcpy(void * dst,const void * src,size_t count, cudaMemcpyKind kind)
```

这个函数是内存拷贝过程，可以完成以下几种过程（cudaMemcpyKind kind）

- cudaMemcpyHostToHost
- cudaMemcpyHostToDevice
- cudaMemcpyDeviceToHost
- cudaMemcpyDeviceToDevice

这四个过程的方向可以清楚的从字面上看出来，这里就不废话了，如果函数执行成功，则会返回 cudaSuccess 否则返回 cudaErrorMemoryAllocation。

使用下面这个指令可以将上面的错误代码翻译成详细信息：

```cpp
char* cudaGetErrorString(cudaError_t error)
```



两个向量的加法：

```cpp
/*
* https://github.com/Tony-Tan/CUDA_Freshman
* 3_sum_arrays
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include "freshman.h"


void sumArrays(float * a,float * b,float * res,const int size)
{
  for(int i=0;i<size;i+=4)
  {
    res[i]=a[i]+b[i];
    res[i+1]=a[i+1]+b[i+1];
    res[i+2]=a[i+2]+b[i+2];
    res[i+3]=a[i+3]+b[i+3];
  }
}

__global__ void sumArraysGPU(float*a,float*b,float*res)
{
  int i=threadIdx.x;
  res[i]=a[i]+b[i];
}
int main(int argc,char **argv)
{
  int dev = 0;
  cudaSetDevice(dev);

  int nElem=32;
  printf("Vector size:%d\n",nElem);
  int nByte=sizeof(float)*nElem;
  float *a_h=(float*)malloc(nByte);
  float *b_h=(float*)malloc(nByte);
  float *res_h=(float*)malloc(nByte);
  float *res_from_gpu_h=(float*)malloc(nByte);
  memset(res_h,0,nByte);
  memset(res_from_gpu_h,0,nByte);

  float *a_d,*b_d,*res_d;
  CHECK(cudaMalloc((float**)&a_d,nByte));
  CHECK(cudaMalloc((float**)&b_d,nByte));
  CHECK(cudaMalloc((float**)&res_d,nByte));

  initialData(a_h,nElem);
  initialData(b_h,nElem);

  CHECK(cudaMemcpy(a_d,a_h,nByte,cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(b_d,b_h,nByte,cudaMemcpyHostToDevice));

  dim3 block(nElem);
  dim3 grid(nElem/block.x);
  sumArraysGPU<<<grid,block>>>(a_d,b_d,res_d);
  printf("Execution configuration<<<%d,%d>>>\n",block.x,grid.x);

  CHECK(cudaMemcpy(res_from_gpu_h,res_d,nByte,cudaMemcpyDeviceToHost));
  sumArrays(a_h,b_h,res_h,nElem);

  checkResult(res_h,res_from_gpu_h,nElem);
  cudaFree(a_d);
  cudaFree(b_d);
  cudaFree(res_d);

  free(a_h);
  free(b_h);
  free(res_h);
  free(res_from_gpu_h);

  return 0;
}
```

使用nvcc编译程序

解释下内存管理部分的代码：

```cpp
cudaMalloc((float**)&a_d,nByte);
```

分配设备端的内存空间，为了区分设备和主机端内存，我们可以给变量加后缀或者前缀h_表示host，d_表示device

> 一个经常会发生的错误就是混用设备和主机的内存地址！！



## 2.1 CUDA编程模型概述（二）

#### 1 核函数启动

启动核函数，通过的以下的ANSI C扩展出的CUDA C指令：

```cpp
dim3 block(int);	// 内核中线程的数目
dim3 grid (int);	// 内核中使用的线程布局

// 使用dim3类型的grid维度和block维度配置内核
kernel_name <<<grid,block>>> (argument list);	

// 使用int类型的变量，或者常量直接初始化：
kernel_name<<<4,8>>>(argument list);	
```

这个三个尖括号’<<<grid,block>>>’内是对设备代码执行的线程结构的配置（或者简称为对内核进行配置，也就是线程结构中的网格块。

我们通过CUDA C内置的数据类型dim3类型的变量来配置grid和block（上文提到过：在设备端访问grid和block属性的数据类型是uint3不能修改的常类型结构，这里反复强调一下）。

我们的核函数是同时复制到多个线程执行的，上文我们说过一个对应问题，多个计算执行在一个数据，肯定是浪费时间，所以为了让多线程按照我们的意愿对应到不同的数据，就要<u>给线程一个唯一的标识</u>，由于设备内存是线性的（基本市面上的内存硬件都是线性形式存储数据的）

可以用threadIdx.x 和blockIdx.x 来组合获得对应的线程的唯一标识

改变核函数的配置，产生运行出结果一样，但效率不同的代码：

```cpp
// 下列代码如果没有特殊结构在核函数中，执行结果应该一致，但是有些效率会一直比较低。
kernel_name<<<1,32>>>(argument list);	// 1个快
kernel_name<<<32,1>>>(argument list);	// 32个快
```

上面这些是启动部分，当主机启动了核函数，控制权马上回到主机，而不是主机等待设备完成核函数的运行，这一点我们上一篇文章也有提到过（就是等待hello world输出的那段代码后面要加一句）

```cpp
cudaError_t cudaDeviceSynchronize(void);	// 主机等待设备端执行
```



#### 2 编写核函数

|   限定符   |    执行    |                     调用                      |           备注           |
| :--------: | :--------: | :-------------------------------------------: | :----------------------: |
| __global__ | 设备端执行 | 可以从主机调用也可以从计算能力3以上的设备调用 | 必须有一个void的返回类型 |
| __device__ | 设备端执行 |                  设备端调用                   |                          |
|  __host__  | 主机端执行 |                   主机调用                    |         可以省略         |

而且这里有个特殊的情况就是有些函数可以同时定义为 **device** 和 **host** ，这种函数可以同时被设备和主机端的代码调用，主机端代码调用函数很正常，设备端调用函数与C语言一致，但是要声明成设备端代码，告诉nvcc编译成设备机器码，同时声明主机端设备端函数，那么就要告诉编译器，生成两份不同设备的机器码。

Kernel核函数编写有以下限制

1. 只能访问设备内存
2. 必须有void返回类型
3. 不支持可变数量的参数
4. 不支持静态变量
5. 显示异步行为

并行程序中经常的一种现象：把串行代码并行化时对串行代码块for的操作，也就是<u>把for并行化</u>。

```cpp
/// 串行
void sumArraysOnHost(float *A, float *B, float *C, const int N) 
{
  for (int i = 0; i < N; i++)
  {
      C[i] = A[i] + B[i];
  }
}

/// 并行
__global__ void sumArraysOnGPU(float *A, float *B, float *C) 
{
  int i = threadIdx.x;
  C[i] = A[i] + B[i];
}
```



#### 3 验证核函数

进行调试时可以把核函数配置成单线程的

```cpp
kernel_name<<<1,1>>>(argument list)
```

#### 4 错误处理

```cpp
#define CHECK(call)\
{\  
	const cudaError_t error=call;\  
	if(error!=cudaSuccess)\  
	{\      
		printf("ERROR: %s:%d,",__FILE__,__LINE__);\
        printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\      
		exit(1);\  
	}\
}
```



## 2.2 给核函数计时



## 2.3 组织并行线程



