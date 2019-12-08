# AutoKeras-Dogs-vs-Cats
The detailed comments are in chinese:
https://blog.csdn.net/lvsolo/article/details/103445431

    environment:
    autokeras 0.4.0;
    torch 1.3.1;
    cuda10.0;
    cudnn 7.5.1;
    gpu rtx2070
The main process flow is as follows:

    1)Preprocess the datasets,restore into numpy file, preventing the repeated loading work;
    2)Use AutoKeras to choose the model for the probloms
    3)Use the pytorch to continuely train the model
    4)Generate the result csv for kaggle.

之前做过一些简单的深度学习项目，在我看来主要是一些调包工程师的工作，应用现有的模型对一些项目进行训练。初入kaggle，打算以最简单的项目为切入点，提升自己的姿势水平。
	
	环境：autokeras 0.4.0;
	torch 1.3.1;
	cuda10.0;
	cudnn 7.5.1;
	gpu rtx2070
本文记录了这一项目进行的主要逻辑流程，主要步骤如下：
	
	1）数据预处理，比较简单，reshape并保存为numpy的格式存储
	2）使用autokeras进行模型的初筛，通过短时间的预训练搜索出较为合适的模型
	3）用pytorch加载现有的预训练模型，进行进一步的训练
	4）对测试数据进行预测生成csv文件，上传
