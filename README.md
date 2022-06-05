# SimpleTransferLearning-Pytorch-master

The purpose of the project is to show beginners how to use Pytorch for simple transfer learning by my own dataset.

Detailed explanation has been published on CSDN and Quora(Chinese) Zhihu.

[CSDN](https://blog.csdn.net/XiaoyYidiaodiao/article/details/125127107?spm=1001.2014.3001.5501)

[Quora(Chinese)Zhihu](https://zhuanlan.zhihu.com/p/522597095)


This project uses vgg16 as Backbone, CIFAR10 as pretrained dataset, pets (my own designed dataset), AdamW as gradient descent strategy, and ReduceLROnPlateau as learning adjustment mechanism.

I used my own computer Lenovo Saver, besides GPU is 2060.

## The file structure of the project

```
D:
|
|
|
|----data
       |
       |
       |
       |----CIFAR10
       |
       |
       |
       |----pet(my own designed dataset, if you need this dataset, you can contact me by CSDN or Zhihu.)
       
D:
|
|
|
|----PycharmProject----SimpleTransferLearning-Pytorch-master
                        |
                        |
                        |
                        |----tensorboard(args.tensorboard=True, visualization loss)
                        |
                        |
                        |
                        |----log(classification_log)
                        |
                        |
                        |
                        |----checkpoints(save model pretrained:CIFAR10_vgg16.pth, the model for training my own dataset: pets_vgg16.pth)
                        |         |
                        |         |
                        |         |
                        |         |----__init__.py
                        |         |
                        |         |
                        |         |
                        |         |----vgg16.py
                        |         
                        |
                        |
                        |----tool----classification
                        |                  |
                        |                  |
                        |                  |
                        |                  |----train.py
                        |
                        |
                        |
                        |----utils
                               |
                               |
                               |
                               |----get_logger.py(log)
                               |
                               |
                               |
                               |----path.py(path)
                               |
                               |
                               |
                               |----AverageMeter.py(AP)
                               |
                               |
                               |
                               |----accuracy.py
                          
```

## Before transfer learning

## transfer learning
