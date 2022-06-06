# SimpleTransferLearning-Pytorch-master

The purpose of the project is to show beginners how to use Pytorch for simple transfer learning by my own dataset.

Detailed explanation has been published on CSDN and Quora(Chinese) Zhihu.

[CSDN](https://blog.csdn.net/XiaoyYidiaodiao/article/details/125127107?spm=1001.2014.3001.5501)

[Quora(Chinese)Zhihu](https://zhuanlan.zhihu.com/p/522597095)


This project uses vgg16 (image_size as 64 * 64 * 3 because of my computer) as Backbone, CIFAR10 as pretrained dataset, pets (my own designed dataset), AdamW as gradient descent strategy, and ReduceLROnPlateau as learning adjustment mechanism.

I used my own computer Lenovo Saver, besides GPU is 2060.

You should create your own **D:\data\CIFAR10** or **D:\data\pet**(dataset), **checkpoint**(model save), **log**, and **tenshorboard**(loss visualization) file package.

## The file structure of the project

```
D:
|
|
|
|----data|----CIFAR10
         |----pet(my own designed dataset, if you need this dataset, you can contact me by CSDN or Zhihu.)
       
D:
|
|
|
|----PycharmProject----SimpleTransferLearning-Pytorch-master
                            |----tensorboard(args.tensorboard=True, visualization loss)
                            |----log(classification_log)
                            |----checkpoints(save model pretrained:CIFAR10_vgg16.pth, the model for training my own dataset: pets_vgg16.pth)
                            |----models
                            |       |----__init__.py
                            |       |----vgg16.py
                            |
                            |----tool----classification
                            |                  |----train.py
                            |----utils
                                   |----get_logger.py(log)
                                   |----path.py(path)
                                   |----AverageMeter.py(AP)
                                   |----accuracy.py
                          
```

## Before transfer learning

we should run `python tool\classification\train.py`.
And the file `train.py` maybe need to be modified.
```
1.
    parser.add_argument('--dataset',
                        type=str,
                        default='CIFAR10',
                        choices=['CIFAR10', 'pets'],
                        help=' CIFAR10 or pets')
```

```
2.
    parser.add_argument('--dataset_root',
                        type=str,
                        default=CIFAR_path,
                        choices=[CIFAR_path, PETS_path],
                        help='Dataset root directory path')
```

```
3.
    parser.add_argument('--pretrained',
                        type=str,
                        default=False,
                        help='Checkpoint state_dict file to resume training from')
```

```
4.
    parser.add_argument('--epochs',
                        type=int,
                        default=20,
                        help='Number of epochs')
```

```
5.
    parser.add_argument('--num_classes',
                        type=int,
                        default=10,
                        help='the number classes')
```

```
6.
    acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
```

```
7.
    return top1.avg, top5.avg, losses.avg
```

### results

|AP|96.88|
|:---:|:---:|


## transfer learning
In `models\vgg16.py`.
module.classifier is 
```
self.classifier = nn.Linear(4096, num_classes)
```

this is because there are ten categories of dataset for CIAFAR10, but there are two categories of dataset for my own designed.

we can modify `models\vgg16.py`.

```
del pretrained_models['module.classifier.7.bias']
```
Then, we should run `python tool\classification\train.py`.
And the file `train.py` maybe need to be modified.
```
1.
    parser.add_argument('--dataset',
                        type=str,
                        default='pets',
                        choices=['CIFAR10', 'pets'],
                        help=' CIFAR10 or pets')
```

```
2.
    parser.add_argument('--dataset_root',
                        type=str,
                        default=PETS_path,
                        choices=[CIFAR_path, PETS_path],
                        help='Dataset root directory path')
```

```
3.
    parser.add_argument('--pretrained',
                        type=str,
                        default=True,
                        help='Checkpoint state_dict file to resume training from')
```

```
4.
    parser.add_argument('--epochs',
                        type=int,
                        default=200,
                        help='Number of epochs')
```

```
5.
    parser.add_argument('--num_classes',
                        type=int,
                        default=2,
                        help='the number classes')
```

```
6.
    acc1, _ = accuracy(outputs, targets, topk=(1, 1))
```

```
7.
    return top1.avg, losses.avg
```

```
8.
    a) optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    or
    
    b) optimizer = optim.AdamW(model.module.classifier.parameters(), lr=args.lr)
```


### results

||8.a|8.b|
|:---:|:---:|:---:|
|AP|93.48|73.91|

