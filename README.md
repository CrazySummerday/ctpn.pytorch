# ctpn.pytorch
Pytorch implementation of CTPN (Detecting Text in Natural Image with Connectionist Text Proposal Network)

# Paper
https://arxiv.org/pdf/1609.03605.pdf

# train
training dataset: ICDAR2013 and ICDAR2017.  
If you want to train your own dataset, you need to change the 'img_dir' and 'label_dir' in file *ctpn/config.py*, then run 
```
python train.py
```

![training loss](https://github.com/CrazySummerday/ctpn.pytorch/raw/master/log/training_loss.png) 


# predict
Download pretrained model from './weights/', change the test image path in file *predict.py*, then run:  
```
python predict.py
```
## result
![result_1](https://github.com/CrazySummerday/ctpn.pytorch/raw/master/log/1.jpg)  
![result_2](https://github.com/CrazySummerday/ctpn.pytorch/raw/master/log/2.jpg)  

# references
https://github.com/opconty/pytorch_ctpn  

https://github.com/courao/ocr.pytorch
