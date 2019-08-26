# ctpn.pytorch
Pytorch implementation of CTPN (Detecting Text in Natural Image with Connectionist Text Proposal Network)

# Paper
https://arxiv.org/pdf/1609.03605.pdf

# train
training dataset: ICDAR2013 adn ICDAR2017
run   
'''
python train.py
'''

![training loss](https://github.com/CrazySummerday/ctpn.pytorch/tree/master/log/training_loss.png) 


# predict
Download pretrained model from './weights/', change the test image path in file predict.py, then run:  
'''
python predict.py
'''

# references
https://github.com/opconty/pytorch_ctpn  
https://github.com/courao/ocr.pytorch
