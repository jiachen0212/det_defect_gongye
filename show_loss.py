# coidng=utf-8
# show loss 
import os
import matplotlib.pyplot as plt

cls0, cls1, cls2, acc0, acc1, acc2, losses = [], [], [], [], [], [], []
clss, box_loss = [], []

files = ['output_cascade4/20220819_043527.log']
lines = open(files[0]).readlines()
for line in lines:
    if "lr: " in line:
        epoch = line.split('Epoch ')[1][:7]
        lr_ = line.split('lr: ')[1][:9]
        
        # cascade
        loss = line.split('loss: ')[1][:6]     

        loss_rpn_bbox = line.split('loss_rpn_bbox: ')[1][:6]
        loss_rpn_cls = line.split('loss_rpn_cls: ')[1][:6]
        
        s0loss_cls = line.split('s0.loss_cls: ')[1][:6]
        s0acc = line.split('s0.acc: ')[1][:6]

        s1loss_cls = line.split('s1.loss_cls: ')[1][:6]
        s1acc = line.split('s1.acc: ')[1][:6]

        s2loss_cls = line.split('s2.loss_cls: ')[1][:6]
        s2acc = line.split('s2.acc: ')[1][:6]

        print('epoch: {}, lr: {}, loss_rpn_bbox: {}, loss_rpn_cls: {}, s0.loss_cls: {}, s0.acc: {}, s1.loss_cls: {}, s1.acc: {}, s2.loss_cls: {}, s2.acc: {}, loss: {}'.format(epoch, lr_, loss_rpn_bbox, loss_rpn_cls, s0loss_cls, s0acc, s1loss_cls, s1acc, s2loss_cls, s2acc, loss))
        cls0.append(float(s0loss_cls))
        cls1.append(float(s1loss_cls))
        cls2.append(float(s2loss_cls))
        acc0.append(float(s0acc))
        acc1.append(float(s1acc))
        acc2.append(float(s2acc))
        losses.append(float(loss))

lens = len(losses)
# x = [int(a*50/311) for a in range(lens)]
x = [a*50 for a in range(lens)]
plt.plot(x,losses)
plt.savefig('cascade4_loss.png')
plt.show()