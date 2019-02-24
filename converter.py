import pandas as pd
import os
import os.path as osp

data_frame = pd.read_csv('train_set2.csv')

height = 480
width = 640
if not os.path.exists(osp.join(os.getcwd(), 'data/swish')):
    os.makedirs(osp.join(os.getcwd(), 'data/swish'))
if not os.path.exists(osp.join(os.getcwd(), 'data/swish/labels/')):
    os.makedirs(osp.join(os.getcwd(), 'data/swish/labels/'))
print(os.getcwd())
txt = open(osp.join(os.getcwd(), 'data/swish/train.txt'), 'w+')
for idx, name in enumerate(data_frame['image_name']):
    txt.write(osp.join(os.getcwd(), 'data/swish/images/', name) + '\n')
    labels = open(osp.join(os.getcwd(), 'data/swish/labels/', name.replace('png', 'txt')), 'w+')
    x1 = data_frame.iloc[idx, 1] / width
    x2 = data_frame.iloc[idx, 2] / width
    y1 = data_frame.iloc[idx, 3] / height
    y2 = data_frame.iloc[idx, 4] / height

    midx = (x1 + x2) / 2
    midy = (y1 + y2) / 2
    wid = x2 - x1
    hei = y2 - y1
    labels.write('{0} {1} {2} {3} {4}'.format(0, midx, midy, wid, hei))
    print('Wrote ' + name)
print('writing')
txt.close()
