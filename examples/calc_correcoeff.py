import pickle 
import numpy as np
import matplotlib.pyplot as plt

filename = '/home/yainoue/meg2image/results/20230421_sbj01_kamitani_regression/sbj01-0.25_0.45-0.531/ridge_regression.pkl'

with open(filename, 'rb') as f:
    data = pickle.load(f)

def get_averaged_feature(pred_y, true_y, labels):
    '''Return category-averaged features'''

    labels_set = np.unique(labels)

    pred_y_av = np.array([np.mean(pred_y[labels == c, :], axis=0) for c in labels_set])
    true_y_av = np.array([np.mean(true_y[labels == c, :], axis=0) for c in labels_set])

    return pred_y_av, true_y_av, labels_set


preds = data['pred_y']
gts = data['true_y']
labels = data['test_label']

preds, gts, labels = get_averaged_feature(preds, gts, labels)
print(gts.shape, preds.shape)
corr_list = []
for pred, gt in zip(preds, gts):
    print(pred.shape, gt.shape)
    src = np.stack([pred, gt], axis=0)
    R = np.corrcoef(src)
    print(R[0,1])
    corr_list.append(R[0,1])
print(len(corr_list))
print(np.mean(corr_list))

def clip_sigma(data):
    std = data.std(-1)
    mean = data.mean(-1)
    data[data > mean + 3*std] = mean + 3*std
    data[data < mean - 3*std] = mean - 3*std
    return data

ch=0
gts[ch] = clip_sigma(gts[ch])
preds[ch] = clip_sigma(preds[ch])
fig, axes = plt.subplots(ncols=2, figsize=(12,6))
axes[0].plot(np.arange(len(preds[ch])), gts[ch], label='gt')
axes[0].plot(np.arange(len(preds[ch])), preds[ch], label='pred')
axes[0].legend()
axes[0].set_title('corr coef: {:.3f}'.format(np.corrcoef(preds[ch], gts[ch])[0,1]))
axes[1].scatter(gts[ch], preds[ch])
axes[1].set_xlabel('gt')
axes[1].set_ylabel('pred')
plt.savefig('/home/yainoue/meg2image/results/20230421_sbj01_kamitani_regression/calc.png')
import pdb; pdb.set_trace()