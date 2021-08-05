import numpy as np


class BinaryIOU(object):
    def __init__(self):
        self.inter = dict()
        self.union = dict()
        self.cls_iou = None

    def _update(self, label_true, label_pred, cls_id):
        combination = (label_true + label_pred)
        inter = np.sum(combination == 2, dtype=np.float)
        union = np.sum((combination == 1) + (combination == 2), dtype=np.float)
        if cls_id not in self.inter:
            self.inter[cls_id] = inter
        else:
            self.inter[cls_id] += inter

        if cls_id not in self.union:
            self.union[cls_id] = union
        else:
            self.union[cls_id] += union

    def update(self, label_trues, label_pred, cls_ids):
        label_pred = label_pred.data.max(1)[1].cpu().numpy()
        label_trues = label_trues.data.cpu().numpy()
        for label_true, label_pred, cls_id in zip(label_trues, label_pred, cls_ids):
            # ignore when the mask does not contain any foreground objects
            if np.sum(label_true) == 0:
                continue
            self._update(label_true, label_pred, cls_id)

    def mean_iou(self):
        self.cls_iou = {cls_id: self.inter[cls_id] / self.union[cls_id]
                        for cls_id in self.inter.keys()}
        return float(np.mean(list(self.cls_iou.values())))

    def class_iou(self):
        if self.cls_iou is None:
            self.mean_iou()
        return self.cls_iou

    def mean_bf_iou(self):
        """
        Treat all foreground classes as the same classes,
        and calculate the mean iou.
        """
        inter_sum = sum([self.inter[cls_id] for cls_id in self.inter.keys()])
        union_sum = sum([self.union[cls_id] for cls_id in self.union.keys()])
        return inter_sum / union_sum


class FullIOU(object):
    def __init__(self, num_classes=21):
        self.num_classes = num_classes
        self.mat = None
        self.cls_iou = None

    def update(self, label_trues, label_preds):
        label_trues = label_trues.flatten().cpu().numpy()
        label_preds = label_preds.argmax(1).flatten().cpu().numpy()
        n = self.num_classes
        if self.mat is None:
            self.mat = np.zeros((n, n))
        k = (label_trues >= 0) & (label_trues < n)
        inds = n * label_trues[k].astype(int) + label_preds[k]
        self.mat += np.bincount(inds, minlength=n ** 2).reshape(n, n)

    def mean_iou(self, eps=1e-8):
        hist = self.mat.astype(float)
        numerator = np.diag(hist)
        denominator = hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist)
        denominator = np.clip(denominator, eps, np.inf)
        iu = numerator / denominator
        self.cls_iou = {cls_id: iou for cls_id, iou in enumerate(iu)}
        return np.nanmean(iu)

    def class_iou(self):
        if self.cls_iou is None:
            self.mean_iou()
        return self.cls_iou

    def mean_subclasses_iou(self, subclasses):
        mean = [v if k in subclasses else 0 for k, v in self.class_iou().items()]
        return sum(mean) / len(mean)

    def subclasses_iou(self, subclasses):
        return {k: v for k, v in self.class_iou().items() if k in subclasses}
