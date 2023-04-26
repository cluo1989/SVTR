# coding: utf-8
# ref:
# https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/metrics/rec_metric.py
from rapidfuzz.distance import Levenshtein
import torch
import string


class RecMetric(object):
    def __init__(self,
                 main_indicator='acc',
                 is_filter=False,
                 ignore_space=True,
                 **kwargs):
        self.main_indicator = main_indicator
        self.is_filter = is_filter
        self.ignore_space = ignore_space
        self.eps = 1e-5
        self.reset()

    def _normalize_text(self, text):
        text = ''.join(
            filter(lambda x: x in (string.digits + string.ascii_letters), text))
        return text.lower()

    def __call__(self, preds, labels, *args, **kwargs):
        with torch.no_grad():
            correct_num = 0
            all_num = 0
            norm_edit_dis = 0.0
            # for (pred, pred_conf), (target, _) in zip(preds, labels):
            for (pred, target) in zip(preds, labels):
                if self.ignore_space:
                    pred = pred.replace(" ", "")
                    target = target.replace(" ", "")
                if self.is_filter:
                    pred = self._normalize_text(pred)
                    target = self._normalize_text(target)
                norm_edit_dis += Levenshtein.normalized_distance(pred, target)
                if pred == target:
                    correct_num += 1
                all_num += 1
            self.correct_num += correct_num
            self.all_num += all_num
            self.norm_edit_dis += norm_edit_dis
            return {
                'acc': correct_num / (all_num + self.eps),
                'norm_edit_dis': 1 - norm_edit_dis / (all_num + self.eps)
            }

    def get_metric(self):
        """
        return metrics {
                 'acc': 0,
                 'norm_edit_dis': 0,
            }
        """
        acc = 1.0 * self.correct_num / (self.all_num + self.eps)
        norm_edit_dis = 1 - self.norm_edit_dis / (self.all_num + self.eps)
        self.reset()
        return {'acc': acc, 'norm_edit_dis': norm_edit_dis}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0


if __name__ == '__main__':
    pred = ['我们这里', '天气真好', '世界和平']
    label = ['我们这里', '天气真好', '世界合平']
    accuracy = RecMetric()
    print(accuracy(pred, label))