"""Exact accelerated accumulation of frozen BangKhop matches, including score ties.

Matching, IoUs, categories, maxDets and 101 recall thresholds remain unchanged.
Groups preserve original ordering: score, image, replica, within-image detection.
"""
import numpy as np
from numba import njit


@njit(cache=True, nogil=True)
def category_ap(starts, ends, images, hits, ignored, gt_counts, counts, thresholds):
    ngt = 0
    for i in range(len(counts)):
        ngt += gt_counts[i] * counts[i]
    if ngt == 0:
        return np.full(10, np.nan)
    answer = np.zeros(10)
    for t in range(10):
        tp, fp = 0, 0
        for group in range(len(starts)):
            mult = counts[images[group]]
            for j in range(starts[group], ends[group]):
                if not ignored[t, j]:
                    if hits[t, j]:
                        tp += mult
                    else:
                        fp += mult
        recall_index = 100
        while recall_index > 0 and thresholds[recall_index] > tp / ngt:
            recall_index -= 1
        best, total, any_detection = 0.0, 0.0, False
        for group in range(len(starts) - 1, -1, -1):
            mult = counts[images[group]]
            if not mult:
                continue
            group_tp, group_fp = 0, 0
            for j in range(starts[group], ends[group]):
                if not ignored[t, j]:
                    if hits[t, j]:
                        group_tp += 1
                    else:
                        group_fp += 1
            any_detection = True
            # A precision-envelope maximum can only occur at a true positive.
            # Skip entire false-positive/ignored groups without expanding replicas.
            if not group_tp:
                fp -= group_fp * mult
                continue
            for replica in range(counts[images[group]]):
                for j in range(ends[group] - 1, starts[group] - 1, -1):
                    if not ignored[t, j]:
                        if hits[t, j]:
                            precision = tp / (tp + fp + np.spacing(1.0))
                            best = max(best, precision)
                            previous_recall = (tp - 1) / ngt
                            while recall_index > 0 and thresholds[recall_index] > previous_recall:
                                total += best
                                recall_index -= 1
                            tp -= 1
                        else:
                            fp -= 1
        if any_detection:
            total += best
        answer[t] = total / 101
    return answer


class FastAP:
    def __init__(self, original):
        self.img_ids = original.img_ids
        self.categories = []
        self.thresholds = np.linspace(0, 1, 101)
        for k in range(len(original.cat_ids)):
            scores, hits, ignored, image_ids = [], [], [], []
            gt = np.zeros(len(self.img_ids), dtype=np.int64)
            for i in range(len(self.img_ids)):
                data = original.du_lieu.get((k, i))
                if data is None:
                    continue
                score, hit, ignore, ngt = data
                gt[i] = ngt
                scores.extend(score)
                image_ids.extend([i] * len(score))
                hits.append(hit)
                ignored.append(ignore)
            if scores:
                score = np.asarray(scores)
                order = np.argsort(-score, kind='mergesort')
                score, img = score[order], np.asarray(image_ids, dtype=np.int64)[order]
                hit = np.ascontiguousarray(np.concatenate(hits, axis=1)[:, order])
                ignore = np.ascontiguousarray(np.concatenate(ignored, axis=1)[:, order])
                start = np.flatnonzero(np.r_[True, (score[1:] != score[:-1]) | (img[1:] != img[:-1])]).astype(np.int64)
                end = np.r_[start[1:], len(score)].astype(np.int64)
                group_images = img[start]
            else:
                start = end = group_images = np.empty(0, dtype=np.int64)
                hit = ignore = np.empty((10, 0), dtype=np.bool_)
            self.categories.append((start, end, group_images, hit, ignore, gt))

    def ap(self, indices=None):
        counts = np.ones(len(self.img_ids), dtype=np.int64) if indices is None else np.bincount(indices, minlength=len(self.img_ids)).astype(np.int64)
        values = np.asarray([category_ap(*cat, counts, self.thresholds) for cat in self.categories])
        return float(100 * np.nanmean(values)), float(100 * np.nanmean(values[:, 0]))
