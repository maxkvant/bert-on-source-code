from typing import List


def compute_metrics(ground_truth: List[str], candidates: List[List[str]]):

    def get_precision(gt: List[str], gen: List[str]):
        gt = set(gt)
        return sum(tok in gt for tok in gen) / len(gen)

    def get_recall(gt: List[str], gen: List[str]):
        gen = set(gen)
        return sum(tok in gen for tok in gt) / len(gt)

    def get_f1(precision, recall):
        if (precision + recall) == 0:
            return 0
        return 2 * (precision * recall) / (precision + recall)

    best_candidate = candidates[0]

    precision_top_1 = get_precision(ground_truth, best_candidate)
    precision_top_5 = max(get_precision(ground_truth, c) for c in candidates)
    recall_top_1 = get_recall(ground_truth, best_candidate)
    recall_top_5 = max(get_recall(ground_truth, c) for c in candidates)

    return {
        'exact-top-1': int(ground_truth == best_candidate),
        'exact-top-5': int(any([ground_truth == c for c in candidates])),
        'precision-top-1': precision_top_1,
        'precision-top-5': precision_top_5,
        'recall-top-1': recall_top_1,
        'recall-top-5': recall_top_5,
        'f1-top1': get_f1(precision_top_1, recall_top_1),
        'f1-top5': get_f1(precision_top_5, recall_top_5)
    }
