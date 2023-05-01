import numpy as np
import tqdm
import matplotlib.pyplot as plt
from numba import jit

import torch

# utils


@jit
def contained(tb, pb):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(tb[0], pb[0])
    yA = max(tb[1], pb[1])
    xB = min(tb[2], pb[2])
    yB = min(tb[3], pb[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    tbArea = (tb[2] - tb[0] + 1) * (tb[3] - tb[1] + 1)
    # boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(tbArea)  # + boxBArea - interArea)
    # return the intersection over union value
    return iou


# JIT doesnt work here, because of list ?
# @jit
def f_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


# @jit
def mesh_func(x1, y1, x2, y2, pbs):
    xx, yy = torch.meshgrid(
        torch.linspace(x1, x2, x2 - x1 + 1).cuda(),
        torch.linspace(y1, y2, y2 - y1 + 1).cuda(),
        indexing="xy",  # Unsure, "xy" should be default numpy behaviour, see https://github.com/pytorch/pytorch/issues/50276
    )
    # print(pbs.shape, pbs)
    outxx = (xx.reshape((1, -1)) >= pbs[:, 0, None]) & (
        xx.reshape((1, -1)) <= (pbs[:, 2, None])
    )
    outyy = (yy.reshape((1, -1)) >= pbs[:, 1, None]) & (
        yy.reshape((1, -1)) <= (pbs[:, 3, None])
    )

    Z = torch.any(outxx & outyy, axis=0).reshape((x2 - x1 + 1, y2 - y1 + 1))
    return Z


def getStretch(pred_boxes, conf_boxes):
    stretches = []
    area = lambda x: (x[:, 2] - x[:, 0] + 1) * (x[:, 3] - x[:, 1] + 1)
    for i in range(len(pred_boxes)):
        stretches.append(area(conf_boxes[i]) / area(pred_boxes[i]))
    return torch.cat(stretches).mean()


# @jit
def compute_risk_image_wise(
    true_boxes,
    conf_boxes,
    loss="recall",
    B=1,
    return_original_risk=False,
    beta=0.25,
):
    # right now just a default risk: TODO generalize
    risk = []
    for i in range(len(true_boxes)):
        tbs = true_boxes[i]
        current_boxes = conf_boxes[i]
        if len(tbs) == 0:
            risk.append(0)
            continue
        elif len(current_boxes) == 0:
            risk.append(1)
            continue
        # pbs = torch.stack(list(map(lambda x: torch.FloatTensor(x[0]), current_boxes))).cuda()
        pbs = current_boxes
        areas = []

        for tb in tbs:
            p1, p2 = tb["points"]
            (x1, y1), (x2, y2) = tb["points"]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            Z = mesh_func(x1, y1, x2, y2, pbs)

            area = Z.sum().detach().cpu().item() / ((x2 - x1 + 1) * (y2 - y1 + 1))
            areas.append(area)

        if loss == "degrancey":
            rsk = np.array(areas) < 0.99  # due to potential problem with the +1 above ?
            rsk = np.mean(rsk)
            rsk = 1 if rsk > beta else 0
            risk.append(rsk)
        elif loss == "boxes":
            rsk = np.array(areas) < 0.99  # due to potential problem with the +1 above ?
            rsk = np.mean(rsk)
            risk.append(rsk)
        elif loss == "recall":
            risk.append(1 - np.mean(areas))
        else:
            raise ValueError(f"Loss {loss} not accepted")

    n = len(risk)
    corrected_risk = (n / (n + 1)) * np.mean(risk) + B / (n + 1)
    if return_original_risk:
        return np.mean(risk), corrected_risk
    else:
        return corrected_risk


def get_recall_precision(
    images,
    true_boxes,
    pred_boxes,
    scores,
    IOU_THRESHOLD=0.5,
    SCORE_THRESHOLD=0.5,
    verbose=True,
    replace_iou=None,
):
    recalls = []
    precisions = []
    my_scores = []
    # print("first call")
    for i in tqdm.tqdm(range(len(images)), disable=not verbose):
        tbs = true_boxes[i]
        pbs = pred_boxes[i]
        batch_scores = scores[i]
        # print(len(tbs), pbs.shape, batch_scores.shape)
        # print(SCORE_THRESHOLD)
        pbs = pbs[batch_scores >= SCORE_THRESHOLD]

        already_assigned = []

        tp = 0
        # print("image")
        for tb in tbs:
            # convert format
            p1, p2 = tb["points"]
            tb = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=float)

            my_score = 0

            for k, pb in enumerate(pbs):
                if k in already_assigned:
                    continue
                # IoU
                if replace_iou is not None:
                    iou = replace_iou(tb, pb.detach().cpu().numpy())
                else:
                    # TODO : redo in torch/batch version
                    # print(tb)
                    # print(pb)
                    iou = f_iou(tb, pb.detach().cpu().numpy())
                if iou > IOU_THRESHOLD:
                    already_assigned.append(k)
                    tp += 1
                    my_score = iou
                    my_scores.append(my_score)
                    break

        nb_preds = len(pbs)
        nb_true = len(tbs)
        recall = tp / nb_true if nb_true > 0 else 1
        if nb_preds > 0:
            precision = tp / nb_preds
        else:
            precision = 1

        recalls.append(recall)
        precisions.append(precision)

    if verbose:
        print(
            f"Average Recall = {np.mean(recalls)}, Average Precision = {np.mean(precisions)}"
        )
    return recalls, precisions, my_scores


# ! BOXWISE !
def getCoverage(
    images,
    true_boxes,
    pred_boxes,
    conf_boxes,
    scores,
    verbose=True,
    objectness_threshold=0.5,
    iou_threshold=0.5,
):
    rec1, prec1, ious1 = get_recall_precision(
        images,
        true_boxes,
        pred_boxes,  # PREDICTIONS
        scores,
        IOU_THRESHOLD=iou_threshold,
        SCORE_THRESHOLD=objectness_threshold,
        verbose=False,
    )
    # passer index des vrais positifs et tester uniquement sur ces indexs
    rec2, prec2, ious2 = get_recall_precision(
        images,
        true_boxes,
        conf_boxes,  # CONFORMAL BOXES
        scores,
        IOU_THRESHOLD=0.99,
        SCORE_THRESHOLD=objectness_threshold,
        verbose=False,
        replace_iou=contained,
    )

    # #(Conf boxes that contain truth entirely) / #(nb of true positives)
    return len(ious2) / len(ious1)


def getRisk(
    images,
    true_boxes,
    pred_boxes,
    scores,
    loss="recall",  # lambda x, y: 1 - contained(x, y),
    objectness_threshold=0.3,
    verbose=True,
):
    # todo : add verbose
    pred_boxes = list(
        [x[y >= objectness_threshold] for x, y in zip(pred_boxes, scores)]
    )

    risk, corrected_risk = compute_risk_image_wise(
        true_boxes, pred_boxes, loss=loss, B=1, return_original_risk=True
    )
    # if verbose:
    #    print("Average risk =", risk)
    return risk


def getAveragePrecision(
    images, true_boxes, pred_boxes, scores, verbose=True, iou_threshold=0.3
):
    # TODO Mettre dans une fonction
    total_recalls = []
    total_precisions = []
    # predictor.input_format = 'BGR'
    # predictor.input_format = 'RGB' # default
    threshes_objectness = np.linspace(0, 1, 40)
    pbar = tqdm.tqdm(threshes_objectness, disable=not verbose)
    for thresh in pbar:
        tmp_recalls, tmp_precisions, _ = get_recall_precision(
            images,
            true_boxes,
            pred_boxes,
            scores,
            IOU_THRESHOLD=iou_threshold,
            SCORE_THRESHOLD=thresh,
            verbose=False,
        )
        pbar.set_description(
            f"Average Recall = {np.mean(tmp_recalls)}, Average Precision = {np.mean(tmp_precisions)}"
        )
        total_recalls.append(np.mean(tmp_recalls))
        total_precisions.append(np.mean(tmp_precisions))

    AP = np.trapz(x=list(reversed(total_recalls)), y=list(reversed(total_precisions)))
    return AP, total_recalls, total_precisions, threshes_objectness


# plot recall and precisions given objectness threshold OR given IoU threshold
def plot_recall_precision(total_recalls, total_precisions, threshes_objectness):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(threshes_objectness, total_recalls, label="Recall")
    ax1.plot(threshes_objectness, total_precisions, label="Precision")
    ax1.xlabel("Objectness score threshold")
    ax2.plot(total_recalls, total_precisions)
    ax2.xlabel("Recall")
    ax2.ylabel("Precision")
    plt.legend()
    plt.show()
