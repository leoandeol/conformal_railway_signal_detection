import numpy as np
import tqdm
from numba import jit
import matplotlib.pyplot as plt

import torch

from confvision.evaluate import f_iou, contained, compute_risk_image_wise


def _old_apply_margins(pred_boxes, Qs, mode="additive"):
    new_boxes = []
    n = len(pred_boxes)
    for i in range(n):
        n2 = len(pred_boxes[i])
        new_boxes.append([None] * n2)
        for j in range(n2):
            if mode == "additive":
                new_boxes[i][j] = [
                    pred_boxes[i][j][0] + np.multiply([-1, -1, 1, 1], Qs),
                    pred_boxes[i][j][1],
                ]
            elif mode == "multiplicative":
                pb = pred_boxes[i][j][0]
                x1, y1, x2, y2 = pb
                w, h = x2 - x1, y2 - y1
                new_boxes[i][j] = [
                    pred_boxes[i][j][0] + np.multiply([-w, -h, w, h], Qs),
                    pred_boxes[i][j][1],
                ]
    return new_boxes


def apply_margins(pred_boxes, Qs, mode="additive"):
    n = len(pred_boxes)
    new_boxes = [None] * n
    Qst = torch.FloatTensor([Qs]).cuda()
    for i in range(n):
        if mode == "additive":
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.FloatTensor([[-1, -1, 1, 1]]).cuda(), Qst
            )
        elif mode == "multiplicative":
            w = pred_boxes[i][:, 2] - pred_boxes[i][:, 0]
            h = pred_boxes[i][:, 3] - pred_boxes[i][:, 1]
            new_boxes[i] = pred_boxes[i] + torch.mul(
                torch.stack((-w, -h, w, h), axis=-1), Qst
            )
    return new_boxes


def conformalize_preds(
    images,
    true_boxes,
    pred_boxes,
    pred_objs,
    IOU_THRESHOLD=0.5,
    objectness_threshold=0.5,
    alpha=0.1,
    tqdm_on=True,
    replace_iou=None,
    method="additive",
    coordinate_wise=False,
):
    n = len(images)

    Rxmins = []
    Rymins = []
    Rxmaxs = []
    Rymaxs = []
    for i in tqdm.tqdm(range(n), disable=not tqdm_on):
        tbs = true_boxes[i]
        pbs = pred_boxes[i]
        scores = pred_objs[i]
        pbs = pbs[scores >= objectness_threshold]

        already_assigned = []

        tp = 0

        for tb in tbs:
            # convert format
            p1, p2 = tb["points"]
            tb = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=float)

            for k, pb in enumerate(pbs):
                if k in already_assigned:
                    continue
                # IoU
                if replace_iou is not None:
                    iou = replace_iou(tb, pb)
                else:
                    iou = f_iou(tb, pb)
                if iou >= IOU_THRESHOLD:
                    already_assigned.append(k)
                    tp += 1

                    if method == "additive":
                        Rxmin = pb[0] - tb[0]
                        Rymin = pb[1] - tb[1]
                        Rxmax = tb[2] - pb[2]
                        Rymax = tb[3] - pb[3]
                    elif method == "multiplicative":
                        w, h = pb[2] - pb[0], pb[3] - pb[1]
                        Rxmin = (pb[0] - tb[0]) / w
                        Rymin = (pb[1] - tb[1]) / h
                        Rxmax = (tb[2] - pb[2]) / w
                        Rymax = (tb[3] - pb[3]) / h

                    Rxmins.append(Rxmin.cpu().item())
                    Rymins.append(Rymin.cpu().item())
                    Rxmaxs.append(Rxmax.cpu().item())
                    Rymaxs.append(Rymax.cpu().item())

                    break

    if coordinate_wise:
        real_alpha = alpha / 4
        qxmin = np.quantile(
            Rxmins, (1 - real_alpha) * (n + 1) / n, method="inverted_cdf"
        )
        qymin = np.quantile(
            Rymins, (1 - real_alpha) * (n + 1) / n, method="inverted_cdf"
        )
        qxmax = np.quantile(
            Rxmaxs, (1 - real_alpha) * (n + 1) / n, method="inverted_cdf"
        )
        qymax = np.quantile(
            Rymaxs, (1 - real_alpha) * (n + 1) / n, method="inverted_cdf"
        )
        if method == "additive":
            Qs = [qxmin, qymin, qxmax, qymax]
        elif method == "multiplicative":
            Qs = [qxmin, qymin, qxmax, qymax]
        else:
            raise ValueError(f"mode unknown {method}")
    else:
        R = np.max(np.stack((Rxmins, Rymins, Rxmaxs, Rymaxs)), axis=0)
        # print(R.shape)
        q = np.quantile(R, (1 - alpha) * (n + 1) / n, method="inverted_cdf")
        if method == "additive":
            Qs = [q] * 4
        elif method == "multiplicative":
            Qs = [q] * 4
        else:
            raise ValueError(f"mode unknown {method}")
    return (
        Qs,
        apply_margins(pred_boxes, Qs, mode=method),
        [Rxmins, Rymins, Rxmaxs, Rymaxs],
    )


class Conformalizer:
    def __init__(self, mode="box", method="additive", coordinate_wise=False):
        # self.alpha = alpha
        self.mode = mode
        if mode not in ["box", "image"]:
            raise ValueError(f"mode '{mode}' not accepted")
        self.method = method
        if method not in ["additive", "multiplicative"]:
            raise ValueError(f"method '{method}' not accepted")
        self.margin = None
        self.coordinate_wise = coordinate_wise

    def calibrate(self, preds, alpha=0.1, objectness_threshold=0.5, iou_threshold=0.5):
        print(f"Calibrating with alpha={alpha}")
        if self.margin is not None:
            print("Replacing previously computed lambda")
        Qs, conf_boxes, residuals = conformalize_preds(
            preds.images,
            preds.true_boxes,
            preds.pred_boxes,
            preds.scores,
            alpha=alpha,
            method=self.method,
            objectness_threshold=objectness_threshold,
            IOU_THRESHOLD=iou_threshold,
            coordinate_wise=self.coordinate_wise,
        )
        self.residuals = residuals
        self.margin = Qs
        print("Obtained margin =", self.margin)
        preds.set_conf_boxes(conf_boxes)
        return Qs, conf_boxes

    def conformalize(self, preds):
        conf_boxes = apply_margins(
            preds.pred_boxes,
            self.margin,
            mode=self.method,
        )
        preds.set_conf_boxes(conf_boxes)
        return conf_boxes

    def plot_residuals(self):
        for residual in self.residuals:
            plt.hist(residual)


class RiskConformalizer:
    def __init__(self, method="additive", loss="recall"):
        self.lbd = None
        self.method = method
        self.loss = loss
        if method not in ["additive", "multiplicative"]:
            raise ValueError(f"method '{method}' not accepted")

    def calibrate(self, preds, alpha=0.1, objectness_threshold=0.5, depth=13):
        if self.lbd is not None:
            print("Replacing previously computed lambda")
        lbd, conf_boxes = conformalize_risk_preds(
            preds.images,
            preds.true_boxes,
            preds.pred_boxes,
            preds.scores,
            alpha=alpha,
            mode=self.method,
            objectness_threshold=objectness_threshold,
            depth=depth,
            loss=self.loss,
        )
        self.lbd = lbd
        preds.set_conf_boxes(conf_boxes)
        return lbd, conf_boxes

    def conformalize(self, preds):
        conf_boxes = apply_margins(preds.pred_boxes, [self.lbd] * 4, mode=self.method)
        preds.set_conf_boxes(conf_boxes)
        return conf_boxes


def conformalize_risk_preds(
    images,
    true_boxes,
    pred_boxes,
    pred_objs,
    loss="recall",  # lambda x, y: 1 - contained(x, y),
    alpha=0.1,
    mode="additive",
    tqdm_on=True,
    replace_iou=None,
    objectness_threshold=0.3,
    depth=13,
):
    left, right = 0, 1000
    nb_iters = depth

    B = 1

    pbar = tqdm.tqdm(range(nb_iters), disable=not tqdm_on)

    pred_boxes_filtered = list(
        [x[y >= objectness_threshold] for x, y in zip(pred_boxes, pred_objs)]
    )
    # pred_boxes = list([list([x for x in ls if x[1]>=objectness_threshold]) for ls in pred_boxes])

    for k in pbar:
        lbd = (left + right) / 2

        conf_boxes = apply_margins(pred_boxes_filtered, [lbd, lbd, lbd, lbd], mode=mode)

        corrected_risk = compute_risk_image_wise(true_boxes, conf_boxes, loss=loss, B=B)

        pbar.set_description(
            f"[{left:.2f}, {right:.2f}] -> {lbd:.2f}. Corrected Risk = {corrected_risk:.2f}"
        )
        if corrected_risk <= alpha:
            right = lbd
        else:
            left = lbd

    lbd = (left + right) / 2
    conf_boxes = apply_margins(pred_boxes, [lbd, lbd, lbd, lbd], mode=mode)
    corrected_risk = compute_risk_image_wise(true_boxes, conf_boxes, loss=loss, B=B)
    pbar.set_description(
        f"[{left:.2f}, {right:.2f}] -> {lbd:.2f}. Corrected Risk = {corrected_risk:.2f}"
    )

    return lbd, conf_boxes


def conformalize_risk_preds_objectness(
    images,
    true_boxes,
    pred_boxes,
    loss=lambda x, y: 1 - contained(x, y),
    alpha_obj=0.2,
    alpha_mar=0.1,
    mode="additive",
    tqdm_on=True,
    replace_iou=None,
    depth=13,
):
    nb_iters = depth

    B = 1

    perms = np.random.permutation(len(pred_boxes))
    _, pred_boxes, true_boxes = list(zip(*sorted(zip(perms, pred_boxes, true_boxes))))
    pred_boxes_1, pred_boxes_2 = pred_boxes[:500], pred_boxes[500:]
    true_boxes_1, true_boxes_2 = true_boxes[:500], true_boxes[500:]

    left_obj, right_obj = 0, 1
    pbar = tqdm.tqdm(range(nb_iters), disable=not tqdm_on)
    for k in pbar:
        lbd_obj = (left_obj + right_obj) / 2

        pred_boxes_obj = list(
            [list([x for x in ls if x[1] >= lbd_obj]) for ls in pred_boxes_1]
        )

        corrected_risk = compute_risk(true_boxes_1, pred_boxes_obj, loss=loss, B=B)

        pbar.set_description(
            f"[{left_obj}, {right_obj}] -> {lbd_obj}. Corrected Risk = {corrected_risk}"
        )
        if corrected_risk <= alpha_obj:
            left_obj = lbd_obj
        else:
            right_obj = lbd_obj

    lbd_obj = (left_obj + right_obj) / 2

    pred_boxes_obj = list(
        [list([x for x in ls if x[1] >= lbd_obj]) for ls in pred_boxes_1]
    )

    corrected_risk = compute_risk(true_boxes_1, pred_boxes_obj, loss=loss, B=B)

    pbar.set_description(
        f"[{left_obj}, {right_obj}] -> {lbd_obj}. Corrected Risk = {corrected_risk}"
    )
    pred_boxes_2 = list(
        [list([x for x in ls if x[1] >= lbd_obj]) for ls in pred_boxes_2]
    )

    left_mar, right_mar = 0, 1000
    pbar = tqdm.tqdm(range(nb_iters), disable=not tqdm_on)
    for k in pbar:
        lbd_mar = (left_mar + right_mar) / 2

        conf_boxes = apply_margins(
            pred_boxes_2, [lbd_mar, lbd_mar, lbd_mar, lbd_mar], mode=mode
        )

        corrected_risk = compute_risk(true_boxes_2, conf_boxes, loss=loss, B=B)

        pbar.set_description(
            f"[{left_mar}, {right_mar}] -> {lbd_mar}. Corrected Risk = {corrected_risk}"
        )
        if corrected_risk <= alpha_mar:
            right_mar = lbd_mar
        else:
            left_mar = lbd_mar

    lbd_mar = (left_mar + right_mar) / 2
    conf_boxes = apply_margins(
        pred_boxes_2, [lbd_mar, lbd_mar, lbd_mar, lbd_mar], mode=mode
    )
    corrected_risk = compute_risk(true_boxes_2, conf_boxes, loss=loss, B=B)
    pbar.set_description(
        f"[{left_mar}, {right_mar}] -> {lbd_mar}. Corrected Risk = {corrected_risk}"
    )

    return lbd_mar, lbd_obj, conf_boxes


def compute_risk(true_boxes, conf_boxes, loss, B=1):
    risk = []
    for i in range(len(true_boxes)):
        tbs = true_boxes[i]
        # print("1", len(conf_boxes[i]))
        current_boxes = conf_boxes[i]
        pbs = list(map(lambda x: x[0], current_boxes))
        # print("2", len(current_boxes))
        #             pbs = list(map(lambda x: x[0], current_boxes))
        #             #print("3", len(pbs))
        #             scores = list(map(lambda x: x[1].detach().cpu().numpy(), current_boxes))
        #             scores = np.array(scores)
        #             #print(scores)
        #             idxs = np.where(scores>=SCORE_THRESHOLD)[0]
        #             pbs = list([x for k,x in enumerate(pbs) if k in idxs])
        #             #print("e", len(pbs))
        #             #assert
        #             assert len(pbs)==len(current_boxes)

        already_assigned = []

        for tb in tbs:
            # convert format
            p1, p2 = tb["points"]
            tb = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=float)

            broke = False

            #                 for k, pb in enumerate(pbs):

            #                     #if k in already_assigned:
            #                     #    continue
            #                     #IoU
            #                     error = loss(tb, pb)
            #                     if error < 1:
            #                         #already_assigned.append(k) #TEMPORARY, smarter approach : each ground truth captured by at most one bounding box

            #                         risk.append(error)

            #                         broke = True
            #                         break
            #                 if not broke:
            #                     risk.append(1)

            errors = []
            for k, pb in enumerate(pbs):
                # if k in already_assigned:
                #    continue
                # IoU
                error = loss(tb, pb)
                if error < 1:
                    # already_assigned.append(k) #TEMPORARY, smarter approach : each ground truth captured by at most one bounding box

                    # risk.append(error)

                    broke = True
                    # break
                errors.append(error)
            if not broke:
                risk.append(1)
            else:
                risk.append(np.min(errors))

    n = len(risk)
    corrected_risk = (n / (n + 1)) * np.mean(risk) + B / (n + 1)
    return corrected_risk


def conformalize_risk_preds_set_of_boxes(
    images,
    true_boxes,
    pred_boxes,
    loss="recall",  # lambda x, y: 1 - contained(x, y),
    SCORE_THRESHOLD=0.3,
    alpha=0.1,
    tqdm_on=True,
    replace_iou=None,
):
    left, right = 0, 1000
    nb_iters = 13
    not_found = True

    B = 1

    pbar = tqdm.tqdm(range(nb_iters), disable=not tqdm_on)

    for k in pbar:
        lbd = (left + right) / 2

        risk = []

        conf_boxes = apply_margins(pred_boxes, [-lbd, -lbd, lbd, lbd])

        for i in range(len(images)):
            tbs = true_boxes[i]
            current_boxes = conf_boxes[i]
            pbs = list(map(lambda x: x[0], current_boxes))
            scores = list(map(lambda x: x[1].detach().cpu().numpy(), current_boxes))
            scores = np.array(scores)
            idxs = np.where(scores >= SCORE_THRESHOLD)[0]
            pbs = list([x for k, x in enumerate(pbs) if k in idxs])

            already_assigned = []

            for tb in tbs:
                # convert format
                p1, p2 = tb["points"]
                tb = np.array([p1[0], p1[1], p2[0], p2[1]], dtype=float)

                broke = False

                #                 for k, pb in enumerate(pbs):

                #                     #if k in already_assigned:
                #                     #    continue
                #                     #IoU
                #                     error = loss(tb, pb)
                #                     if error < 1:
                #                         #already_assigned.append(k) #TEMPORARY, smarter approach : each ground truth captured by at most one bounding box

                #                         risk.append(error)

                #                         broke = True
                #                         break
                #                 if not broke:
                #                     risk.append(1)

                errors = []
                for k, pb in enumerate(pbs):
                    # if k in already_assigned:
                    #    continue
                    # IoU
                    error = loss(tb, pb)
                    if error < 1:
                        # already_assigned.append(k) #TEMPORARY, smarter approach : each ground truth captured by at most one bounding box

                        # risk.append(error)

                        broke = True
                        # break
                    errors.append(error)
                if not broke:
                    risk.append(1)
                else:
                    risk.append(np.min(errors))

        n = len(risk)
        corrected_risk = (n / (n + 1)) * np.mean(risk) + B / (n + 1)
        pbar.set_description(
            f"[{left:.2f}, {right:.2f}] -> {lbd:.2f}. Corrected Risk = {corrected_risk}"
        )
        if corrected_risk <= alpha:
            right = lbd
        else:
            left = lbd

    pbar.set_description(
        f"[{left:.2f}, {right:.2f}] -> {lbd:.2f}. Corrected Risk = {corrected_risk} (TODO: that's the previous not the last one)"
    )

    lbd = (left + right) / 2
    return lbd, apply_margins(pred_boxes, [-lbd, -lbd, lbd, lbd])
