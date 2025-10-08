import torch
import torch.nn as nn


class EDiceLoss(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        ce = 0
        CE_L = torch.nn.BCELoss()
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
            ce = ce + CE_L(torch.sigmoid(inputs[:, i, ...]), target[:, i, ...])
        final_dice = (0.7 * dice + 0.3 * ce) / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class EDiceLoss_Val(nn.Module):
    """Dice loss tailored to Brats need.
    """

    def __init__(self, do_sigmoid=True):
        super(EDiceLoss_Val, self).__init__()
        self.do_sigmoid = do_sigmoid
        self.labels = ["ET", "TC", "WT"]
        self.device = "cpu"

    def binary_dice(self, inputs, targets, label_index, metric_mode=False):
        smooth = 1.
        if self.do_sigmoid:
            inputs = torch.sigmoid(inputs)

        if metric_mode:
            inputs = inputs > 0.5
            if targets.sum() == 0:
                print(f"No {self.labels[label_index]} for this patient")
                if inputs.sum() == 0:
                    return torch.tensor(1., device="cuda")
                else:
                    return torch.tensor(0., device="cuda")
            # Threshold the pred
        intersection = EDiceLoss_Val.compute_intersection(inputs, targets)
        if metric_mode:
            dice = (2 * intersection) / ((inputs.sum() + targets.sum()) * 1.0)
        else:
            dice = (2 * intersection + smooth) / (inputs.pow(2).sum() + targets.pow(2).sum() + smooth)
        if metric_mode:
            return dice
        return 1 - dice

    @staticmethod
    def compute_intersection(inputs, targets):
        intersection = torch.sum(inputs * targets)
        return intersection

    def forward(self, inputs, target):
        dice = 0
        for i in range(target.size(1)):
            dice = dice + self.binary_dice(inputs[:, i, ...], target[:, i, ...], i)
        final_dice = dice / target.size(1)
        return final_dice

    def metric(self, inputs, target):
        dices = []
        for j in range(target.size(0)):
            dice = []
            for i in range(target.size(1)):
                dice.append(self.binary_dice(inputs[j, i], target[j, i], i, True))
            dices.append(dice)
        return dices


class memory_Loss(nn.Module):
    def __init__(self, modal_num, class_num, temperature_f, temperature_l, device="cuda"):
        super(memory_Loss, self).__init__()
        self.modal_num = modal_num
        self.class_num = class_num
        self.temperature_f = temperature_f
        self.temperature_l = temperature_l
        self.device = device

        # self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N // 2):
            mask[i, N // 2 + i] = 0
            mask[N // 2 + i, i] = 0
        mask = mask.bool()
        return mask

    def forward_modal(self, h_i, h_j):
        N = 2 * self.class_num
        h = torch.cat((h_i, h_j), dim=1)
        sim = torch.matmul(h.T, h) / self.temperature_f
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_modal_cos(self, h_i, h_j):
        loss_i_j = 0.0
        tt = 0
        for i in range(self.class_num):
            for j in range(i + 1, self.class_num):
                positive_samples = torch.exp(torch.cosine_similarity(h_i[:, i], h_i[:, j], dim=0) / self.temperature_f)
                negative_samples = 0
                for k in range(self.class_num):
                    negative_samples = negative_samples + torch.exp(
                        torch.cosine_similarity(h_i[:, i], h_j[:, k], dim=0) / self.temperature_f)
                tt = tt + 1
            loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)

        for i in range(self.class_num):
            for j in range(i + 1, self.class_num):
                positive_samples = torch.exp(torch.cosine_similarity(h_j[:, i], h_j[:, j], dim=0) / self.temperature_f)
                negative_samples = 0
                for k in range(self.class_num):
                    negative_samples = negative_samples + torch.exp(
                        torch.cosine_similarity(h_j[:, i], h_i[:, k], dim=0) / self.temperature_f)
                tt = tt + 1
            loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)

        return loss_i_j / tt

    def forward_class(self, h_i, h_j):
        N = 2 * self.modal_num
        h = torch.cat((h_i, h_j), dim=1)

        sim = torch.matmul(h.T, h) / self.temperature_f
        sim_i_j = torch.diag(sim, self.modal_num)
        sim_j_i = torch.diag(sim, -self.modal_num)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_samples = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

    def forward_class_cos(self, h_i, h_j):
        loss_i_j = 0.0
        tt = 0
        for i in range(self.modal_num):
            for j in range(i + 1, self.modal_num):
                positive_samples = torch.exp(torch.cosine_similarity(h_i[:, i], h_i[:, j], dim=0) / self.temperature_f)
                negative_samples = 0
                for k in range(self.modal_num):
                    negative_samples = negative_samples + torch.exp(
                        torch.cosine_similarity(h_i[:, i], h_j[:, k], dim=0) / self.temperature_f)
                tt = tt + 1
            loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)

        for i in range(self.modal_num):
            for j in range(i + 1, self.modal_num):
                positive_samples = torch.exp(torch.cosine_similarity(h_j[:, i], h_j[:, j], dim=0) / self.temperature_f)
                negative_samples = 0
                for k in range(self.modal_num):
                    negative_samples = negative_samples + torch.exp(
                        torch.cosine_similarity(h_j[:, i], h_i[:, k], dim=0) / self.temperature_f)
                tt = tt + 1
            loss_i_j = loss_i_j - torch.log(positive_samples / negative_samples)

        return loss_i_j / tt

