import functools

import torch

import librl.task


# Given a set of classification task samples, train the task's classifer on the task data.
# This helper method assumes classifiers do not share components, and that we are
# only dealing in assigning a single label to each data element.
def train_single_label_classifier(task_samples):

    classifiers = { (id(task.classifier),task.classifier) for task in task_samples}
    correct, total = [], []
    for task in task_samples:
        train_percent = task.train_percent
        validation_percent = task.validation_percent
        assert task.problem_type == librl.task.ProblemTypes.Classification
        assert (0.0 < train_percent <= 1.0
            and 0.0 < validation_percent <= 1.0)
        for dataloader in task.train_data_iter:
            task.classifier.train()
            observed_train_data = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(task.device), target.to(task.device)
                loss, selected = task.train_batch(task.classifier, task.criterion, data, target)
                observed_train_data += len(data)
                if observed_train_data >= train_percent * len(dataloader):
                    break
        local_correct, local_total = 0,0
        for dataloader in task.validation_data_iter:
            task.classifier.eval()
            observed_valid_data = 0
            for batch_idx, (data, target) in enumerate(dataloader):
                data, target = data.to(task.device), target.to(task.device)
                loss, selected = task.validate_batch(task.classifier, task.criterion, data, target)
                local_correct += torch.eq(selected, target).sum() 
                local_total += float(target.shape[0])
                observed_valid_data += len(data)
                if observed_valid_data >= validation_percent * len(dataloader):
                    break
        correct.append(local_correct), total.append(local_total)
    return correct, total
       