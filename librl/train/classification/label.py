import torch

import librl.task



# Given a set of classification task samples, train the task's classifer on the task data.
# This helper method assumes classifiers do not share components, and that we are
# only dealing in assigning a single label to each data element.
def train_single_label_classifier(task_samples):
    classifiers = { (id(task.classifier),task.classifier) for task in task_samples}
    correct, total = [], []
    for task in task_samples:
        assert task.problem_type == librl.task.ProblemTypes.Classification
        for dataloader in task.train_data_iter:
            task.classifier.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                loss, selected = task.train_batch(task.classifier, task.criterion, data, target)
        local_correct, local_total = 0,0
        for dataloader in task.validation_data_iter:
            task.classifier.eval()
            for batch_idx, (data, target) in enumerate(dataloader):
                loss, selected = task.validate_batch(task.classifier, task.criterion, data, target)
                local_correct += torch.eq(selected, target).sum() 
                local_total += float(target.shape[0])
        correct.append(local_correct), total.append(local_total)
    return correct, total
       