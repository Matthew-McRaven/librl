import torch

import librl.task



# Given a set of classification task samples, train the task's classifer on the task data.
# This helper method assumes classifiers do not share components, and that we are
# only dealing in assigning a single label to each data element.
def train_single_label_classifier(task_samples):
    classifiers = { (id(task.classifier),task.classifier) for task in task_samples}
    for task in task_samples:
        assert task.problem_type == librl.task.ProblemTypes.Classification
        for dataloader in task.train_data_iter:
            task.classifier.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                loss, selected = task.train_batch(task.classifier, task.criterion, data, target)
                #accuracy = torch.eq(selected, target).sum() / float(target.shape[0])
                #print(f"Batch {batch_idx} with loss {loss}")

        for dataloader in task.validation_data_iter:
            task.classifier.eval()
            for batch_idx, (data, target) in enumerate(dataloader):
                loss, selected = task.validate_batch(task.classifier, task.criterion, data, target)
                #print(target, selected)
                accuracy = torch.eq(selected, target).sum() / float(target.shape[0])
                print(f"Accuracy of {accuracy}")
       