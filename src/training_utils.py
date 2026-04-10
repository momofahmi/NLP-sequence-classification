import torch
from datasets import concatenate_datasets
from transformers import Trainer


# to compute class weights for imbalanced datasets 
def class_weights(dataset, label_col: str) -> torch.Tensor:
    
    labels = dataset[label_col]
    n_total = len(labels)
    n_class1 = sum(labels) # (Sarcasm=1 or Sentiment=1)
    n_class0 = n_total - n_class1 # (Sarcasm=0 or Sentiment=0)

    # formula : total / (2 * count_per_class)
    # minority class gets higher weight 
    weight_0 = n_total / (2 * n_class0)
    weight_1 = n_total / (2 * n_class1)

    print(f"[{label_col}] class 0 : {n_class0} samples  weight : {weight_0:.3f}")
    print(f"[{label_col}] class 1 : {n_class1} samples  weight : {weight_1:.3f}")

    return torch.tensor([weight_0, weight_1], dtype=torch.float)


def new_weighted_class(class_weights: torch.Tensor):

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels  = inputs.pop("labels") # real labels from the dataset
            outputs = model(**inputs) 
            logits  = outputs.logits # model predictions 

            weights = class_weights.to(logits.device) 
            loss_fn = torch.nn.CrossEntropyLoss(weight=weights) # loss function with class weights to handle imbalance
            loss    = loss_fn(logits, labels)

            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer