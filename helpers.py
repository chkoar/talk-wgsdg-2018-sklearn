from sklearn.metrics import recall_score

def balanced_accuracy_score(y_true, y_pred, sample_weight=None):

    return recall_score(y_true, y_pred,
                        pos_label=None,
                        average='macro',
                        sample_weight=sample_weight)