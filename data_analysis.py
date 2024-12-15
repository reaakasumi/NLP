from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

def detailed_analysis(df_results, true_label_col, pred_label_col, hyp_col, tgt_col):
    # Extracting true and predicted labels
    true_labels = df_results[true_label_col]
    pred_labels = df_results[pred_label_col]

    # Generating classification report
    print("Classification Report:")
    print(classification_report(true_labels, pred_labels))

    # Generating confusion matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(true_labels, pred_labels)
    print(cm)

    # Plotting confusion matrix
    plt.matshow(cm, cmap='coolwarm')
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    # Computing average length of hypothesis and target entries
    hyp_lengths = df_results[hyp_col].apply(len)
    tgt_lengths = df_results[tgt_col].apply(len)

    avg_hyp_length = hyp_lengths.mean()
    avg_tgt_length = tgt_lengths.mean()

    print(f"Average Hypothesis Length: {avg_hyp_length:.2f}")
    print(f"Average Target Length: {avg_tgt_length:.2f}")

    # Calculating precision and recall
    tp = cm[1, 1]
    fp = cm[0, 1]
    fn = cm[1, 0]

    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        precision = 0

    recall = tp / (tp + fn)

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    return {
        'classification_report': classification_report(true_labels, pred_labels, output_dict=True),
        'confusion_matrix': cm,
        'avg_hyp_length': avg_hyp_length,
        'avg_tgt_length': avg_tgt_length,
        'precision': precision,
        'recall': recall
    }
