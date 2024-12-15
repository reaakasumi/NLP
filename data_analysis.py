from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def analyze_predictions(df_test_labeled, prediction_col="predicted_vectara", true_label_col="true_labels"):
    # Classification report
    print("Classification Report:")
    report = classification_report(df_test_labeled[true_label_col], df_test_labeled[prediction_col], zero_division=0)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(df_test_labeled[true_label_col], df_test_labeled[prediction_col])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=['Not Hallucination', 'Hallucination'],
                yticklabels=['Not Hallucination', 'Hallucination'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()

    # Calculate average length difference for hallucination and non-hallucination
    df_test_labeled['hyp_length'] = df_test_labeled['hyp_normalized'].apply(len)
    df_test_labeled['tgt_length'] = df_test_labeled['tgt_normalized'].apply(len)
    df_test_labeled['length_diff'] = abs(df_test_labeled['hyp_length'] - df_test_labeled['tgt_length'])

    hallucination = df_test_labeled[df_test_labeled[prediction_col] == 1]
    non_hallucination = df_test_labeled[df_test_labeled[prediction_col] == 0]

    avg_length_diff_hallucination = hallucination['length_diff'].mean()
    avg_length_diff_non_hallucination = non_hallucination['length_diff'].mean()

    print(f"Average Length Difference for Hallucination: {avg_length_diff_hallucination:.2f}")
    print(f"Average Length Difference for Non-Hallucination: {avg_length_diff_non_hallucination:.2f}")

    # Precision and Recall
    precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) != 0 else 0
    recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) != 0 else 0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    # Return summary data for further inspection if needed
    return {
        "classification_report": report,
        "confusion_matrix": cm,
        "avg_length_diff_hallucination": avg_length_diff_hallucination,
        "avg_length_diff_non_hallucination": avg_length_diff_non_hallucination,
        "precision": precision,
        "recall": recall,
    }


# Analyze for MT
print("Analysis for MT Task:")
analyze_predictions(df_mt_test)

# Analyze for DM
print("Analysis for DM Task:")
analyze_predictions(df_dm_test)

# Analyze for PG
print("Analysis for PG Task:")
analyze_predictions(df_pg_test)
