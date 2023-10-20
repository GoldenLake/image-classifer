import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score
import csv

def calculate_metrics(label_list, pred_list):
    # 计算混淆矩阵
    confusion = confusion_matrix(label_list, pred_list)

    # 计算每个类别的精确度、灵敏度、特异度和F1值
    num_classes = confusion.shape[0]
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    specificity = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        # 精确度（Precision）
        TP = confusion[i, i]
        FP = sum(confusion[:, i]) - TP
        precision[i] = TP / (TP + FP)

        # 灵敏度（Recall）
        FN = sum(confusion[i, :]) - TP
        recall[i] = TP / (TP + FN)

        # 特异度（Specificity）
        TN = np.sum(np.delete(np.delete(confusion, i, axis=0), i, axis=1))
        FP = np.sum(confusion[i, :]) - TP
        specificity[i] = TN / (TN + FP)

        # F1值
        f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    # 计算AUC（针对每个类别）
    auc = np.zeros(num_classes)
    for i in range(num_classes):
        y_true = np.zeros(confusion.shape[0])
        y_true[i] = 1  # 正类别为1，其他为0
        y_score = confusion[i, :]
        auc[i] = roc_auc_score(y_true, y_score)
    print(auc)
    # 计算各项指标的平均值
    average_precision = np.mean(precision)
    average_recall = np.mean(recall)
    average_specificity = np.mean(specificity)
    average_f1_score = np.mean(f1_score)
    average_auc = np.mean(auc)

    # 计算准确率
    accuracy = accuracy_score(label_list, pred_list)

    return {
        "confusion_matrix": confusion,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "average_precision": average_precision,
        "average_recall": average_recall,
        "average_specificity": average_specificity,
        "average_f1_score": average_f1_score,
        "average_auc": average_auc
    }

# 将结果写入CSV文件
def write_csv(label_list, pred_list, max_accuracy):
    result = calculate_metrics(label_list, pred_list)

    if result["accuracy"] > max_accuracy[0]:
        max_accuracy[0] = result["accuracy"]
        with open('performance_metrics.csv', mode='w', newline='') as csv_file:
            fieldnames = ['Metric', 'Value']
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow({'Metric': 'Confusion Matrix', 'Value': result["confusion_matrix"]})
            writer.writerow({'Metric': 'Accuracy', 'Value': result["accuracy"]})
            writer.writerow({'Metric': 'Average Precision', 'Value': result["average_precision"]})
            writer.writerow({'Metric': 'Average Recall', 'Value': result["average_recall"]})
            writer.writerow({'Metric': 'Average Specificity', 'Value': result["average_specificity"]})
            writer.writerow({'Metric': 'Average F1 Score', 'Value': result["average_f1_score"]})
            writer.writerow({'Metric': 'Average AUC', 'Value': result["average_auc"]})

    # 打印结果
    for key, value in result.items():
        if key != "confusion_matrix":
            print(f"{key}: {value}")


