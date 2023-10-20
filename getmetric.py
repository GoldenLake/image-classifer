import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.metrics import roc_auc_score, roc_curve, auc
import csv
import matplotlib.pyplot as plt


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


def plot_roc(y_true, y_scores, classes):
    # 假设有N个类别
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    n_classes = y_scores.shape[1]  # 类别数
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_scores[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']  # 每个类别对应的颜色
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{classes[i]}: {round(roc_auc[i], 3)}'
                 )

    plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 绘制对角线
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc.jpg')

# 假设有N个类别
# y_true = np.array([0, 1, 1, 0, 2, 2])  # 真实标签
# y_scores = np.array([[0.9, 0.05, 0.05],  # 预测概率，每一行代表一个样本的预测概率分布
#                      [0.1, 0.8, 0.1],
#                      [0.3, 0.3, 0.4],
#                      [0.7, 0.2, 0.1],
#                      [0.2, 0.3, 0.5],
#                      [0.1, 0.2, 0.7]])
# classes = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink']
#
# plot_roc(y_true, y_scores, classes)
