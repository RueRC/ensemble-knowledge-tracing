from typing import List
import torch
import torch.nn as nn
import argparse
import numpy as np
from src.modules import load_data, load_feature
from src.models.DKT_seq import DeepKnowledgeTracing
import src.configs as C
from sklearn.metrics import mean_squared_error, roc_curve, auc, r2_score
from math import sqrt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Deep Knowledge tracing model')
parser.add_argument('-epsilon', type=float, default=1e-08, help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.000, help='Lambda for l2 loss')
parser.add_argument('-learning_rate', type=float, default=0.005, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Clip gradients to this norm')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Keep probability for dropout')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='The number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=300, help='The number of hidden nodes')
parser.add_argument('-evaluation_interval', type=int, default=1, help='Evaluation and print result every x epochs')
parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow device soft device placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log placement of ops on devices')
parser.add_argument('-train_data_path', type=str, default=C.ASSIST2017_SEQ_TRAIN_PATH, help='Path to the training dataset')
parser.add_argument('-test_data_path', type=str, default=C.ASSIST2017_SEQ_TEST_PATH, help='Path to the testing dataset')
args = parser.parse_args()
print(args)


def add_gradient_noise(t, stddev=1e-3):
    noise = torch.normal(mean=torch.zeros_like(t), std=stddev)
    return t + noise


def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def fusion(m1, m2, students, fea, batch_size, num_steps, num_skills, epoch=1):
    total_loss = 0
    input_size1 = num_skills * 2
    input_size2 = 43
    index = 0
    hidden1 = m1.init_hidden(batch_size)
    hidden2 = m2.init_hidden(batch_size)
    batch_num = len(students) // batch_size

    for batch_idx in range(batch_num):
        x1 = np.zeros((num_steps, batch_size))
        x2 = np.zeros((batch_size, num_steps, input_size2))
        target_id_temp = []
        target_correctness_temp = []
        actual_labels = []
        pred_labels = []

        for i in range(batch_size):
            student = students[index + i]
            problem_ids = student[1]
            correctness = student[2]
            seq_len = len(problem_ids) - 1

            for j in range(seq_len):
                problem_id = int(problem_ids[j])
                label_index = problem_id if correctness[j] == 0 else problem_id + num_skills
                x1[j, i] = label_index

                target_id_temp.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j + 1]))
                target_correctness_temp.append(int(correctness[j + 1]))
                actual_labels.append(int(correctness[j + 1]))
                x2[i, j] = fea[index + i][j]

        index += batch_size

        # Prepare inputs
        x1_tensor = torch.tensor(x1, dtype=torch.int64)
        x1_tensor = x1_tensor.unsqueeze(2)
        input_data1 = torch.zeros(num_steps, batch_size, input_size1, dtype=torch.float32)
        input_data1.scatter_(2, x1_tensor, 1)

        input_data2 = torch.tensor(x2, dtype=torch.float32).permute(1, 0, 2)

        target_id = torch.tensor(target_id_temp, dtype=torch.int64)
        target_correctness = torch.tensor(target_correctness_temp, dtype=torch.float32)

        with torch.no_grad():
            m1.eval()
            output1, hidden1 = m1(input_data1, hidden1)
            output1 = output1.contiguous().view(-1)
            logits1 = torch.gather(output1, 0, target_id)
            preds1 = torch.sigmoid(logits1)
            hidden1 = repackage_hidden(hidden1)

            m2.eval()
            output2, hidden2 = m2(input_data2, hidden2)
            output2 = output2.contiguous().view(-1)
            logits2 = torch.gather(output2, 0, target_id)
            preds2 = torch.sigmoid(logits2)
            hidden2 = repackage_hidden(hidden2)

            logits = 0.2 * logits1 + 0.8 * logits2
            preds = 0.2 * preds1 + 0.8 * preds2

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)
            total_loss += loss.item()

            pred_labels.extend(preds.cpu().numpy())

        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, _ = roc_curve(actual_labels, pred_labels, pos_label=1)
        auc_score = auc(fpr, tpr)

        print(f"Epoch: {epoch}, Batch {batch_idx + 1}/{batch_num} AUC: {auc_score:.4f}, Loss: {loss.item():.4f}")

        r2 = r2_score(actual_labels, pred_labels)

    return total_loss / batch_num, auc_score


def main():
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size

    train_students, train_max_skill_num = load_data(train_data_path)
    num_steps = 500
    num_skills = train_max_skill_num
    num_layers = args.hidden_layer_num
    test_students, test_max_skill_num = load_data(test_data_path)

    model1 = DeepKnowledgeTracing('LSTM', num_skills * 2, 300, num_skills, num_layers)
    model2 = DeepKnowledgeTracing('LSTM', 43, 50, num_skills, num_layers)
    features = load_feature(C.ASSIST2017_FEA_PATH, num_steps)
    test_fea = features[int(len(features) * 0.8):]

    checkpoint1 = torch.load('train1.pth')
    model1.load_state_dict(checkpoint1['model'])
    checkpoint2 = torch.load('train2.pth')
    model2.load_state_dict(checkpoint2['model'])

    for i in range(args.epochs):
        model1.cpu()
        model2.cpu()
        fusion(model1, model2, test_students, test_fea, batch_size, num_steps, test_max_skill_num, epoch=i + 1)


if __name__ == '__main__':
    main()
