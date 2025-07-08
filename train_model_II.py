from typing import List
import torch
import torch.nn as nn
import argparse
import numpy as np
from src.modules import load_data, load_feature
from src.models.DKT_seq import DeepKnowledgeTracing
import src.configs as C
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn import metrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Deep Knowledge Tracing model')
parser.add_argument('-epsilon', type=float, default=1e-08, help='Epsilon value for Adam Optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.000, help='Lambda for L2 regularization')
parser.add_argument('-learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Max gradient norm for clipping')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Dropout keep probability')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='Number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=50, help='Number of hidden units per layer')
parser.add_argument('-evaluation_interval', type=int, default=1, help='Interval (in epochs) for evaluation and logging')
parser.add_argument('-batch_size', type=int, default=64, help='Batch size for training')
parser.add_argument('-epochs', type=int, default=50, help='Number of epochs to train')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow device soft placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log device placement')
parser.add_argument('-train_data_path', type=str, default=C.ASSIST2017_SEQ_TRAIN_PATH, help='Path to training dataset')
parser.add_argument('-test_data_path', type=str, default=C.ASSIST2017_SEQ_TEST_PATH, help='Path to testing dataset')
args = parser.parse_args()
print(args)


def add_gradient_noise(t, stddev=1e-3):
    """Add Gaussian noise to gradients to improve optimization."""
    noise = torch.normal(mean=torch.zeros_like(t), std=torch.full_like(t, stddev))
    return t + noise


def repackage_hidden(h):
    """Detach hidden states from their history to prevent backpropagating through entire training history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, optimizer, students, features, batch_size, num_steps, num_skills, training=True, epoch=1):
    """
    Run one training or evaluation epoch.

    Args:
        model: DeepKnowledgeTracing model instance.
        optimizer: optimizer instance.
        students: list of student data.
        features: input feature tensor.
        batch_size: batch size.
        num_steps: maximum sequence length.
        num_skills: number of skills (output dimension).
        training: boolean flag for training or evaluation mode.
        epoch: current epoch number.

    Returns:
        average loss, RMSE, AUC, R2 score for the epoch.
    """

    # resume
    # checkpoint = torch.load('train.pth')
    # m.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # epoch = checkpoint['epoch'] + 1

    total_loss = 0
    input_size = features.shape[2]  # feature dimension, here 43
    index = 0
    hidden = model.init_hidden(num_steps)
    count = 0
    batch_num = len(students) // batch_size

    while index + batch_size < len(students):
        # Prepare batch inputs and targets
        x = np.zeros((batch_size, num_steps, input_size), dtype=np.float32)
        target_id_temp: List[int] = []
        target_correctness_temp = []
        actual_labels = []
        pred_labels = []

        for i in range(batch_size):
            student = students[index + i]
            problem_ids = student[1]
            correctness = student[2]

            for j in range(len(problem_ids) - 1):
                # Append target indices and correctness labels for prediction
                target_id_temp.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j + 1]))
                target_correctness_temp.append(int(correctness[j + 1]))
                actual_labels.append(int(correctness[j + 1]))

                # Load input features
                x[i][j] = features[index + i][j]

        index += batch_size
        count += 1

        # Convert to torch tensor and permute for RNN input shape (seq_len, batch, input_size)
        input_data = torch.FloatTensor(x).permute(1, 0, 2)

        if training:
            model.train()
            target_id = torch.tensor(target_id_temp, dtype=torch.int64).cuda()
            target_correctness = torch.tensor(target_correctness_temp, dtype=torch.float).cuda()
            input_data = input_data.cuda()

            hidden = repackage_hidden(hidden)
            optimizer.zero_grad()

            output, hidden = model(input_data, hidden)
            output = output.contiguous().view(-1)
            logits = torch.gather(output, 0, target_id)

            preds = torch.sigmoid(logits)
            pred_labels.extend(p.item() for p in preds)

            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(logits, target_correctness)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            total_loss += loss.item()
        else:
            model.eval()
            target_id = torch.tensor(target_id_temp, dtype=torch.int64).cpu()
            target_correctness = torch.tensor(target_correctness_temp, dtype=torch.float).cpu()
            with torch.no_grad():
                output, hidden = model(input_data, hidden)
                output = output.contiguous().view(-1)
                logits = torch.gather(output, 0, target_id)
                preds = torch.sigmoid(logits)
                pred_labels.extend(p.item() for p in preds)

                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(logits, target_correctness)
                total_loss += loss.item()

                hidden = repackage_hidden(hidden)

        rmse = sqrt(mean_squared_error(actual_labels, pred_labels))
        fpr, tpr, _ = metrics.roc_curve(actual_labels, pred_labels, pos_label=1)
        auc = metrics.auc(fpr, tpr)
        r2 = r2_score(actual_labels, pred_labels)

        print(f"Epoch: {epoch}, Batch {count}/{batch_num} - AUC: {auc:.4f}")

    return total_loss / batch_num, rmse, auc, r2


def main():
    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    batch_size = args.batch_size
    num_steps = 500

    train_students, train_max_skill_num = load_data(train_data_path)
    features = load_feature(C.ASSIST2017_FEA_PATH, num_steps)
    train_fea = features[:int(len(features) * 0.8)]
    test_fea = features[int(len(features) * 0.8):]
    num_skills = train_max_skill_num
    num_layers = args.hidden_layer_num

    test_students, test_max_skill_num = load_data(test_data_path)

    model = DeepKnowledgeTracing('LSTM', input_size=features.shape[2], hidden_size=args.hidden_size,
                                 output_size=num_skills, num_layers=num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon, weight_decay=args.l2_lambda)

    for epoch in range(args.epochs):
        model.cuda()
        train_loss, train_rmse, train_auc, train_r2 = run_epoch(model, optimizer, train_students, train_fea,
                                                               batch_size, num_steps, num_skills,
                                                               training=True, epoch=epoch)
        print(f"Epoch {epoch} Training Loss: {train_loss:.4f}, AUC: {train_auc:.4f}")

        if (epoch + 1) % args.evaluation_interval == 0:
            print('Testing...')
            model.cpu()
            test_loss, test_rmse, test_auc, test_r2 = run_epoch(model, optimizer, test_students, test_fea,
                                                                batch_size, num_steps, num_skills,
                                                                training=False, epoch=epoch)
            print(f"Epoch {epoch} Test Loss: {test_loss:.4f}, RMSE: {test_rmse:.4f}, AUC: {test_auc:.4f}")

        # Save model checkpoint
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch
        }
        torch.save(checkpoint, 'train2.pth')
        print('Model saved for this epoch')


if __name__ == '__main__':
    main()
