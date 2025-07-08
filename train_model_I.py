from typing import List
import torch
import torch.nn as nn
import argparse
import numpy as np
from src.modules import load_data
from src.models.DKT_seq import DeepKnowledgeTracing
import src.configs as C
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn import metrics

import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='Deep Knowledge Tracing Model')
parser.add_argument('-epsilon', type=float, default=1e-8, help='Epsilon value for Adam optimizer')
parser.add_argument('-l2_lambda', type=float, default=0.0, help='L2 regularization lambda')
parser.add_argument('-learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('-max_grad_norm', type=float, default=20, help='Gradient clipping max norm')
parser.add_argument('-keep_prob', type=float, default=0.6, help='Dropout keep probability')
parser.add_argument('-hidden_layer_num', type=int, default=1, help='Number of hidden layers')
parser.add_argument('-hidden_size', type=int, default=300, help='Hidden layer size')
parser.add_argument('-evaluation_interval', type=int, default=1, help='Epoch interval for evaluation')
parser.add_argument('-batch_size', type=int, default=64, help='Training batch size')
parser.add_argument('-epochs', type=int, default=30, help='Number of training epochs')
parser.add_argument('-allow_soft_placement', type=bool, default=True, help='Allow soft device placement')
parser.add_argument('-log_device_placement', type=bool, default=False, help='Log device placement')
parser.add_argument('-train_data_path', type=str, default=C.ASSIST0910_SEQ_TRAIN_PATH, help='Training dataset path')
parser.add_argument('-test_data_path', type=str, default=C.ASSIST0910_SEQ_TEST_PATH, help='Testing dataset path')
args = parser.parse_args()
print(args)


def add_gradient_noise(tensor: torch.Tensor, stddev: float = 1e-3) -> torch.Tensor:
    """Add Gaussian noise to tensor gradients for regularization."""
    noise = torch.normal(mean=torch.zeros_like(tensor), std=stddev)
    return tensor + noise


def repackage_hidden(h):
    """Detach hidden states from their history to prevent backprop through entire training history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model: DeepKnowledgeTracing, optimizer: torch.optim.Optimizer,
              students: List, batch_size: int, num_steps: int, num_skills: int,
              training: bool = True, epoch: int = 1):
    """
    Run one epoch of training or evaluation.

    Args:
        model: The DeepKnowledgeTracing model instance.
        optimizer: Optimizer for training.
        students: List of student data tuples.
        batch_size: Number of samples per batch.
        num_steps: Length of sequences.
        num_skills: Number of distinct skills.
        training: True for training mode, False for evaluation.
        epoch: Current epoch number.

    Returns:
        Average loss, RMSE, AUC, and R2 score for the epoch.
    """
    total_loss = 0
    input_size = num_skills * 2  # One-hot encoding size
    hidden = model.init_hidden(num_steps)
    batch_num = len(students) // batch_size
    index = 0
    batch_count = 0

    while index + batch_size < len(students):
        # Prepare batch input and targets
        x = np.zeros((num_steps, batch_size), dtype=np.int64)
        target_id_temp: List[int] = []
        target_correctness_temp: List[int] = []
        actual_labels: List[int] = []
        pred_labels: List[float] = []

        for i in range(batch_size):
            student = students[index + i]
            problem_ids = student[1]
            correctness = student[2]

            for j in range(len(problem_ids) - 1):
                problem_id = int(problem_ids[j])
                label_index = problem_id if int(correctness[j]) == 0 else problem_id + num_skills
                x[j, i] = label_index

                # Calculate target indices and correctness for prediction (t predicts t+1)
                target_id_temp.append(j * batch_size * num_skills + i * num_skills + int(problem_ids[j + 1]))
                target_correctness_temp.append(int(correctness[j + 1]))
                actual_labels.append(int(correctness[j + 1]))

        index += batch_size
        batch_count += 1

        # Convert input to one-hot encoding tensor
        x_tensor = torch.tensor(x, dtype=torch.int64).unsqueeze(2)
        input_data = torch.zeros(num_steps, batch_size, input_size)
        input_data.scatter_(2, x_tensor, 1)

        if training:
            model.train()
            input_data, target_id, target_correctness = (
                input_data.cuda(),
                torch.tensor(target_id_temp, dtype=torch.int64).cuda(),
                torch.tensor(target_correctness_temp, dtype=torch.float).cuda(),
            )
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
            with torch.no_grad():
                input_data = input_data.cpu()
                target_id = torch.tensor(target_id_temp, dtype=torch.int64).cpu()
                target_correctness = torch.tensor(target_correctness_temp, dtype=torch.float).cpu()

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

        print(f"Epoch: {epoch}, Batch {batch_count}/{batch_num}, AUC: {auc:.4f}")

    return total_loss / batch_num, rmse, auc, r2


def main():
    train_data_path = C.DATASET_PATH / '0910_a_train.csv'
    test_data_path = C.DATASET_PATH / '0910_a_test.csv'
    batch_size = args.batch_size
    num_steps = 500

    train_students, train_max_skill_num = load_data(train_data_path)
    test_students, test_max_skill_num = load_data(test_data_path)
    num_skills = train_max_skill_num
    num_layers = args.hidden_layer_num

    model = DeepKnowledgeTracing('LSTM', num_skills * 2, args.hidden_size, num_skills, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps=args.epsilon, weight_decay=args.l2_lambda)

    for epoch in range(args.epochs):
        model.cuda()
        train_loss, train_rmse, train_auc, train_r2 = run_epoch(model, optimizer, train_students, batch_size, num_steps, num_skills, training=True, epoch=epoch)
        print(f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}")

        if (epoch + 1) % args.evaluation_interval == 0:
            print('Evaluating on test set...')
            model.cpu()
            test_loss, test_rmse, test_auc, test_r2 = run_epoch(model, optimizer, test_students, batch_size, num_steps, num_skills, training=False)
            print(f"Test Loss: {test_loss:.4f}, Test RMSE: {test_rmse:.4f}, Test AUC: {test_auc:.4f}")

        # Save checkpoint
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, 'train1.pth')
        print('Model saved for this epoch.')


if __name__ == '__main__':
    main()
