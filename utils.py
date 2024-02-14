import numpy as np
import cv2 as cv
import os
import time
import torch as th
import matplotlib.pyplot as plt


def train_epoch(model, optimizer, loss_func, data_loader, device, start, epoch):
    correct = 0
    total = 0
    start_epoch = time.time()
    losses = []
    accuracies = []
    model.train()
    for i, (X_batch, y_batch) in enumerate(data_loader):
        total += y_batch.shape[0]
        # print(X_batch.shape, y_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)

        correct += (y_pred.max(dim=-1)[1] == y_batch).sum().item()
        accuracy = correct / total * 100
        loss = loss_func(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()

        th.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        current_time = time.time()
        print(f'\r--train-- Epoch: {epoch}, Iteration: {i + 1}/{len(data_loader)}, '
              f'Loss: {loss.item()}, Accuracy: {round(accuracy, 2)}%, '
              f'Time/iteration: {round((current_time - start_epoch) / (i + 1), 2)}s, '
              f'Total time: {round(current_time - start, 2)}s', end='')

        losses.append(loss.item())
        accuracies.append(accuracy)
    print()

    return losses, accuracies


def val_epoch(model, loss_func, data_loader, device, start, epoch):
    correct = 0
    total = 0
    start_epoch = time.time()
    losses = []
    accuracies = []
    model.eval()
    for i, (X_batch, y_batch) in enumerate(data_loader):
        total += y_batch.shape[0]
        # print(X_batch.shape, y_batch)
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_pred = model(X_batch)

        correct += (y_pred.max(dim=-1)[1] == y_batch).sum().item()
        accuracy = correct / total * 100
        loss = loss_func(y_pred, y_batch)

        current_time = time.time()
        print(f'\r--val-- Epoch: {epoch}, Iteration: {i + 1}/{len(data_loader)}, '
              f'Loss: {loss.item()}, Accuracy: {round(accuracy, 2)}%, '
              f'Time/iteration: {round((current_time - start_epoch) / (i + 1), 2)}s, '
              f'Total time: {round(current_time - start, 2)}s', end='')

        losses.append(loss.item())
        accuracies.append(accuracy)
    print()

    return losses, accuracies


def visualize_patches(unfolded, out='./out'):
    for i in range(unfolded.shape[1]):
        for j in range(unfolded.shape[2]):
            patch = unfolded[:, i, j, :, :].permute(1, 2, 0).numpy()
            patch = patch * 255
            patch = cv.cvtColor(patch, cv.COLOR_RGB2BGR)
            cv.imwrite(os.path.join(out, f'patch_{i}_{j}.png'), patch)


def plot_progress(out, losses, accuracies):
    xs = [x for x in range(len(losses))]
    plt.plot(xs, losses, label='loss')
    plt.plot(xs, accuracies, label='accuracy')
    plt.legend()
    plt.savefig(f'./logs/{out}.png')
    plt.cla()
