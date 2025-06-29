import pickle
from PIL import Image
import math as m
import numpy as np
import json
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from skimage.util import random_noise
import torch
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import os
from collections import defaultdict
from decimal import Decimal
import pandas as pd
import re

def assert_hparams(hparams):
    '''
    Check that the setted hyperparams are valid, raising errors.
    '''
    allowed_keys = {'lr', 'batch_size', 'model_config'}

    # Check valid keys
    for key in hparams.keys():
        if key not in allowed_keys:
            raise ValueError(f"hyperparam key not valid: '{key}'. Allowed keys are: {allowed_keys}")

    # Check for each hparam
    for key, values in hparams.items():
        if not isinstance(values, list):
            raise TypeError(f"Values for '{key}' must be inside a list.")

        if key == 'lr':
            for v in values:
                if not isinstance(v, float):
                    raise TypeError(f"All values of 'lr' must be floats. Found: {type(v)}")
        
        elif key == 'batch_size':
            for v in values:
                if not isinstance(v, int):
                    raise TypeError(f"All values of 'batch_size' must be ints. Found: {type(v)}")

        elif key == 'model_config':
            for idx, cfg in enumerate(values):
                if not isinstance(cfg, dict):
                    raise TypeError(f"All values of 'model_config' must be dicts. Found: {type(cfg)}")

                required_keys = {'down_ch', 'layers', 'head_features'}
                for k in required_keys:
                    if k not in cfg:
                        raise KeyError(f"Configuration {idx} di 'model_config' does not contain required key: '{k}'")
                    if not isinstance(cfg[k], list) or not all(isinstance(i, int) for i in cfg[k]):
                        raise TypeError(f"Values of '{k}' in 'model_config' Must be list of ints.")


def clean_value(val):
    if isinstance(val, float):
        return str(Decimal(str(val)).normalize())
    return str(val)


def show_inference_samples(classifier, loader, device, class2names, epoch, test, n=16):
    '''
    Save a grid of inferred samples from the loader. Used in the fast training
    Inputs:
    - classifier: trained model
    - loader: dataloader from which to extract the samples to infer
    - device: working device
    - class2names: dict that link a class id to its name
    - epoch: current training epoch
    - test: True if the inference is done in test phase. False if it's done in the validation phase
    '''
    os.makedirs('results/fast_train/epochs', exist_ok=True)

    # Fetch samples from loader
    images, labels = next(iter(loader))
    images = images.to(device)
    labels = labels.to(device)

    # Get predictions
    outputs = classifier(images)  
    probabilities = torch.softmax(outputs, dim=1)  
    confs, preds = torch.max(probabilities, dim=1)

    # Plot initialization
    n_samples = min(n, len(images))
    grid_size = int(np.floor(np.sqrt(n_samples)))
    plt.figure(figsize=(grid_size * 2, grid_size * 2))

    # Plot results for each image
    for i in range(int(grid_size**2)):
        img = images[i].cpu()
        label = labels[i].item()
        pred = preds[i].item()
        prob = confs[i].item()

        label_name = class2names.get(label, str(label))
        pred_name = class2names.get(pred, str(pred))

        is_correct = label == pred
        title_color = 'green' if is_correct else 'red'

        img_np = img.permute(1, 2, 0).squeeze().numpy()  
        if img_np.ndim == 2: 
            cmap = 'gray'
        else:
            cmap = None

        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(img_np, cmap=cmap)
        plt.axis('off')
        plt.title(f"T: {label_name}\nP: {pred_name} ({prob * 100:.2f}%)", color=title_color, fontsize=12, weight='bold')

    # Save
    plt.tight_layout()
    if test:
        plt.savefig('results/fast_train/final_inference.png')
    else:
        plt.savefig('results/fast_train/epochs/validation_{}'.format(epoch) + '.png')
    plt.close()


def save_validation_summary(all_preds, all_labels, class2names, epoch, test, save_dir="results/fast_train"):
    '''
    Save model validation summary. Used in the fast training
    Inputs:
    - all_preds: predictions of the model
    - all_labels: true labels of the images
    - class2names: dict that link a class id to its name
    - epoch: current epoch
    - test: True if the inference is done in test phase. False if it's done in the validation phase

    File saved:
    - heatmap of the total confusion matrix (normalized)
    - validation log with total accuracy, precision and recall of all classes
    ''' 
    os.makedirs('results/fast_train/epochs', exist_ok=True)
    
    # Predictions and labels conversion
    preds = torch.cat(all_preds).cpu().numpy()
    labels = torch.cat(all_labels).cpu().numpy()

    # Class managing
    num_classes = len(class2names)
    present_classes = sorted(set(labels) | set(preds))
    filtered_class_names = [class2names[i] for i in present_classes] 

    # Normalized confusion matrix 
    cm = confusion_matrix(labels, preds, labels=present_classes, normalize='true')

    # Heatmap plot 
    plt.figure(figsize=(10, 8))
    class_names = [class2names[i] for i in range(num_classes)] 
    ax = sns.heatmap(cm, annot=False, fmt=".2f", cmap="YlOrRd", xticklabels=filtered_class_names, yticklabels=filtered_class_names, cbar=True)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)

    # Ticks settings
    ax.set_xticklabels(filtered_class_names, rotation=90)
    ax.set_yticklabels(filtered_class_names, rotation=0)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Saving
    if test:
        heatmap_path = os.path.join(save_dir, "final_confmat.png")
    else:
        heatmap_path = os.path.join(save_dir, f"epochs/confmat_{epoch}.png")
    plt.tight_layout()
    plt.savefig(heatmap_path)
    plt.close()

    # Metrics computation
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    accuracy = cm.trace() / cm.sum()
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = np.divide(np.diag(cm), cm.sum(axis=1))
        precisions = np.divide(np.diag(cm), cm.sum(axis=0))
        balanced_accuracy = np.nanmean(recalls)
        f1_per_class = 2 * np.divide(precisions * recalls, precisions + recalls)
        macro_f1 = np.nanmean(f1_per_class)
        recalls = np.nan_to_num(recalls)
        precisions = np.nan_to_num(precisions)

    # Epoch (if test=False) or final (test=True) scores
    scores = {
        'total_performance': round(float((balanced_accuracy + macro_f1 + accuracy) / 3), 4),
        'accuracy': round(float(accuracy), 4),
        'balanced_accuracy': round(float(balanced_accuracy), 4),
        'macro F1': round(float(macro_f1), 4)
    }
    for i, cls in enumerate(range(num_classes)):
        classname = class2names[cls]
        tot = int(cm[i, :].sum())  
        scores[classname] = {
            'p': round(float(precisions[i]), 4),
            'r': round(float(recalls[i]), 4),
            'tot': tot
        }

    # Save
    if test:
        with open(os.path.join(save_dir, "final_scores.json"), 'w') as f:
            json.dump(scores, f, indent=4)
    else:
        if epoch > 0:
            with open(os.path.join(save_dir, "validation_log.json"), 'r') as f:
                old_scores = json.load(f)
            old_scores[f"Epoch {epoch}"] = scores
            with open(os.path.join(save_dir, "validation_log.json"), 'w') as f:
                json.dump(old_scores, f, indent=4)
        else:
            save_scores = {f"Epoch {epoch}": scores}
            with open(os.path.join(save_dir, "validation_log.json"), 'w') as f:
                json.dump(save_scores, f, indent=4)


def save_combination_results(combination_cm, combination_id, n_classes, class2names, epochs):
    '''
    Save results of a combination of hyperparams. Used in the cross-validation script
    Inputs:
    - combination_cm: combination confusion matrix 
    - combination_id: combination name
    - n_classes: number of total classes
    - class2names: dict that link a class id to its name
    - epochs: estimated best number of training epochs for the combination

    File saved:
    - total confusion matrix 
    - heatmap of the total confusion matrix (normalized)
    - global and per-class scores extracted from the confmat
    '''
    np.save(os.path.join('results', combination_id, 'confmat.npy'), combination_cm)

    normalized_cm = combination_cm.astype(np.float32)
    row_sums = normalized_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.divide(normalized_cm, row_sums, where=row_sums != 0)
    
    # Names of used classes
    class_names = [class2names[cls] for cls in range(n_classes)]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(normalized_cm, annot=False, fmt=".2f", cmap="YlOrRd",
                     xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    
    # Tick styling
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names, rotation=0)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    
    os.makedirs('results/cross_validation', exist_ok=True)
    
    # File save
    filename = os.path.join('results', 'cross_validation', combination_id, "confmat.png")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    # Metrics saved in results/scores.json
    with np.errstate(divide='ignore', invalid='ignore'):
        recalls = np.divide(np.diag(combination_cm), combination_cm.sum(axis=1))
        precisions = np.divide(np.diag(combination_cm), combination_cm.sum(axis=0))
        balanced_accuracy = np.nanmean(recalls)
        f1_per_class = 2 * np.divide(precisions * recalls, precisions + recalls)
        macro_f1 = np.nanmean(f1_per_class)
        recalls = np.nan_to_num(recalls)
        precisions = np.nan_to_num(precisions)
    accuracy = combination_cm.trace() / combination_cm.sum()

    # If file already exists, load and update it. Otherwise create it from scratch
    if os.path.exists('results/cross_validation/scores.json'):
        with open('results/cross_validation/scores.json', 'r') as f:
            scores = json.load(f)
    else:
        scores = {}

    # Current combination dict
    combination_scores = {
        'total_performance': round(float((balanced_accuracy + macro_f1 + accuracy) / 3), 4),
        'epochs': epochs,
        'accuracy': round(float(accuracy), 4),
        'balanced_accuracy': round(float(balanced_accuracy), 4),
        'macro F1': round(float(macro_f1), 4)
    }

    for i, cls in enumerate(range(n_classes)):
        classname = class2names[cls]
        tot = int(combination_cm[i, :].sum())  
        combination_scores[classname] = {
            'p': round(float(precisions[i]), 4),
            'r': round(float(recalls[i]), 4),
            'tot': tot
        }

    # Adding combination to general dict
    scores[combination_id] = combination_scores

    # File saving
    with open('results/cross_validation/scores.json', 'w') as f:
        json.dump(scores, f, indent=4)


def save_subfold_results(train_losses, val_losses, combination_id, fold_id, subfold_id):
    '''
    Save results of a specific subfold
    Inputs: 
    - train_losses: list of all training losses for each epoch
    - val_losses: list of all validation losses for each epoch
    - combination_id: name of the hparams combination
    - fold_id: id of the current fold
    - subfold_id: id of the current subfold

    Files saved:
    - Plot of the training and validation losses for each epoch
    '''
    # Filepath
    save_path = os.path.join('results', 'cross_validation', combination_id, 'fold_' + str(fold_id))
    os.makedirs(save_path, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", color="blue", linewidth=2.5)
    plt.plot(val_losses, label="Validation Loss", color="orange", linewidth=2.5)

    # Labels, legend and title
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    if subfold_id is not None:
        plt.title(f"Fold {fold_id} – Subfold {subfold_id}")
    else:
        plt.title(f"Fold {fold_id} - epoch_selection")
    plt.legend()
    plt.grid(True)

    # Saving
    if subfold_id is not None:
        filename = os.path.join(save_path, f"{subfold_id}.png")
    else:
        filename = os.path.join(save_path, "epoch_selection.png")
    plt.savefig(filename)
    plt.close()


def save_fold_results(fold_cm, combination_id, fold_id, n_classes, class2names, f1, acc):
    '''
    Save results of a fold training and validation.
    Inputs:
    - fold_cm: confusion matrix of the fold
    - combination_id: name of the hparams combination
    - fold_id: current fold id
    - n_classes: number of total classes
    - class2names: dict that link a class id to its name
    - f1: macro f1 score obtained in the fold
    - acc: balanced accuracy obtained in the fold

    Files saved:
    - metrics.json: macro f1 and balanced accuracy
    - confusion_matrix.png: heatmap of the normalized confusion matrix
    - confusion_matrix_raw.npy: normal confusion matrix
    '''
    fold_dir = os.path.join('results', 'cross_validation', combination_id, 'fold_' + str(fold_id))
    os.makedirs(fold_dir, exist_ok=True)

    # Metadata saving
    metrics = {
        "accuracy": fold_cm.trace() / fold_cm.sum(),
        "macro_f1_score": f1,
        "balanced_accuracy": acc
    }
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Confmat saving
    normalized_cm = fold_cm.astype(np.float32)
    row_sums = normalized_cm.sum(axis=1, keepdims=True)
    normalized_cm = np.where(row_sums != 0, normalized_cm / row_sums, 0)
    
    # Names of used classes
    class_names = [class2names[cls] for cls in range(n_classes)]

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(normalized_cm, annot=False, fmt=".2f", cmap="YlOrRd",
                     xticklabels=class_names, yticklabels=class_names, cbar=True)
    
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    
    # Tick styling
    ax.set_xticklabels(class_names, rotation=90)
    ax.set_yticklabels(class_names, rotation=0)
    ax.yaxis.set_label_position("left")
    ax.yaxis.tick_left()
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    plt.title(f"Confusion Matrix – Fold {fold_id}")
    plt.tight_layout()
    plt.savefig(os.path.join(fold_dir, "confusion_matrix.png"))
    plt.close()

    # Normal confmat saving
    np.save(os.path.join(fold_dir, "confusion_matrix_raw.npy"), fold_cm)


def save_classification_examples(images, labels, preds, confs, combination_id, fold_id, class2names):
    '''
    Save examples of prediction from the model for a specific fold in a combination
    Inputs:
    - images: list with all the inferred images
    - labels: list with all the actual labels of each image
    - preds: list with all the model predictions for each image
    - confs: list with all the confidence scores of the model for each prediction
    - combination_id: name of the combination
    - fold_id: current fold
    - class2names: dict that link a class id to its name

    Files saved:
    - classification_examples.png: grid of 16 examples of classification for that fold
    '''
    # Select 16 random samples
    indices = random.sample(range(len(images)), 16)

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.flatten()

    for idx, ax in zip(indices, axes):
        img = images[idx]
        true_label = labels[idx].item()
        pred_label = preds[idx].item()
        confidence = confs[idx].item()

        img_np = img.permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1) 

        true_name = class2names[true_label]
        pred_name = class2names[pred_label]
        title_color = "green" if true_label == pred_label else "red"

        ax.imshow(img_np)
        ax.set_title(f"True: {true_name}\nPred: {pred_name}\np = {confidence * 100:.1f}%", color=title_color, fontsize=9)
        ax.axis('off')

    plt.tight_layout()
    save_dir = os.path.join('results', 'cross_validation',  combination_id, f"fold_{fold_id}")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "classification_examples.png"))
    plt.close()


def save_fold_checkpoint(fold_id, combination_cm, balanced_accuracies, f1s, best_epochs_per_fold, completed_folds):
    '''
    Fold-checkpointing. What's saved:
    - list of completed folds for the current hparams combination
    - accumulated combination confusion matrix so far
    - list of the balanced accuracies obtained for the completed folds
    - list of macro f1 scores obtained for the completed folds
    - list of best training epochs obtained for each of the completed folds
    '''
    checkpoint_data = {
        "completed_folds": completed_folds,
        "combination_cm": combination_cm.tolist(),
        "balanced_accuracies": balanced_accuracies,
        "f1s": f1s,
        "best_epochs_per_fold": best_epochs_per_fold
    }
    torch.save(checkpoint_data, os.path.join("checkpoints", f"fold_checkpoint.pt"))


def save_combinations_summary():
    # Scores loading
    with open('results/cross_validation/scores.json', 'r') as f:
        data = json.load(f)

    # Dataframe conversion
    df = pd.DataFrame.from_dict(data, orient='index')
    df.reset_index(inplace=True)
    df = df.rename(columns={"index": "combination"})

    # Combination names compression
    def compress_name(name):
        numbers = re.findall(r'\d+(?:\.\d+)?', name)
        return "_".join(n.replace('.', '') for n in numbers)

    df["short_name"] = df["combination"].apply(compress_name)

    # Parameters setting
    metrics = ["total_performance", "accuracy", "balanced_accuracy", "macro F1"]
    colors = ["green", "red", "blue", "orange"]
    alphas = [1.0, 0.3, 0.3, 0.3]
    bar_width = 0.2
    max_per_subplot = 6

    # Number of total subplots
    n_combinations = len(df)
    n_subplots = m.ceil(n_combinations / max_per_subplot)

    # Plot
    fig, axes = plt.subplots(n_subplots, 1, figsize=(12, 4 * n_subplots), constrained_layout=True)

    if n_subplots == 1:
        axes = [axes]  

    for i in range(n_subplots):
        ax = axes[i]
        start = i * max_per_subplot
        end = min(start + max_per_subplot, n_combinations)
        df_chunk = df.iloc[start:end]

        y_pos = list(range(len(df_chunk)))

        # Max metrics value
        max_val = df_chunk[metrics].values.max()
        xlim_max = min(1.0, max_val + 0.05)

        for j, metric in enumerate(metrics):
            values = df_chunk[metric]
            offset = (j - 1.5) * bar_width
            bars = ax.barh(
                [y + offset for y in y_pos],
                values,
                height=bar_width,
                color=colors[j],
                alpha=alphas[j],
                label=metric if i == 0 else None  
            )

        # Labels
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_chunk["short_name"])
        ax.invert_yaxis()
        ax.set_xlim(0, xlim_max)
        ax.set_title(f"Hyperparameter combinations [{start + 1}–{end}]")

        # Epochs showing
        for y_idx, (y, epoch) in enumerate(zip(y_pos, df_chunk["epochs"])):
            ax.text(
                xlim_max - 0.01,     
                y, 
                f"{int(epoch)} ep", 
                va="center", 
                ha="right",
                fontsize=9,
                color="gray"
            )

    plt.suptitle("Comparison of Hyperparameter Combinations", fontsize=16)

    # Legend
    handles, labels = axes[0].get_legend_handles_labels()
    # Position
    fig.legend(
        handles,
        labels,
        loc='lower center',
        ncol=4,
        bbox_to_anchor=(0.5, -0.02),  
        frameon=False
    )
    plt.subplots_adjust(bottom=0.2)  
    plt.savefig("results/cross_validation/combinations_summary.png", dpi=300)
    plt.close()