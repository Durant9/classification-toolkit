import utils.utils as utils
import utils.data_utils as data_utils
import utils.train_utils as train_utils
from CNNclassifier import CNNClassifier

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap
import json
import os
import torch
import numpy as np
import random
from tqdm import tqdm
from copy import deepcopy
import itertools
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


def to_pure_dict(obj):
    if isinstance(obj, dict):
        return {k: to_pure_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_pure_dict(i) for i in obj]
    else:
        return obj

def parse_config():
    yaml = YAML()
    with open("config/cross_validation_config.yaml") as f:
        config_raw = yaml.load(f)
    config = to_pure_dict(config_raw)
    data_config = config['data_config']
    train_config = config['train_config']
    hyperparams = config['hyperparameters']
    train_config['model_config']['im_ch'] = data_config['im_ch']
    train_config['model_config']['num_classes'] = data_config['num_classes']
    train_config['model_config']['im_size'] = data_config['im_size']
    return data_config, train_config, hyperparams


if __name__ == "__main__":
    # -----------------------------------------      Initialization      -------------------------------------------
    # Working device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    data_config, train_config, hyperparams = parse_config()

    # Assert hyperparameters configuration
    utils.assert_hparams(hyperparams)

    # Load and assert data
    data = data_utils.load_assert_data_file(data_config['filename'] + '.pkl')

    # No classes with less than n_folds samples
    for key in list(data.keys()):
        if len(data[key]) < train_config['n_folds']:
            del data[key]

    # IDs of used classes
    used_classes = [cls - 1 for cls in sorted(data.keys())]

    # Load class2names dict
    with open('class2names.json', 'r') as f:
        class2names_str_keys = json.load(f)
        class2names = {int(k) - 1: v for k, v in class2names_str_keys.items()}

    # Create a summary of the data classes
    if data_config['save_data_summary']:
        if not os.path.exists('data_summary'):
            os.makedirs('data_summary')
        data_utils.show_random_data(data, class2names)
        data_utils.show_frequencies(data, class2names)

    # Folds creation
    folds = data_utils.create_folds(data, n_folds=train_config['n_folds'])

    # Results and checkpoint folder creation
    if not os.path.exists('cross_validation_results'):
        os.makedirs('cross_validation_results')
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # Loss
    criterion = torch.nn.CrossEntropyLoss()


    # --------------------------------------     Hyperparameters combinations     --------------------------------------
    if hyperparams:
        # Checkpoint loading, if existing
        if os.path.exists('checkpoints/combinations_info.json'):
            with open('checkpoints/combinations_info.json') as f:
                combinations_info = json.load(f)
                hyperparams_combinations = combinations_info['selected']
                completed_combinations = combinations_info['completed']

        # Otherwise, creation of combinations from scratch
        else:
            # All possible combinations
            keys, values = zip(*hyperparams.items())
            all_hyperparams_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

            # Grid search: validation with all combination
            if train_config['hparams_search'] == 'grid':
                hyperparams_combinations = all_hyperparams_combinations

            # Random search: validation with only part of all combinations:
            # if < 20 --> 50%
            # if < 50 --> 35%
            # if > 50 --> 20%
            elif train_config['hparams_search'] == 'random':
                if len(all_hyperparams_combinations) < 20:
                    n_samples = round(len(all_hyperparams_combinations) / 2)
                elif len(all_hyperparams_combinations) < 50:
                    n_samples = round((3.5 * len(all_hyperparams_combinations)) / 10)
                else:
                    n_samples = round(len(all_hyperparams_combinations) / 5)
                hyperparams_combinations = random.sample(all_hyperparams_combinations, k=n_samples)

            # Combination selection saving
            completed_combinations = []
            with open('checkpoints/combinations_info.json', 'w') as f:
                json.dump({'selected': hyperparams_combinations, 'completed': completed_combinations}, f, indent=2)
    # If there's no hyperparams, empty dict instantiation
    else:
        hyperparams_combinations = [dict()]
    hyperparams_names = list(hyperparams.keys())
    hyperparams_values = list(hyperparams.values())


    # ----------------------------------     Hyperparameters optimization matrices     ---------------------------------
    # Checkpoints loading, if existing
    if os.path.exists('checkpoints/performance_matrix.npy'):
        performance_matrix = np.load('checkpoints/performance_matrix.npy')
        best_epoch_per_combination = np.load('checkpoints/best_epoch_matrix.npy')
    # Otherwise, creation from scratch
    else:
        if train_config['hparams_search'] == 'grid':
            performance_matrix = np.zeros([len(values) for values in hyperparams_values])
            best_epoch_per_combination = np.zeros([len(values) for values in hyperparams_values])
        elif train_config['hparams_search'] == 'random':
            performance_matrix = np.zeros(len(hyperparams_combinations))
            best_epoch_per_combination = np.zeros(len(hyperparams_combinations))


    # -----------------------------------------------------------------------------------------------------------------
    # ---------------------------------------------      Validation     -----------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------
    # Hyperparameters optimization: k-fold validation on the selected combinations
    for values_comb in hyperparams_combinations:
        # Combination name definition
        # ex: if lr = 0.001, batch size = 64 and 2nd model config is used: 'lr_0001_batch_size_64_model_config1'
        combination_id_parts = []
        for name in hyperparams_names:
            value = values_comb[name]
            if isinstance(value, dict):
                idx = hyperparams[name].index(value)
                combination_id_parts.append(f"{name}{idx}")
            else:
                combination_id_parts.append(f"{name}{utils.clean_value(value)}")
        combination_id = "_".join(combination_id_parts)

        # If combination was already studied earlier, pass to the next one
        if combination_id in completed_combinations:
            print(f"[INFO] Skipping already completed combination: {combination_id}")
            continue

        # Position of the combination in the optimization matrices
        if train_config['hparams_search'] == 'grid':
            indices = [hyperparams[name].index(values_comb[name]) for name in hyperparams_names]
        elif train_config['hparams_search'] == 'random':
            indices = hyperparams_combinations.index(values_comb)

        # Config setting using current hyperparameters values
        for key, value in values_comb.items():
            train_config[key] = value
        model_config = deepcopy(train_config['model_config'])
        model_config['im_ch'] = data_config['im_ch']
        model_config['num_classes'] = data_config['num_classes']
        model_config['im_size'] = data_config['im_size']

        # Variables to fill at each fold
        best_epochs_per_fold = []
        balanced_accuracies = []
        f1s = []
        combination_cm = np.zeros((len(used_classes), len(used_classes)), dtype=int)
        completed_folds = []

        # last combination checkpoint loading
        if os.path.exists('checkpoints/fold_checkpoint.pt'):
            checkpoint_data = torch.load('checkpoints/fold_checkpoint.pt', weights_only=False)
            completed_folds = checkpoint_data['completed_folds']
            combination_cm = np.array(checkpoint_data['combination_cm'])
            balanced_accuracies = checkpoint_data['balanced_accuracies']
            f1s = checkpoint_data['f1s']
            best_epochs_per_fold = checkpoint_data['best_epochs_per_fold']

        # Start of combination validation: k-fold cross validation
        print(f"[INFO] Starting training for combination {combination_id}...")
        for fold_id in tqdm(range(len(folds))):
            # If the current fold was already studied, pass to the next one
            if fold_id in completed_folds:
                print(f"[INFO] Skipping already completed fold {fold_id}")
                continue

            # If nested k-fold validation is used
            if train_config['epoch_estimation'] == 'subfolds':
                best_epochs_per_subfold = []
                # Training and validation for each subfold
                for subfold_id in tqdm(range(len(folds) - 1)):
                    # subfold model and optimizator
                    classifier = CNNClassifier(model_config).to(device)
                    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])
                    # Subfold dataloaders
                    train_loader, val_loader, _ = data_utils.create_dataloaders(None, train_config, data_config,
                                                                                folds=folds, fold=fold_id,
                                                                                subfold=subfold_id)
                    # Subfold cumulative variables
                    subfold_best_performance = np.inf
                    best_epoch = 0
                    current_patience = 0
                    train_losses = []
                    val_losses = []
                    # Subfold training
                    for epoch in range(train_config['n_epochs']):
                        # Single training epoch
                        classifier.train()
                        classifier, optimizer, train_loss = train_utils.train_model(classifier, optimizer, train_loader,
                                                                                    criterion, device)
                        train_losses.append(train_loss)
                        # After-epoch validation
                        classifier.eval()
                        with torch.no_grad():
                            val_loss = train_utils.validate_model(classifier, val_loader, criterion, device)
                        val_losses.append(val_loss)
                        # Early stopping or best performance update
                        if val_loss < subfold_best_performance:
                            subfold_best_performance = val_loss
                            best_epoch = epoch
                            current_patience = 0
                        else:
                            current_patience += 1
                        if current_patience == train_config['patience'] or epoch == train_config['n_epochs'] - 1:
                            best_epochs_per_subfold.append(best_epoch + 1)
                            break
                    # Subfold results saving
                    utils.save_subfold_results(train_losses, val_losses, combination_id, fold_id, subfold_id)
                # Best n_epochs for the current fold
                n_epochs_fold = int(np.median(best_epochs_per_subfold))

            # If the validation is not nested
            elif train_config['epoch_estimation'] == 'holdout':
                # Model and optimizator
                classifier = CNNClassifier(model_config).to(device)
                optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])

                # Dataloaders: subfold emulation to extract validation loader
                train_loader, val_loader, _ = data_utils.create_dataloaders(None, train_config, data_config,
                                                                            folds=folds, fold=fold_id,
                                                                            subfold=np.random.randint(0, 4))

                # Subfold cumulative variables
                subfold_best_performance = np.inf
                best_epoch = 0
                current_patience = 0
                train_losses = []
                val_losses = []

                # Epochs-selection training
                for epoch in range(train_config['n_epochs']):
                    # Single training epoch
                    classifier.train()
                    classifier, optimizer, train_loss = train_utils.train_model(classifier, optimizer, train_loader,
                                                                                criterion, device)
                    train_losses.append(train_loss)
                    # After-epoch validation
                    classifier.eval()
                    with torch.no_grad():
                        val_loss = train_utils.validate_model(classifier, val_loader, criterion, device)
                    val_losses.append(val_loss)
                    # Early stopping or best performance update
                    if val_loss < subfold_best_performance:
                        subfold_best_performance = val_loss
                        best_epoch = epoch
                        current_patience = 0
                    else:
                        current_patience += 1
                    if current_patience == train_config['patience'] or epoch == train_config['n_epochs'] - 1:
                        n_epochs_fold = best_epoch + 1
                        break
                # fold epochs-selection results saving
                utils.save_subfold_results(train_losses, val_losses, combination_id, fold_id, None)

            # Best n_epochs for the current fold
            best_epochs_per_fold.append(n_epochs_fold)

            # Fold actual training
            classifier = CNNClassifier(model_config).to(device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])
            train_loader, test_loader = data_utils.create_dataloaders(None, train_config, data_config, folds=folds,
                                                                      fold=fold_id, train_test_only=True)
            classifier.train()
            for epoch in range(n_epochs_fold):
                classifier, optimizer, _ = train_utils.train_model(classifier, optimizer, train_loader, criterion,
                                                                   device)

            # Fold testing
            with torch.no_grad():
                images, preds, labels, confs = train_utils.test_model(classifier, test_loader, device)
            utils.save_classification_examples(images, labels, preds, confs, combination_id, fold_id, class2names)

            # Fold scores and confusion matrix computing
            balanced_accuracies.append(balanced_accuracy_score(labels, preds))
            f1s.append(f1_score(labels, preds, average='macro'))
            fold_cm = confusion_matrix(labels, preds, labels=used_classes)
            combination_cm += fold_cm

            # Checkpoints and results saving
            completed_folds.append(fold_id)
            utils.save_fold_checkpoint(fold_id, combination_cm, balanced_accuracies, f1s, best_epochs_per_fold,
                                       completed_folds)
            utils.save_fold_results(fold_cm, combination_id, fold_id, used_classes, class2names,
                                    balanced_accuracies[-1], f1s[-1])

        # Combination performance and best epochs saving
        accuracy = combination_cm.trace() / combination_cm.sum()
        combination_epochs = int(np.mean(best_epochs_per_fold))
        if train_config['hparams_search'] == 'grid':
            performance_matrix[tuple(indices)] = (np.mean(balanced_accuracies) + np.mean(f1s) + accuracy) / 3
            best_epoch_per_combination[tuple(indices)] = combination_epochs
        elif train_config['hparams_search'] == 'random':
            performance_matrix[indices] = (np.mean(balanced_accuracies) + np.mean(f1s) + accuracy) / 3
            best_epoch_per_combination[indices] = combination_epochs
        utils.save_combination_results(combination_cm, combination_id, used_classes, class2names, combination_epochs)

        # Combinations checkpoints updating
        np.save('checkpoints/performance_matrix.npy', performance_matrix)
        np.save('checkpoints/best_epoch_matrix.npy', best_epoch_per_combination)
        completed_combinations.append(combination_id)
        with open('checkpoints/combinations_info.json', 'w') as f:
            json.dump({'selected': hyperparams_combinations, 'completed': completed_combinations}, f, indent=2)

        # Folds checkpoint deleting
        os.remove('checkpoints/fold_checkpoint.pt')


    # --------------------------------------------     Results saving     ----------------------------------------------
    # All checkpoints deleting, hyperparameters results and json saving
    os.remove('checkpoints/best_epoch_matrix.npy')
    os.remove('checkpoints/combinations_info.json')
    os.remove('checkpoints/performance_matrix.npy')
    with open('results/hyperparams_checked.json', 'w') as f:
        json.dump({'values': hyperparams, 'combinations': hyperparams_combinations}, f)

    # Combinations summary
    utils.save_combination_summary()

    # Best hyperparams finding
    best_index = np.argmax(performance_matrix)
    best_hyperparams = hyperparams_combinations[np.argmax(performance_matrix)]
    for k, v in best_hyperparams.items():
        train_config[k] = v
    train_config['n_epochs'] = best_epoch_per_combination[best_index]

    # Best config saving
    final_config_raw = {
        'data_config': data_config,
        'train_config': train_config
    }
    del final_config_raw['train_config']['hparams_search']
    del final_config_raw['train_config']['epoch_estimation']
    del final_config_raw['train_config']['n_folds']
    del final_config_raw['train_config']['patience']

    final_config = CommentedMap()
    for i, (key, value) in enumerate(final_config_raw.items()):
        final_config[key] = value
        if i > 0:
            final_config.yaml_set_comment_before_after_key(key, before='\n')
    yaml = YAML()
    yaml.indent(mapping=4, sequence=4, offset=2)
    yaml.preserve_quotes = True
    with open("config/best_config.yaml", "w") as f:
        yaml.dump(final_config, f)

    print('[INFO] Hyperparameters selection and cross-validation completed')