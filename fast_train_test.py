import json
from ruamel.yaml import YAML
import utils.utils as utils
import utils.data_utils as data_utils
import utils.train_utils as train_utils
from CNNclassifier import CNNClassifier
from ViT.transformer import VIT
import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
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
    with open("configs/fast_train_config.yaml") as f:
        config_raw = yaml.load(f)
    config = to_pure_dict(config_raw)
    data_config = config['data_config']
    train_config = config['train_config']
    model_config = config['model_config']
    if train_config['model_class'] == 'ViT': 
        model_config['im_channels'] = data_config['im_ch']
        model_config['image_width'] = data_config['im_size']
        model_config['image_height'] = data_config['im_size']
        model_config['num_classes'] = data_config['num_classes']
    elif train_config['model_class'] == 'CNN':  
        model_config['im_ch'] = data_config['im_ch']
        model_config['num_classes'] = data_config['num_classes']
        model_config['im_size'] = data_config['im_size']
    return data_config, train_config, model_config

if __name__ == "__main__":
    # -------------------------------------------------      Initialization      ---------------------------------------------------
    # Working device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    data_config, train_config, model_config = parse_config()

    # Load and assert data
    data = data_utils.load_assert_data_file(data_config['filename'] + '.pkl')

    # Load class2names dict
    with open('class2names.json', 'r') as f:
        class2names_str_keys = json.load(f)
        class2names = {int(k): v for k, v in class2names_str_keys.items()}

    # Create a summary of the data classes, if specified in the config
    if data_config['save_data_summary']:
        if not os.path.exists('data_summary'):
            os.makedirs('data_summary')
        data_utils.show_random_data(data, class2names)
        data_utils.show_frequencies(data, class2names)

    # Loaders creation
    splits = {'train': train_config['train_split'], 'val': train_config['val_split'], 'test': train_config['test_split']}
    train_loader, val_loader, test_loader = data_utils.create_dataloaders(data, train_config, data_config, mode='train_val_test', splits=splits)

    # Results and checkpoint folder creation
    os.makedirs('results/fast_train', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)

    # Model creation and loading. Checkpoint names: fast_train_classifier_n.pth or best_fast_train_classifier_n.pth
    if train_config['model_class'] == 'ViT': 
        classifier = VIT(model_config).to(device)
    elif train_config['model_class'] == 'CNN':  
        classifier = CNNClassifier(model_config).to(device)
    print("Total classifier's parameters: ", sum(p.numel() for p in classifier.parameters()))
    starting_epoch = 0
    all_ckpts = []
    epochs = []
    # Find every fast_train ckpt
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and 'fast_train' in filename: 
            epoch = int(filename.split('_')[-1].split('.')[0])
            all_ckpts.append(filename)
            epochs.append(epoch)
    # Load last epoch ckpt
    if len(epochs) > 0:
        starting_epoch = max(epochs)
        print('Loading checkpoint from epoch {}...'.format(starting_epoch))
        classifier.load_state_dict(torch.load(os.path.join('checkpoints', all_ckpts[int(np.argmax(epochs))])))

    # Optimizer and loss
    optimizer = torch.optim.Adam(classifier.parameters(), lr = train_config['lr'])
    criterion = torch.nn.CrossEntropyLoss()

    # Training and validation losses
    if starting_epoch == 0:
        best_performance = np.inf
        train_losses = []
        val_losses = []
    else:
        val_losses = list(np.load('checkpoints/fast_train_val_losses.npy'))
        train_losses = list(np.load('checkpoints/fast_train_train_losses.npy'))
        best_performance = min(val_losses)

    # ---------------------------------------------------      Training     ------------------------------------------------------
    for epoch in tqdm(range(starting_epoch, train_config['n_epochs'])):
        # Training epoch
        classifier.train()
        classifier, optimizer, train_loss = train_utils.train_model(classifier, optimizer, train_loader, criterion, device)
        train_losses.append(train_loss)

        # Validation for the current epoch: scores, examples, confusion matrix
        print('Validation...')
        classifier.eval()
        with torch.no_grad():
          utils.show_inference_samples(classifier, val_loader, device, class2names, epoch, test=False)
          val_loss, all_preds, all_labels = train_utils.validate_model(classifier, val_loader, criterion, device)
          val_losses.append(val_loss)
          utils.save_validation_summary(all_preds=all_preds, all_labels=all_labels, class2names=class2names, epoch=epoch, test=False)

        # Log and losses checkpointing
        log_str = "Epoch {} finished | train loss {:.4f} | val loss {:.4f}".format(epoch+1, train_loss, val_loss)
        print(log_str)
        np.save('checkpoints/fast_train_train_losses.npy', np.array(train_losses))
        np.save('checkpoints/fast_train_val_losses.npy', np.array(val_losses))

        # Model checkpointing
        if val_loss < best_performance:
            best_performance = val_loss
            for filename in os.listdir('checkpoints'):
                if filename.endswith('.pth') and 'fast_train' in filename:
                    os.remove(os.path.join('checkpoints', filename))
            torch.save(classifier.state_dict(), os.path.join('checkpoints', 'best_fast_train_classifier_{}.pth'.format(epoch)))
            print("Epoch {}: best model saved".format(epoch+1))
        else:
            for filename in os.listdir('checkpoints'):
                if filename.endswith('.pth') and not filename.startswith('best') and 'fast_train' in filename:
                    os.remove(os.path.join('checkpoints', filename))
            torch.save(classifier.state_dict(), os.path.join('checkpoints', 'fast_train_classifier_{}.pth'.format(epoch)))

    # Loss visualization at the end of training loop
    os.makedirs('results/fast_train', exist_ok=True)
    # plot
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label="Training Loss", color="blue", linewidth=2.5)
    plt.plot(val_losses, label="Validation Loss", color="orange", linewidth=2.5)
    # Labels, legend and title
    plt.xlabel("Epochs")
    plt.ylabel("Losses")
    plt.title("Training losses")
    plt.legend()
    plt.grid(True)
    # Saving
    plt.savefig('results/fast_train/training_losses.png')
    plt.close()

    # Loss checkpoints deleting
    os.remove('checkpoints/fast_train_train_losses.npy')
    os.remove('checkpoints/fast_train_val_losses.npy')

    # ---------------------------------------------------      Testing     ------------------------------------------------------
    # Best ckpt loading
    if train_config['model_class'] == 'ViT': 
        classifier = VIT(model_config).to(device)
    elif train_config['model_class'] == 'CNN':
        classifier = CNNClassifier(model_config).to(device)
    for filename in os.listdir('checkpoints'):
        # ckpt name = best_fast_train_classifier_n.pth
        if filename.endswith('.pth') and filename.startswith('best') and 'fast_train' in filename: 
            print('Loading best checkpoint...')
            classifier.load_state_dict(torch.load(os.path.join('checkpoints', filename)))

    # Testing: scores, examples, confusion matrix
    classifier.eval()
    with torch.no_grad():
        utils.show_inference_samples(classifier, test_loader, device, class2names, epoch=None, test=True, n=25)
        val_loss, all_preds, all_labels = train_utils.validate_model(classifier, val_loader, criterion, device)
        utils.save_validation_summary(all_preds=all_preds, all_labels=all_labels, class2names=class2names, epoch=None, test=True)

    # Saving final classifier ckpt in main dir and deleting previous ckpts
    torch.save(classifier.state_dict(), os.path.join('fast_train_classifier.pth'))
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and 'fast_train' in filename:
            os.remove(os.path.join('checkpoints', filename))
