from CNNclassifier import CNNClassifier
from ViT.transformer import VIT
import utils.train_utils as train_utils
import utils.data_utils as data_utils
import torch
from tqdm import tqdm
import json
import os
from ruamel.yaml import YAML
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
    with open("configs/train_only_config.yaml") as f:
        config_raw = yaml.load(f)
    config = to_pure_dict(config_raw)
    data_config = config['data_config']
    train_config = config['train_config']
    if train_config['model_class'] == 'ViT': 
        train_config['model_config']['im_channels'] = data_config['im_ch']
        train_config['model_config']['image_width'] = data_config['im_size']
        train_config['model_config']['image_height'] = data_config['im_size']
        train_config['model_config']['num_classes'] = data_config['num_classes']
    elif train_config['model_class'] == 'CNN':  
        model_config['im_ch'] = data_config['im_ch']
        model_config['num_classes'] = data_config['num_classes']
        model_config['im_size'] = data_config['im_size']
    return data_config, train_config


if __name__ == "__main__":
    # -----------------------------------------      Initialization      -------------------------------------------
    # Working device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    data_config, train_config = parse_config()

    # Load and assert data
    data = data_utils.load_assert_data_file(data_config['filename'] + '.pkl')

    # Load class2names dict
    with open('class2names.json', 'r') as f:
        class2names_str_keys = json.load(f)
        class2names = {int(k): v for k, v in class2names_str_keys.items()}

    # Create a summary of the data classes
    if data_config['save_data_summary']:
        if not os.path.exists('data_summary'):
            os.makedirs('data_summary')
        data_utils.show_random_data(data, class2names)
        data_utils.show_frequencies(data, class2names)

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Checkpoints folder creation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # model, optimizator and loader initialization
    if train_config['model_class'] == 'ViT': 
        classifier = VIT(model_config).to(device)
    elif train_config['model_class'] == 'CNN':  
        classifier = CNNClassifier(model_config).to(device)
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])
    train_loader = data_utils.create_dataloaders(data, train_config, data_config, mode='train')

    # Check for model checkpoints. ckpt name = full_train_classifier_n.pth
    starting_epoch = 0
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and filename.startswith('full_train'):
            starting_epoch = int(filename.split('_')[-1].split('.')[0])
            print('Loading checkpoint from epoch {}...'.format(starting_epoch))
            classifier.load_state_dict(torch.load(filename))
            break


    # ----------------------------------------     Training     -----------------------------------------------
    print('Training for {} epochs...'.format(train_config['n_epochs']))
    losses = []
    for epoch in tqdm(range(starting_epoch, train_config['n_epochs'])):
        classifier, optimizer, loss = train_utils.train_model(classifier, optimizer, train_loader, criterion, device)
        losses.append(loss)
        # model checkpointing
        for filename in os.listdir('checkpoints'):
            if filename.endswith('.pth') and filename.startswith('full_train'):
                os.remove(os.path.join('checkpoints', filename))
        torch.save(classifier.state_dict(), 'checkpoints/full_train_classifier_{}.pth'.format(epoch))

    # Save training losses
    os.makedirs(os.path.join('results', 'train_only'), exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.plot(losses, label="Training Loss", color="blue", linewidth=2.5)
    
    # Labels, legend and title
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training loss")
    plt.legend()
    plt.grid(True)
    
    # Saving
    filename = os.path.join('results', 'train_only', 'training_loss.png')
    plt.savefig(filename)
    plt.close()

    # checkpoints deleting and final model saving
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and filename.startswith('full_train'):
            os.remove(os.path.join('checkpoints', filename))
    torch.save(classifier.state_dict(), 'full_train_classifier.pth')
