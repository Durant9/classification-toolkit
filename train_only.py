from CNNclassifier import CNNClassifier
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
    with open("config/train_only_config.yaml") as f:
        config_raw = yaml.load(f)
    config = to_pure_dict(config_raw)
    data_config = config['data_config']
    train_config = config['train_config']
    train_config['model_config']['im_ch'] = data_config['im_ch']
    train_config['model_config']['num_classes'] = data_config['num_classes']
    train_config['model_config']['im_size'] = data_config['im_size']
    return data_config, train_config


if __name__ == "__main__":
    # -----------------------------------------      Initialization      -------------------------------------------
    # Working device
    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load config
    data_config, train_config = parse_config()

    # Load and assert data
    data = data_utils.load_assert_data_file(data_config['filename'] + '.pkl')

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

    # Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Checkpoints folder creation
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    # model, optimizator and loader initialization
    classifier = CNNClassifier(train_config['model_config']).to(device)
    classifier.train()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=train_config['lr'])
    train_loader = data_utils.create_dataloaders(data, train_config, data_config, train_only=True)

    # Check for model checkpoints. ckpt name = full_train_classifier_n.pth
    starting_epoch = 0
    all_ckpts = []
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and filename.startswith('full_train'):
            starting_epoch = int(filename.split('_')[-1].split('.')[0])
            print('Loading checkpoint from epoch {}...'.format(starting_epoch))
            classifier.load_state_dict(torch.load(filename))
            break


    # ----------------------------------------     Training     -----------------------------------------------
    print('Training for {} epochs...'.format(train_config['n_epochs']))
    for epoch in tqdm(range(starting_epoch, train_config['n_epochs'])):
        classifier, optimizer, _ = train_utils.train_model(classifier, optimizer, train_loader, criterion, device)
        # model checkpointing
        for filename in os.listdir('checkpoints'):
            if filename.endswith('.pth') and filename.startswith('full_train'):
                os.remove(filename)
        torch.save(classifier.state_dict(), 'full_train_classifier_{}.pth'.format(epoch))

    # checkpoints deleting and final model saving
    for filename in os.listdir('checkpoints'):
        if filename.endswith('.pth') and filename.startswith('full_train'):
            os.remove(filename)
    torch.save(classifier.state_dict(), 'full_train_classifier.pth')
