import torch
import sys
import tqdm
import logging
from absl import app, flags
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import yaml
from model import PixMoModel, Trainer, inference
from datasets import get_pixmo_data

FLAGS = flags.FLAGS
flags.DEFINE_string('exp_name', 'pixmo_demo',
                    'The experiment with corresponding hyperparameters to run. See config.yaml')
flags.DEFINE_string('output_dir', 'runs_pixmo', 'Output Directory')
flags.DEFINE_string('data_dir', './pixmo_data', 'Directory with pixmo data')

def setup_logging():
    log_formatter = logging.Formatter(
        '%(asctime)s: %(levelname)s %(filename)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logging.getLogger().handlers = []
    if len(logging.getLogger().handlers) == 0: 
        logging.getLogger().addHandler(console_handler)
    logging.getLogger().setLevel(logging.INFO)

def logger(tag, value, global_step):
    if tag == '':
       logging.info('')
    else:
       logging.info(f'  {tag:>15s} [{global_step:07d}]: {value:5f}')

class SummaryWriterWithPrinting(SummaryWriter):
    def add_scalar(self, tag, value, global_step): 
        super(SummaryWriterWithPrinting, self).add_scalar(tag, value, global_step)
        logger(tag, value, global_step)

def get_config(exp_name):
    dir_name = f'{FLAGS.output_dir}'

    # add/modify hyperparameters of your class in config.yaml
    encoder_registry = {
        'PixMoModel': PixMoModel,
    }
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    epochs = config['epochs']
    optimizer = config['optimizer']
    net_class = encoder_registry[config['net_class']]
    batch_size = config['batch_size']
    num_classes = config['num_classes']
    
    # Extract model-specific parameters with defaults
    model_params = {
        'num_classes': num_classes,
        'feature_dim': config.get('in_features', 512),
        'num_heads': config.get('num_heads', 2),
        'num_layers': config.get('num_layers', 2),
        'patch_tokens': config.get('patch_tokens', 1),
        'dropout': config.get('dropout', 0.1)
    }
    
    # Extract trainer-specific parameters
    trainer_params = {
        'optimizer': optimizer,
        'lr': lr,
        'wd': config.get('weight_decay', 0.01),
        'momentum': config.get('momentum', 0.99),
        'scheduler': config.get('scheduler', None),
        'epochs': epochs
    }
    
    return net_class, model_params, trainer_params, dir_name, batch_size

    return net_class, model_params, trainer_params, dir_name, batch_size


def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    print(f"Running: {FLAGS.exp_name}")

    net_class, model_params, trainer_params, dir_name, batch_size = \
        get_config(FLAGS.exp_name)

    # Log the configuration being used
    print("\n" + "="*60)
    print("MODEL CONFIGURATION:")
    for key, value in model_params.items():
        print(f"  {key}: {value}")
    print("\nTRAINER CONFIGURATION:")
    for key, value in trainer_params.items():
        print(f"  {key}: {value}")
    print(f"  batch_size: {batch_size}")
    print("="*60 + "\n")

    train_data = get_pixmo_data(FLAGS.data_dir,'train')
    val_data = get_pixmo_data(FLAGS.data_dir,'val')
    test_data = get_pixmo_data(FLAGS.data_dir,'test')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)

    tmp_file_name = dir_name + '/best_model.pth'
    device = torch.device('cuda:0')
    # For Mac Users
    # device = torch.device('mps')

    writer = SummaryWriterWithPrinting(f'{dir_name}', flush_secs=10)

    # Create model with all parameters from config
    model = net_class(**model_params)
    model.to(device)

    # Create trainer with all parameters from config
    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                      device=device, **trainer_params)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"\nlr: {trainer_params['lr']:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")

    print("Training complete--------------------")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{dir_name}/best_model.pth', weights_only=True))
    inference(test_dataloader, model, device, result_path=dir_name + '/test_pixmo.txt')


if __name__ == '__main__':
    app.run(main)