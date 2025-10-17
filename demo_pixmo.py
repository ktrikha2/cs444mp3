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

    return net_class, (num_classes,), dir_name, (optimizer, lr, epochs, batch_size)


def main(_):
    setup_logging()
    torch.set_num_threads(4)
    torch.manual_seed(0)

    print(f"Running: {FLAGS.exp_name}")

    net_class, (num_classes,), dir_name, \
        (optimizer, lr, epochs, batch_size) = \
        get_config(FLAGS.exp_name)

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

    model = net_class(num_classes)
    model.to(device)

    trainer = Trainer(model, train_dataloader, val_dataloader, writer,
                      optimizer=optimizer, lr=lr, epochs=epochs, device=device)

    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"lr: {lr:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")

    print("Training complete--------------------")

    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model.load_state_dict(torch.load(f'{dir_name}/best_model.pth', weights_only=True))
    inference(test_dataloader, model, device, result_path=dir_name + '/test_pixmo.txt')


if __name__ == '__main__':
    app.run(main)
