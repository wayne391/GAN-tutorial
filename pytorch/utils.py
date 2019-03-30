import os
import torch


class Saver(object):
    def __init__(self, save_dir):
        self.save_dir = save_dir

        # folder for saving
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # log
        log_file = os.path.join(self.save_dir, 'log.txt')
        if os.path.exists(log_file):
            self.logFile = open(log_file, 'a')
        else:
            self.logFile = open(log_file, 'w')

    def save_log(self, log):
        self.logFile.write(log + '\n')

    def save_model(self, model, name='model'):
        torch.save(
            model.state_dict(),
            os.path.join(self.save_dir, name + '_state.pt'))
        # torch.save(
        #     model,
        #     os.path.join(self.save_dir, name+'_obj.pt'))

    def save_multiple_model(self, model_dict):
        new_dict = dict()
        for k in model_dict.keys():
            new_dict[k] = model_dict[k].state_dict()
        torch.save(new_dict, os.path.join(self.save_dir, 'model_state.pt'))

    def load_model(self, model, name='model'):
        load_path = os.path.join(self.save_dir, name + '_state.pt')
        model.load_state_dict(torch.load(load_path))
        print(" [*] loaded from %s" % load_path)
        return model

    def load_multiple_model(self, model_dict):
        ckpt = torch.load(os.path.join(self.save_dir, 'model_state.pt'))
        for k in model_dict.keys():
            model_dict[k].load_state_dict(ckpt[k])
        return model_dict
