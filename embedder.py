import numpy as np
import torch
import torch.nn as nn

from argument import printConfig, config2string
from tensorboardX import SummaryWriter

from models import LogisticRegression
from torch_geometric.nn import GCNConv
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)


class embedder:

    def __init__(self, args):
        self.args = args
        self.hidden_layers = args.layers
        printConfig(args)

        self.config_str = config2string(args)
        print("\n[Config] {}\n".format(self.config_str))
        self.writer = SummaryWriter(log_dir="runs/{}".format(self.config_str))

        # Model Checkpoint Path
        CHECKPOINT_PATH = "RGRL_checkpoints/"
        self.check_dir = CHECKPOINT_PATH + self.config_str + ".pt"

    def infer_embeddings(self):
        
        self._model.train(False)
        self._embeddings = self._labels = None

        self._dataset.data.to(self._device)
        _, _, _, _, embeddings = self._model(
            x1=self._dataset.data.x, x2=self._dataset.data.x,
            edge_index1=self._dataset.data.edge_index,
            edge_index2=self._dataset.data.edge_index,
            edge_weight1=self._dataset.data.edge_attr,
            edge_weight2=self._dataset.data.edge_attr)

        emb = embeddings.detach()
        y = self._dataset.data.y.detach()
        if self._embeddings is None:
            self._embeddings, self._labels = emb, y
        else:
            self._embeddings = torch.cat([self._embeddings, emb])
            self._labels = torch.cat([self._labels, y])


    def evaluate(self, epoch):

        emb_dim, num_class = self._embeddings.shape[1], self._labels.unique().shape[0]

        dev_accs, test_accs = [], []

        for i in range(20):

            self._train_mask = self._dataset.data.train_mask[i]
            self._dev_mask = self._dataset.data.val_mask[i]
            if self._args.dataset == "wikics":
                self._test_mask = self._dataset.data.test_mask
            else:
                self._test_mask = self._dataset.data.test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class).to(self._device)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

            for _ in range(100):
                classifier.train()
                logits, loss = classifier(self._embeddings[self._train_mask], self._labels[self._train_mask])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            dev_logits, _ = classifier(self._embeddings[self._dev_mask], self._labels[self._dev_mask])
            test_logits, _ = classifier(self._embeddings[self._test_mask], self._labels[self._test_mask])
            dev_preds = torch.argmax(dev_logits, dim=1)
            test_preds = torch.argmax(test_logits, dim=1)

            dev_acc = (torch.sum(dev_preds == self._labels[self._dev_mask]).float() /
                       self._labels[self._dev_mask].shape[0]).detach().cpu().numpy()
            test_acc = (torch.sum(test_preds == self._labels[self._test_mask]).float() /
                        self._labels[self._test_mask].shape[0]).detach().cpu().numpy()

            dev_accs.append(dev_acc * 100)
            test_accs.append(test_acc * 100)

        dev_accs = np.stack(dev_accs)
        test_accs = np.stack(test_accs)

        dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
        test_acc, test_std = test_accs.mean(), test_accs.std()

        print('** [{}] [Epoch: {}] Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(self.args.embedder, epoch, dev_acc, dev_std, test_acc, test_std))

        if dev_acc > self.best_dev_acc:
            self.best_dev_acc = dev_acc
            self.best_test_acc = test_acc
            self.best_dev_std = dev_std
            self.best_test_std = test_std
            self.best_epoch = epoch
            checkpoint = {'epoch': epoch, 'embeddings': self._embeddings.detach().cpu().numpy()}
            torch.save(checkpoint, self.check_dir)

        self.writer.add_scalar("acc/val_accuracy", dev_acc, epoch+1)
        self.writer.add_scalar("acc/best_val_accuracy", self.best_dev_acc, epoch+1)
        self.writer.add_scalar("acc/test_accuracy", test_acc, epoch+1)
        self.writer.add_scalar("acc/best_test_accuracy", self.best_test_acc, epoch+1)

        self.best_dev_accs.append(self.best_dev_acc)
        self.st_best = '** [Best epoch: {}] Best val | Best test: {:.4f} ({:.4f}) / {:.4f} ({:.4f})**\n'.format(
            self.best_epoch, self.best_dev_acc, self.best_dev_std, self.best_test_acc, self.best_test_std)
        print(self.st_best)


class Encoder(nn.Module):

    def __init__(self, layer_config, dropout=None, project=False, **kwargs):
        super().__init__()
        
        self.conv1 = GCNConv(layer_config[0], layer_config[1])
        self.bn1 = nn.BatchNorm1d(layer_config[1], momentum = 0.01)
        self.prelu1 = nn.PReLU()
        self.conv2 = GCNConv(layer_config[1],layer_config[2])
        self.bn2 = nn.BatchNorm1d(layer_config[2], momentum = 0.01)
        self.prelu2 = nn.PReLU()

    def forward(self, x, edge_index, edge_weight=None):
        
        x = self.conv1(x, edge_index, edge_weight=edge_weight)
        x = self.prelu1(self.bn1(x))
        x = self.conv2(x, edge_index, edge_weight=edge_weight)
        x = self.prelu2(self.bn2(x))
        
        return x
