import torch_geometric.transforms as T

import torch
import torch.nn as nn
from torch import optim

import torch_geometric.utils as tg_utils
import torch.nn.functional as F
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

import numpy as np
np.random.seed(0)

import os
import sys
import copy
from utils import EMA, set_requires_grad, init_weights, update_moving_average, Augmentation, KLD
from data import Dataset
from embedder import embedder
from embedder import Encoder


class RGRL_ModelTrainer(embedder):
    def __init__(self, args):
        embedder.__init__(self, args)
        self._args = args
        self._init()

    def _init(self):
        args = self._args
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        self._device = f'cuda:{args.device}' if torch.cuda.is_available() else "cpu"
        torch.cuda.set_device(self._device)

        try :
            diffusion = torch.load("./diffusion/{}_top{}.pt".format(args.dataset, args.topk))

        except :
            temp = Dataset(root=args.root, dataset=args.dataset)[0]
            data_diffusion = T.GDC(sparsification_kwargs={'k': args.topk, 'dim': 1, 'method': 'topk'})(temp)
            diffusion = data_diffusion.edge_index[1].reshape(-1, args.topk)
        
        self._dataset = Dataset(root=args.root, dataset=args.dataset)

        # Create Inverse Degree distribution
        degree = np.log(np.asarray(tg_utils.degree(self._dataset.data.edge_index[0], num_nodes = self._dataset.data.x.shape[0])) + 1)
        inv_degree = np.power(args.alpha, degree).astype('float64')
        inv_degree += args.beta
        inv_degree[inv_degree > 1] = 1
        inv_degree /= inv_degree.sum()
        not_zero = np.where(np.asarray(tg_utils.degree(self._dataset.data.edge_index[0], num_nodes = self._dataset.data.x.shape[0])) != 0)[0]

        print(f"Data: {self._dataset.data}")
        hidden_layers = [int(l) for l in args.layers]
        layers = [self._dataset.data.x.shape[1]] + hidden_layers
        self._model = RGRL(layer_config=layers, pred_hid = args.pred_hid, topk= args.sample, temperature= [args.temp_t, args.temp_s, args.temp_t_diff, args.temp_s_diff], 
                                                diffusion = diffusion, degree = inv_degree, not_zero = not_zero, device = self._device, dropout=args.dropout, epochs=args.epochs).to(self._device)
        print(self._model)

        self._optimizer = optim.AdamW(params=self._model.parameters(), lr=args.lr, weight_decay=1e-5)

    def train(self):

        f_final = open("results/{}_{}.txt".format(self._args.embedder, self._args.dataset), "a")

        self.best_test_acc, self.best_dev_acc, self.best_test_std, self.best_dev_std, self.best_epoch = 0, 0, 0, 0, 0
        self.best_dev_accs = []

        # get initial test results
        self.infer_embeddings()
        print("initial accuracy ")
        self.evaluate(0)

        criterion = KLD().to(self._device)

        # start training
        print("Training Start!")
        self._model.train()
        for epoch in range(self._args.epochs):
            augmentation = Augmentation(float(self._args.aug_params[0]), float(self._args.aug_params[1]), float(self._args.aug_params[2]), float(self._args.aug_params[3]))
            view1, view2 = augmentation._feature_masking(self._dataset.data, self._device)

            sim_q1, sim_k1, sim_q2, sim_k2, _ = self._model(x1=view1.x, x2=view2.x, edge_index1=view1.edge_index, edge_index2=view2.edge_index, 
                                    edge_weight1=view1.edge_attr, edge_weight2=view2.edge_attr)

            loss = criterion(inputs=sim_q1[0], targets=sim_k1[0]) + criterion(inputs=sim_q2[0], targets=sim_k2[0])
            loss_diffusion = criterion(inputs=sim_q1[1], targets=sim_k1[1]) + criterion(inputs=sim_q2[1], targets=sim_k2[1])
            loss = loss + self._args.lam * loss_diffusion
            self._optimizer.zero_grad()
            loss.backward()
            self._optimizer.step()
            self._model.update_moving_average()
            
            st = '[Epoch {}/{}] Loss: {:.4f} '.format(epoch, self._args.epochs, loss.item())
            print(st)

            self.writer.add_scalar("loss/training_loss", loss.item(), epoch+1)

            if (epoch) % self._args.eval_freq == 0:
                self.infer_embeddings()
                self.evaluate(epoch)

                if len(self.best_dev_accs) > int(self._args.es / self._args.eval_freq):
                    if self.best_dev_accs[-1] == self.best_dev_accs[-int(self._args.es / self._args.eval_freq)]:
                        print("Early stop!!")
                        print("[Final] {}".format(self.st_best))
                        f_final.write("[Early Stopped at {}] {} -> {}\n".format(epoch, self.config_str, self.st_best))
                        sys.exit()

        print("\nTraining Done!")
        print("[Final] {}".format(self.st_best))

        f_final.write("{} -> {}\n".format(self.config_str, self.st_best))
        f_final.close()


class RGRL(nn.Module):
    def __init__(self, layer_config, pred_hid, topk, temperature, diffusion, degree, not_zero, device, dropout=0.0, moving_average_decay=0.99, epochs=1000, **kwargs):
        super().__init__()

        self.student_encoder = Encoder(layer_config=layer_config, dropout=dropout, **kwargs)
        rep_dim = layer_config[-1]
        self.student_predictor = nn.Sequential(nn.Linear(rep_dim, pred_hid), nn.PReLU(), nn.Linear(pred_hid, rep_dim))
        self.student_predictor.apply(init_weights)

        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)
        self.teacher_ema_updater = EMA(moving_average_decay, epochs)

        self.topk = topk
        self.temp_teacher = temperature[0]
        self.temp_student = temperature[1]
        self.temp_teacher_diffusion = temperature[2]
        self.temp_student_diffusion = temperature[3]

        self.diffusion = diffusion
        self.degree = degree / degree.sum()
        self.not_zero = not_zero
        self.device = device

    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None
    
    @torch.no_grad()
    def init_student_network(self):
        for param_q, param_k in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            param_q.data = param_k.data.T
    
    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def forward(self, x1, x2, edge_index1, edge_index2, edge_weight1=None, edge_weight2=None):
        
        student1 = self.student_encoder(x = x1, edge_index = edge_index1, edge_weight = edge_weight1)
        student2 = self.student_encoder(x = x2, edge_index = edge_index2, edge_weight = edge_weight2)

        pred1 = F.normalize(self.student_predictor(student1), p = 2.0, dim = 1)
        pred2 = F.normalize(self.student_predictor(student2), p = 2.0, dim = 1)

        with torch.no_grad():
            teacher1 = F.normalize(self.teacher_encoder(x = x1, edge_index = edge_index1, edge_weight=edge_weight1), p = 2.0, dim = 1)
            teacher2 = F.normalize(self.teacher_encoder(x = x2, edge_index = edge_index2, edge_weight=edge_weight2), p = 2.0, dim = 1)

            sample = torch.tensor(np.random.choice(x1.shape[0], size = (self.topk), replace = False, p = self.degree)).to(self.device)
        
        sim_q1 = torch.mm(pred1, teacher2[sample].T) / self.temp_student
        sim_k1 = torch.mm(teacher2, teacher2[sample].T) / self.temp_teacher

        sim_q2 = torch.mm(pred2, teacher1[sample].T) / self.temp_student
        sim_k2 = torch.mm(teacher1, teacher1[sample].T) / self.temp_teacher

        sim_q1_diffusion = torch.bmm(teacher2[self.diffusion], pred1.reshape(pred1.shape[0], pred1.shape[1], 1)).squeeze(2) / self.temp_student_diffusion
        sim_k1_diffusion = torch.bmm(teacher2[self.diffusion], teacher2.reshape(teacher2.shape[0], teacher2.shape[1], 1)).squeeze(2) / self.temp_teacher_diffusion

        sim_q2_diffusion = torch.bmm(teacher1[self.diffusion], pred2.reshape(pred2.shape[0], pred2.shape[1], 1)).squeeze(2) / self.temp_student_diffusion
        sim_k2_diffusion = torch.bmm(teacher1[self.diffusion], teacher1.reshape(teacher1.shape[0], teacher1.shape[1], 1)).squeeze(2) / self.temp_teacher_diffusion

        return [sim_q1, sim_q1_diffusion[self.not_zero]], [sim_k1, sim_k1_diffusion[self.not_zero]], [sim_q2, sim_q2_diffusion[self.not_zero]], [sim_k2, sim_k2_diffusion[self.not_zero]], student1