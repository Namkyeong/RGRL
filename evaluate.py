import numpy as np
import torch

from models import LogisticRegression
# To fix the random seed
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
import random
random.seed(0)


def evaluate_node(embeddings, dataset, name):

    labels = dataset.y
    emb_dim, num_class = embeddings.shape[1], dataset.y.unique().shape[0]

    dev_accs, test_accs = [], []

    for i in range(20):
        
        if name == "ogbn-arxiv":
            labels = dataset.y.reshape(-1)
            train_mask = dataset.train_mask[i]
            dev_mask = dataset.val_mask[i]
            test_mask = dataset.test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class)
            optimizer = torch.optim.AdamW(classifier.parameters(), lr=0.01)
        
        else :
            train_mask = dataset.train_mask[i]
            dev_mask = dataset.val_mask[i]
            if name == "wikics":
                test_mask = dataset.test_mask
            else:
                test_mask = dataset.test_mask[i]

            classifier = LogisticRegression(emb_dim, num_class)
            optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01)

        for _ in range(100):
            classifier.train()
            logits, loss = classifier(embeddings[train_mask], labels[train_mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        dev_logits, _ = classifier(embeddings[dev_mask], labels[dev_mask])
        test_logits, _ = classifier(embeddings[test_mask], labels[test_mask])
        dev_preds = torch.argmax(dev_logits, dim=1)
        test_preds = torch.argmax(test_logits, dim=1)

        dev_acc = (torch.sum(dev_preds == labels[dev_mask]).float() /
                       labels[dev_mask].shape[0]).detach().cpu().numpy()
        test_acc = (torch.sum(test_preds == labels[test_mask]).float() /
                        labels[test_mask].shape[0]).detach().cpu().numpy()

        dev_accs.append(dev_acc * 100)
        test_accs.append(test_acc * 100)

    dev_accs = np.stack(dev_accs)
    test_accs = np.stack(test_accs)

    dev_acc, dev_std = dev_accs.mean(), dev_accs.std()
    test_acc, test_std = test_accs.mean(), test_accs.std()
    
    print('Evaluate node classification results')
    print('** Val: {:.4f} ({:.4f}) | Test: {:.4f} ({:.4f}) **'.format(dev_acc, dev_std, test_acc, test_std))