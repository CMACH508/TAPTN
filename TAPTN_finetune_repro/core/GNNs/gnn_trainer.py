import os
import torch
from time import time
import numpy as np


from core.GNNs.gnn_utils import EarlyStopping
from core.data_utils.load import load_data, load_gpt_preds
from core.utils import time_logger, lm_emb_path, infer_emb_dim
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


LOG_FREQ = 10


class GNNTrainer():

    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = cfg.device
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs

        # ! Load data
        data, num_classes = load_data(
            self.dataset_name, use_dgl=False, use_text=False, seed=self.seed)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()

        # ! Init gnn feature
        topk = 5
        if self.feature_type == 'ogb':
            print("Loading OGB features...")
            features = data.x
        elif self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = lm_emb_path(cfg, self.dataset_name, 'TA', self.seed)
            print(f"LM_emb_path: {LM_emb_path}")
            _dim = infer_emb_dim(LM_emb_path, self.num_nodes, cfg.lm.model.feat_shrink)
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, _dim)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations, sem) ...")
            LM_emb_path = lm_emb_path(cfg, self.dataset_name, 'E', self.seed)
            print(f"LM_emb_path: {LM_emb_path}")
            _dim = infer_emb_dim(LM_emb_path, self.num_nodes, cfg.lm.model.feat_shrink)
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, _dim)))
            ).to(torch.float32)
        elif self.feature_type == 'EN':
            print("Loading pretrained LM features (explanations, no_sem) ...")
            LM_emb_path = lm_emb_path(cfg, self.dataset_name, 'EN', self.seed)
            print(f"LM_emb_path: {LM_emb_path}")
            _dim = infer_emb_dim(LM_emb_path, self.num_nodes, cfg.lm.model.feat_shrink)
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, _dim)))
            ).to(torch.float32)
        elif self.feature_type == 'P':
            print("Loading top-k prediction features ...")
            features = load_gpt_preds(self.dataset_name, topk)
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        use_pred = self.feature_type == 'P'

        if self.gnn_model_name == "SAGE":
            from core.GNNs.SAGE.model import SAGE as GNN
        elif self.gnn_model_name == "GCN2":
            from core.GNNs.GCN2.model import GCN as GNN
        elif self.gnn_model_name == "DirGNN":
            from core.GNNs.DirGNN.model import GAT as GNN
        elif self.gnn_model_name == "GAT":
            from core.GNNs.GATv2.model import GAT as GNN
        elif self.gnn_model_name == "GATv2" or self.gnn_model_name == "Saint":
            from core.GNNs.GATv2.model import GATv2 as GNN
        elif self.gnn_model_name == "ASC":
            from core.GNNs.ASC.model import ASC as GNN
        elif self.gnn_model_name == "FSGNN":
            from core.GNNs.FSGNN.model import FSGNN as GNN
        elif self.gnn_model_name == "ACMGNN":
            from core.GNNs.ACMGNN.model import ACMGNN as GNN
        elif self.gnn_model_name == "DMP":
            from core.GNNs.DMP.model import DMP as GNN
        elif self.gnn_model_name == "DGI":
            from core.GNNs.DGI.model import DGI as GNN
        elif self.gnn_model_name == "APPNP":
            from core.GNNs.APPNP.model import APPNPModel as GNN
        elif self.gnn_model_name == "ChebNet":
            from core.GNNs.Cheb.model import ChebNet as GNN
        elif self.gnn_model_name == "GraphTARIF":
            from core.GNNs.GraphTARIF.model import GraphTARIF as GNN
        else:
            print(f"Model {self.gnn_model_name} is not supported! Loading GATv2 ...")
            from core.GNNs.GATv2.model import GATv2 as GNN

        self.model = GNN(in_channels=self.hidden_dim*topk if use_pred else self.features.shape[1],
                         hidden_channels=self.hidden_dim,
                         out_channels=self.num_classes,
                         num_layers=self.num_layers,
                         dropout=self.dropout,
                         use_pred=use_pred).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.0)

        trainable_params = sum(p.numel()
                               for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of parameters: {trainable_params}")
        self.ckpt = f"output/{self.dataset_name}/{self.gnn_model_name}.pt"
        self.stopper = EarlyStopping(
            patience=cfg.gnn.train.early_stop, path=self.ckpt) if cfg.gnn.train.early_stop > 0 else None
        if self.gnn_model_name == "DGI":
            self.loss_func = self.model.dgi.loss
        else:
            self.loss_func = torch.nn.CrossEntropyLoss()

        from core.GNNs.gnn_utils import Evaluator
        self._evaluator = Evaluator(name=self.dataset_name)
        self.evaluator = lambda pred, labels: self._evaluator.eval(
            {"y_pred": pred.argmax(dim=-1, keepdim=True),
             "y_true": labels.view(-1, 1)}
        )["acc"]

    def _forward(self, x, edge_index):
        logits = self.model(x, edge_index)  # small-graph
        return logits

    def _train(self):
        # ! Shared
        self.model.train()
        self.optimizer.zero_grad()
        # ! Specific
        logits = self._forward(self.features, self.data.edge_index)
        if self.gnn_model_name == "DGI":
            pos_z, neg_z, summary = logits
            loss = self.loss_func(pos_z, neg_z, summary)
            with torch.no_grad():
                embeddings = self.model.encoder(self.features, self.data.edge_index)
                clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1500)
                clf.fit(embeddings[self.data.train_mask].cpu().numpy(), self.data.y[self.data.train_mask].cpu().numpy())
                predictions = clf.predict(embeddings[self.data.train_mask].cpu().numpy())
                train_acc = accuracy_score(self.data.y[self.data.train_mask].cpu().numpy(), predictions)
        else:
            loss = self.loss_func(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
            train_acc = self.evaluator(
                logits[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()

        return loss.item(), train_acc

    @ torch.no_grad()
    def _evaluate(self):
        import warnings
        warnings.filterwarnings('ignore')

        self.model.eval()
        logits = self._forward(self.features, self.data.edge_index)
        if self.gnn_model_name == "DGI":
            pos_z, neg_z, summary = logits
            loss = self.loss_func(pos_z, neg_z, summary)
            with torch.no_grad():
                embeddings = self.model.encoder(self.features, self.data.edge_index)
                clf = LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1500)
                clf.fit(embeddings[self.data.train_mask].cpu().numpy(), self.data.y[self.data.train_mask].cpu().numpy())
                val_predictions = clf.predict(embeddings[self.data.val_mask].cpu().numpy())
                test_predictions = clf.predict(embeddings[self.data.test_mask].cpu().numpy())
                val_acc = accuracy_score(self.data.y[self.data.val_mask].cpu().numpy(), val_predictions)
                test_acc = accuracy_score(self.data.y[self.data.test_mask].cpu().numpy(), test_predictions)
            logits = torch.tensor(clf.predict(embeddings.cpu().numpy())).to(self.device).unsqueeze(-1)
        else:
            val_acc = self.evaluator(
                logits[self.data.val_mask], self.data.y[self.data.val_mask])
            test_acc = self.evaluator(
                logits[self.data.test_mask], self.data.y[self.data.test_mask])
        return val_acc, test_acc, logits

    @time_logger
    def train(self):
        # ! Training
        

        if self.gnn_model_name == "Saint":
            data=self.data.clone()
            data.x=self.features.to(self.device)
            from torch_geometric.loader import GraphSAINTRandomWalkSampler
            # 按 (dataset,seed) 唯一缓存目录：GraphSAINT 的 norm 缓存文件名不含图标识，
            # 若跨数据集共用同一目录会加载到错误的归一化缓存。
            import tempfile
            saint_dir = os.path.join(
                tempfile.gettempdir(), 'saint_cache',
                f'{self.dataset_name}_seed{self.seed}')
            os.makedirs(saint_dir, exist_ok=True)
            loader = GraphSAINTRandomWalkSampler(
                data.cpu(),
                batch_size=1000,
                walk_length=2,
                num_steps=5,
                sample_coverage=100,
                save_dir=saint_dir
            )
            
        for epoch in range(self.epochs):
            t0, es_str = time(), ''
            if self.gnn_model_name == "Saint":
                self.model.train()
                self.optimizer.zero_grad()
                for batch_data in loader:
                    batch_data = batch_data.to(self.device)
                    logits = self._forward(batch_data.x, batch_data.edge_index)
                    loss = self.loss_func(
                        logits[batch_data.train_mask], batch_data.y[batch_data.train_mask])
                    train_acc = self.evaluator(
                        logits[batch_data.train_mask], batch_data.y[batch_data.train_mask])
                    loss.backward()
                    self.optimizer.step()
            else:
                loss, train_acc = self._train()
            val_acc, test_acc, _ = self._evaluate()
            if self.stopper is not None:
                es_flag, es_str = self.stopper.step(val_acc, self.model, epoch)
                if es_flag:
                    print(
                        f'Early stopped, loading model from epoch-{self.stopper.best_epoch}')
                    break
            if epoch % LOG_FREQ == 0:
                print(
                    f'Epoch: {epoch}, Time: {time()-t0:.4f}, Loss: {loss:.4f}, TrainAcc: {train_acc:.4f}, ValAcc: {val_acc:.4f}, ES: {es_str}')

        # ! Finished training, load checkpoints
        if self.stopper is not None:
            self.model.load_state_dict(torch.load(self.stopper.path))

        return self.model

    @ torch.no_grad()
    def eval_and_save(self):
        torch.save(self.model.state_dict(), self.ckpt)
        val_acc, test_acc, logits = self._evaluate()
        print(
            f'[{self.gnn_model_name} + {self.feature_type}] ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc}
        return logits, res
