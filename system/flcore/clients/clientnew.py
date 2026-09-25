from collections import defaultdict
import copy
import torch
import torch.nn as nn
import numpy as np
import time
import wandb
from flcore.clients.clientbase import Client

# Định danh kế thừa ClientBase
ClientBase = Client


class clientNew(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.global_prototypes = {}
        self.global_stds = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()
        self.lamda = getattr(args, 'lamda', 1.0)

    def extract_local_stats(self, dataloader=None):
        """
        Trích xuất đặc trưng không tính gradient và tính toán thống kê cục bộ theo từng class:
        - Mean: torch.mean(features, dim=0)
        - STD: torch.std(features, dim=0, unbiased=False)
        Xử lý biên:
        - 1 mẫu: std gán vector epsilon 1e-6
        - 0 mẫu: mean/std = vector 0, count = 0
        Đóng gói gửi lên Server: {class_c: {'mean': Tensor, 'std': Tensor, 'count': int}}
        """
        if dataloader is None:
            dataloader = self.load_train_data()

        self.model.eval()
        features_per_class = defaultdict(list)
        feat_dim = None

        with torch.no_grad():
            for i, (x, y) in enumerate(dataloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Trích xuất feature qua backbone
                if hasattr(self.model, 'base'):
                    rep = self.model.base(x)
                elif hasattr(self.model, 'features'):
                    rep = self.model.features(x)
                else:
                    rep = self.model(x)

                if feat_dim is None and rep.size(0) > 0:
                    feat_dim = rep.size(1)

                for feat, label in zip(rep, y):
                    features_per_class[label.item()].append(feat.unsqueeze(0))

        if feat_dim is None:
            feat_dim = 512

        num_classes = getattr(self, 'num_classes', 10)
        all_classes = set(range(num_classes)).union(features_per_class.keys())

        client_stats = {}
        for c in sorted(all_classes):
            if c in features_per_class and len(features_per_class[c]) > 0:
                class_feats = torch.cat(features_per_class[c], dim=0)
                num_samples = class_feats.size(0)

                # Prototype (Mean)
                mean_c = torch.mean(class_feats, dim=0)

                # Standard Deviation (STD)
                if num_samples > 1:
                    std_c = torch.std(class_feats, dim=0, unbiased=False)
                else:
                    # Biên 1 mẫu: std = vector 1e-6
                    std_c = torch.full_like(mean_c, 1e-6)

                client_stats[c] = {
                    'mean': mean_c.detach().cpu(),
                    'std': std_c.detach().cpu(),
                    'count': int(num_samples)
                }
            else:
                # Biên 0 mẫu: mean/std = vector 0, count = 0
                client_stats[c] = {
                    'mean': torch.zeros(feat_dim, device='cpu'),
                    'std': torch.zeros(feat_dim, device='cpu'),
                    'count': 0
                }

        self.local_stats = client_stats
        return client_stats

    def receive_global_stats(self, global_stats):
        """
        Nhận {mean, std} từ Server và chuyển tensor sang đúng device của client:
        - self.global_prototypes
        - self.global_stds
        """
        if global_stats is None:
            return

        for c, stats in global_stats.items():
            if 'mean' in stats and stats['mean'] is not None:
                self.global_prototypes[c] = stats['mean'].clone().detach().to(self.device)
            if 'std' in stats and stats['std'] is not None:
                self.global_stds[c] = stats['std'].clone().detach().to(self.device)

    def set_protos(self, global_stats):
        """Alias tương thích theo format PFLlib clientProto."""
        self.receive_global_stats(global_stats)

    def train(self):
        """
        Vòng lặp huấn luyện chuẩn:
        1. Huấn luyện cục bộ: tính loss phân loại chuẩn loss = self.loss(output, y) từ output = self.model(x).
        2. Tạm thời không can thiệp loss với prototype/std để đảm bảo môi trường sạch cho thử nghiệm.
        3. Sau khi huấn luyện, gọi self.extract_local_stats(trainloader) để cập nhật thống kê cục bộ gửi lên Server.
        """
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max_local_steps // 2)

        losses = []
        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                output = self.model(x)
                loss = self.loss(output, y)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses.append(loss.item())

        # Ghi log chỉ số client lên WandB nếu cờ log được kích hoạt
        if getattr(self.args, 'log', False) and len(losses) > 0:
            avg_loss = float(np.mean(losses))
            try:
                if wandb.run is not None:
                    wandb.log({
                        f"clients/client_{self.id}_train_loss": avg_loss,
                    }, commit=False)
            except Exception:
                pass

        # Tự động trích xuất và cập nhật local stats (mean, std, count) mới nhất sau khi train
        # để gửi trả về Server ở round kế tiếp
        self.extract_local_stats(trainloader)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


# Alias tên class

ClientNew = clientNew
