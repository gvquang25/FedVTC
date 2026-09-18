import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import numpy as np

from flcore.clients.clientbase import Client


class clientVTC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lamda = getattr(args, 'lam', 0.1)
        self.tc_model = kwargs.get('tc_model', None)
        self.mse_loss = nn.MSELoss()

    def train_vtc(self, global_protos, global_sigmas, tc_params, server_round):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        if self.tc_model is not None:
            self.tc_model.to(self.device)
            if tc_params is not None:
                self.tc_model.load_state_dict(tc_params)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        prototypes = {c: [] for c in range(self.num_classes)}
        sample_counts = {c: 0 for c in range(self.num_classes)}

        for epoch in range(self.local_epochs):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)
                outputs, protos = out if isinstance(out, tuple) else (out, None)

                loss1 = self.loss(outputs, y)

                # Prototype Alignment loss kết hợp độ phân tán sigma
                if global_protos is not None and server_round > 0 and protos is not None:
                    loss2 = torch.tensor(0.0, device=self.device)
                    matched_samples = 0
                    for p, l in zip(protos, y):
                        lbl = l.item()
                        if lbl in global_protos and global_protos[lbl] is not None:
                            target_c = global_protos[lbl].to(self.device)
                            sigma = global_sigmas.get(lbl, 1.0) if global_sigmas is not None else 1.0
                            sigma = max(float(sigma), 1e-4)
                            loss2 += self.mse_loss(p.float(), target_c.float()) / sigma
                            matched_samples += 1
                    
                    loss = loss1 + (loss2 / max(matched_samples, 1)) * self.lamda
                else:
                    loss = loss1

                loss.backward()
                optimizer.step()

                # Thu thập prototype tại epoch huấn luyện cuối cùng
                if epoch == self.local_epochs - 1 and protos is not None:
                    with torch.no_grad():
                        for p, l in zip(protos, y):
                            lbl = l.item()
                            prototypes[lbl].append(p.detach().cpu())
                            sample_counts[lbl] += 1

        # Tính prototype trung bình cục bộ cho từng class
        local_proto_avg = {}
        for c in range(self.num_classes):
            if len(prototypes[c]) > 0:
                local_proto_avg[c] = torch.mean(torch.stack(prototypes[c], dim=0), dim=0)
            else:
                local_proto_avg[c] = None

        # Tối ưu hóa TC_net và ước lượng phân tán sigma
        updated_tc_params = None
        updated_sigmas = {}
        if global_protos is not None and self.tc_model is not None and server_round > 0:
            updated_tc_params, updated_sigmas = self.optimize_tc(global_protos, global_sigmas, trainloader)

        return local_proto_avg, sample_counts, updated_tc_params, updated_sigmas

    def optimize_tc(self, global_protos, sigmas, dataloader, epochs=5):
        # Đóng băng gradient và chuyển sang eval để bảo vệ tham số ResNet & BatchNorm
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.tc_model.train()

        tc_optimizer = torch.optim.Adam(self.tc_model.parameters(), lr=0.001, betas=(0.5, 0.999))

        sd = {}
        for idx in range(self.num_classes):
            s_val = sigmas.get(idx, 1.0) if sigmas is not None else 1.0
            sd[idx] = torch.sqrt(torch.tensor(max(float(s_val), 1e-4), device=self.device, dtype=torch.float32)).requires_grad_()

        sd_optimizer = torch.optim.Adam(list(sd.values()), lr=1e-5, betas=(0.5, 0.999))

        for _ in range(epochs):
            for x, y in dataloader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)

                # Cắt đồ thị gradient (detach) để giải phóng VRAM GPU
                with torch.no_grad():
                    out = self.model(x)
                    features = out[1] if isinstance(out, tuple) else out

                # Reset gradient trước mỗi batch
                tc_optimizer.zero_grad()
                x_hats = self.tc_model(features)
                out_hat = self.model(x_hats)
                z_hats = out_hat[1] if isinstance(out_hat, tuple) else out_hat

                kl_mean_loss = torch.tensor(0.0, device=self.device)
                total = 0
                for zh, l in zip(z_hats, y):
                    idx = l.item()
                    if idx in global_protos and global_protos[idx] is not None:
                        c_proto = global_protos[idx].to(self.device)
                        s = sigmas.get(idx, 1.0) if sigmas is not None else 1.0
                        s = max(float(s), 1e-4)
                        kl_mean_loss += self.mse_loss(zh.float(), c_proto.float()) / s
                        total += 1

                if total > 0:
                    kl_mean_loss = kl_mean_loss / total
                    kl_mean_loss.backward()
                    tc_optimizer.step()

            for idx in range(self.num_classes):
                sd_optimizer.zero_grad()
                s = sigmas.get(idx, 1.0) if sigmas is not None else 1.0
                s = max(float(s), 1e-4)
                kl_var_loss = -torch.log(torch.square(sd[idx]) + 1e-8) + (torch.square(sd[idx]) / s)
                kl_var_loss.backward()
                sd_optimizer.step()

        # Mở khóa gradient cho mô hình chính
        for param in self.model.parameters():
            param.requires_grad = True

        updated_vars = {idx: torch.square(sd[idx].detach()).item() for idx in sd}
        return copy.deepcopy(self.tc_model.state_dict()), updated_vars

    def extra_fine_tune(self, synthetic_dataset, ft_epochs=1):
        if synthetic_dataset is None or len(synthetic_dataset) == 0:
            return

        syn_loader = DataLoader(synthetic_dataset, batch_size=self.batch_size, shuffle=True)
        real_loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for _ in range(ft_epochs):
            # 1. Huấn luyện trên ảnh chưng cất từ Server
            for x, y in syn_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                outputs = out[0] if isinstance(out, tuple) else out
                loss = self.loss(outputs, y)
                loss.backward()
                optimizer.step()

            # 2. Huấn luyện duy trì trên dữ liệu cục bộ
            for x, y in real_loader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                outputs = out[0] if isinstance(out, tuple) else out
                loss = self.loss(outputs, y)
                loss.backward()
                optimizer.step()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloaderfull:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except Exception:
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                output = self.model(x)
                if isinstance(output, tuple):
                    output = output[0]
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num