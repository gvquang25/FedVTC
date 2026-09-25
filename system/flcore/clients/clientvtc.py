import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import numpy as np

from flcore.clients.clientbase import Client
from flcore.trainmodel.tcnet import TC_net, total_variation_loss


class clientVTC(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.lamda = getattr(args, 'lam', 0.1)
        self.z_dim = getattr(args, 'z_dim', 512)
        self.tv_weight = getattr(args, 'tv_weight', 1e-3)
        self.tc_model = kwargs.get('tc_model', None)
        self.mse_loss = nn.MSELoss()

    def train_vtc(self, global_protos, global_sigmas, tc_params, server_round):
        trainloader = self.load_train_data()
        self.model.to(self.device)
        self.model.train()

        # Build tensor cache for prototypes & mean-normalized relative sigma weights
        # w = (1 / sig); w = w / w.mean() to prevent feedback loop and scale blow-up
        proto_bank = None
        w = None
        valid = None

        if global_protos is not None and server_round > 0:
            C = self.num_classes
            D = self.z_dim
            dev = self.device

            sig = torch.tensor([max(float(global_sigmas.get(c, 0.05)), 1e-2) for c in range(C)], device=dev)
            w = 1.0 / sig
            w = w / w.mean()  # relative weights, mean = 1: no scale blow-up

            proto_bank = torch.stack([
                global_protos[c].to(dev).flatten() if (c in global_protos and global_protos[c] is not None)
                else torch.zeros(D, device=dev) for c in range(C)
            ])
            valid = torch.tensor([c in global_protos and global_protos[c] is not None for c in range(C)], device=dev)

        # Warm up lambda smoothly over 10 rounds: 0 -> lamda
        current_lam = self.lamda * min(1.0, float(server_round) / 10.0)

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.local_epochs):
            for batch_idx, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                out = self.model(x)
                outputs, protos = out if isinstance(out, tuple) else (out, None)

                loss1 = self.loss(outputs, y)

                # Safer Prototype Alignment Loss
                if proto_bank is not None and protos is not None:
                    protos_flat = protos.flatten(1).float()
                    t, m = proto_bank[y], valid[y]
                    per = ((protos_flat - t.detach()) ** 2).mean(1) * w[y]
                    loss2 = per[m].mean() if m.any() else protos_flat.sum() * 0
                    loss = loss1 + current_lam * loss2

                    # Log once per round on client 0 for diagnostics
                    if self.id == 0 and epoch == 0 and batch_idx == 0:
                        min_sig = min(float(global_sigmas[c]) for c in range(self.num_classes) if c in global_sigmas)
                        feat_norm = protos_flat.norm(dim=1).mean().item()
                        print(f"  [Diag C0] r{server_round} loss1={loss1.item():.3f} loss2={loss2.item():.3f} "
                              f"feat_norm={feat_norm:.2f} sigma_min={min_sig:.2e} lam={current_lam:.3f}")
                else:
                    loss = loss1

                loss.backward()
                # Safety net against gradient explosion
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

        # Step 2: Compute Empirical Prototypes and Closed-form Sample Variance
        self.model.eval()
        local_proto_avg = {c: None for c in range(self.num_classes)}
        local_sigmas = {c: None for c in range(self.num_classes)}
        sample_counts = {c: 0 for c in range(self.num_classes)}

        features_by_class = {c: [] for c in range(self.num_classes)}

        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                _, protos = out if isinstance(out, tuple) else (out, None)

                if protos is not None:
                    protos_flat = protos.flatten(1)
                    for c in range(self.num_classes):
                        idx = (y == c)
                        if idx.any():
                            features_by_class[c].append(protos_flat[idx])

        for c in range(self.num_classes):
            if len(features_by_class[c]) > 0:
                all_feats = torch.cat(features_by_class[c], dim=0)  # Shape: (N_c, D)
                n_c = all_feats.shape[0]
                sample_counts[c] = n_c

                mean_c = torch.mean(all_feats, dim=0)  # Shape: (D,)
                local_proto_avg[c] = mean_c.cpu()

                # Closed-form empirical sample variance per dimension:
                # Floor at 1e-2 to prevent variance collapse feedback loop
                if n_c > 1:
                    var_c = torch.sum((all_feats - mean_c) ** 2).item() / (n_c * self.z_dim)
                else:
                    var_c = 0.05

                local_sigmas[c] = max(var_c, 1e-2)

        # Step 3: Optimize TC_net generator (1 epoch per round for efficiency)
        updated_tc_params = None
        if self.tc_model is not None and server_round > 0 and proto_bank is not None:
            updated_tc_params = self.optimize_tc(proto_bank, w, valid, tc_params, trainloader, epochs=1)

        return local_proto_avg, sample_counts, updated_tc_params, local_sigmas

    def optimize_tc(self, proto_bank, w, valid, tc_params, dataloader, epochs=1):
        """
        Optimizes TC_net generator to reconstruct realistic images from latent features.
        Guarantees safety with try/finally to restore model gradients.
        Uses Total Variation Loss to suppress adversarial noise artifacts.
        """
        self.tc_model.to(self.device)
        if tc_params is not None:
            self.tc_model.load_state_dict(tc_params)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.tc_model.train()

        tc_optimizer = torch.optim.Adam(self.tc_model.parameters(), lr=0.001, betas=(0.5, 0.999))

        try:
            for _ in range(epochs):
                for x, y in dataloader:
                    if isinstance(x, list):
                        x = x[0]
                    x, y = x.to(self.device), y.to(self.device)

                    with torch.no_grad():
                        out = self.model(x)
                        features = out[1] if isinstance(out, tuple) else out
                        features = features.flatten(1)

                    tc_optimizer.zero_grad()
                    x_hat = self.tc_model(features)
                    out_hat = self.model(x_hat)
                    z_hat = out_hat[1] if isinstance(out_hat, tuple) else out_hat
                    z_hat = z_hat.flatten(1)

                    t, m = proto_bank[y], valid[y]
                    if m.any():
                        per = ((z_hat.float() - t.detach()) ** 2).mean(1) * w[y]
                        feat_loss = per[m].mean()
                    else:
                        feat_loss = self.mse_loss(z_hat, features.detach())

                    # Total Variation Loss to suppress adversarial artifacts
                    tv_loss = total_variation_loss(x_hat)

                    total_loss = feat_loss + self.tv_weight * tv_loss
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.tc_model.parameters(), 10.0)
                    tc_optimizer.step()

            result_state = copy.deepcopy(self.tc_model.state_dict())
        finally:
            for param in self.model.parameters():
                param.requires_grad = True
            self.tc_model.to('cpu')

        return result_state

    def extra_fine_tune(self, synthetic_dataset, ft_epochs=1):
        """
        Fine-tunes the local client on synthetic distilled data + local real data.
        """
        if synthetic_dataset is None or len(synthetic_dataset) == 0:
            return

        syn_loader = DataLoader(synthetic_dataset, batch_size=self.batch_size, shuffle=True)
        real_loader = self.load_train_data()

        self.model.to(self.device)
        self.model.train()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

        for _ in range(ft_epochs):
            # 1. Distilled data training
            for x, y in syn_loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                outputs = out[0] if isinstance(out, tuple) else out
                loss = self.loss(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                optimizer.step()

            # 2. Local real data training (preserve local personalization)
            for x, y in real_loader:
                if isinstance(x, list):
                    x = x[0]
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                outputs = out[0] if isinstance(out, tuple) else out
                loss = self.loss(outputs, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
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