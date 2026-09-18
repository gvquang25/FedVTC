import os
import copy
import time
import torch
from torch.utils.data import TensorDataset
from collections import OrderedDict
import wandb

from flcore.servers.serverbase import Server
from flcore.clients.clientvtc import clientVTC
from flcore.trainmodel.tcnet import TC_net


class FedVTC(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # 1. Khởi tạo danh sách clients
        self.set_slow_clients()
        self.set_clients(clientVTC)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []

        self.z_dim = getattr(args, 'z_dim', 512)
        self.extra_rounds = getattr(args, 'extra_rounds', 50)
        self.num_fake_samples = getattr(args, 'num_fake_samples', 200)

        # Xác định kích thước ảnh theo bộ dữ liệu
        in_ch = 1 if ("mnist" in args.dataset.lower() or "emnist" in args.dataset.lower()) else 3
        img_sz = 28 if ("mnist" in args.dataset.lower() or "emnist" in args.dataset.lower()) else 32

        # Khởi tạo TC_net toàn cục
        self.tc_model = TC_net(in_features=self.z_dim, out_channels=in_ch, img_size=img_sz).to(self.device)

        # Khởi tạo Prototypes và Sigmas
        self.global_protos = {c: None for c in range(self.num_classes)}
        self.global_sigmas = {c: 1.0 for c in range(self.num_classes)}
        self.global_covs = {c: torch.eye(self.z_dim, device=self.device) for c in range(self.num_classes)}

        for client in self.clients:
            client.tc_model = copy.deepcopy(self.tc_model)

    def train(self):
        for i in range(self.global_rounds):
            self.current_round = i  # Đồng bộ round cho WandB
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            print(f"\n------------- Round {i}/{self.global_rounds - 1} Starting -------------")

            uploaded_protos = []
            uploaded_counts = []
            uploaded_tcs = []
            uploaded_sigmas = []

            # Huấn luyện cục bộ trên từng Client và bấm giờ chi tiết
            for client in self.selected_clients:
                c_start = time.time()

                lp, lc, l_tc, l_sig = client.train_vtc(
                    global_protos=self.global_protos,
                    global_sigmas=self.global_sigmas,
                    tc_params=self.tc_model.state_dict(),
                    server_round=i
                )

                c_time = time.time() - c_start
                print(f"  [Client {client.id:2d}] finished local training | Time: {c_time:.2f}s")

                uploaded_protos.append(lp)
                uploaded_counts.append(lc)
                if l_tc is not None:
                    uploaded_tcs.append(l_tc)
                if l_sig is not None:
                    uploaded_sigmas.append(l_sig)

            self.receive_models()
            self.aggregate_parameters()

            # Tổng hợp TC_net
            if len(uploaded_tcs) > 0:
                self.aggregate_tc(uploaded_tcs)

            # Tổng hợp Prototypes & Sigmas
            self.aggregate_protos_and_sigmas(uploaded_protos, uploaded_counts, uploaded_sigmas)

            round_cost = time.time() - s_t
            self.Budget.append(round_cost)
            print(f"------------- Round {i} Complete | Total Round Time: {round_cost:.2f}s -------------\n")

            if i % self.eval_gap == 0 or i == self.global_rounds - 1:
                print("Evaluate global model:")
                self.evaluate()

        # Extra Fine-Tuning Rounds với ảnh chưng cất
        if self.extra_rounds > 0:
            print("\n================ Generating Distilled Dataset ================")
            gen_start = time.time()
            synthetic_dataset = self.generate_distilled_data()
            print(f"Distilled Dataset generated | Time: {time.time() - gen_start:.2f}s")

            print(f"\n================ Starting {self.extra_rounds} Extra Rounds ================")
            for q in range(self.extra_rounds):
                self.current_round = self.global_rounds + q  # Đồng bộ round fine-tune cho WandB
                ft_round_start = time.time()
                for client in self.clients:
                    c_ft_start = time.time()
                    client.extra_fine_tune(synthetic_dataset, ft_epochs=1)
                    c_ft_time = time.time() - c_ft_start
                    print(f"  [Client {client.id:2d}] fine-tune finished | Time: {c_ft_time:.2f}s")

                print(f"Extra Fine-tune Round {q+1}/{self.extra_rounds} Complete | Time: {time.time() - ft_round_start:.2f}s")

                if (q + 1) % self.eval_gap == 0 or q == self.extra_rounds - 1:
                    print(f"\nEvaluate after Extra Round {q+1}:")
                    self.evaluate()

        self.print_(max(self.rs_test_acc), max(self.rs_test_auc), min(self.rs_train_loss))
        self.save_results()

        # Đóng phiên WandB để lượt chạy FedAvg kế tiếp tạo run riêng biệt
        if getattr(self.args, 'log', False):
            wandb.finish()

    def aggregate_tc(self, uploaded_tcs):
        avg_tc = OrderedDict()
        for key in uploaded_tcs[0].keys():
            avg_tc[key] = torch.zeros_like(uploaded_tcs[0][key], dtype=torch.float32)
            for state in uploaded_tcs:
                avg_tc[key] += state[key].float()
            avg_tc[key] = avg_tc[key] / len(uploaded_tcs)
            avg_tc[key] = avg_tc[key].to(dtype=uploaded_tcs[0][key].dtype)
        self.tc_model.load_state_dict(avg_tc)

    def aggregate_protos_and_sigmas(self, uploaded_protos, uploaded_counts, uploaded_sigmas):
        for c in range(self.num_classes):
            total_samples = 0
            weighted_proto = torch.zeros(self.z_dim, device=self.device)

            for p_dict, c_dict in zip(uploaded_protos, uploaded_counts):
                if p_dict[c] is not None:
                    count = c_dict[c]
                    weighted_proto += p_dict[c].to(self.device) * count
                    total_samples += count

            if total_samples > 0:
                self.global_protos[c] = weighted_proto / total_samples

            # Tổng hợp phương sai có trọng số bình phương mẫu
            denom = 0.0
            var_accum = 0.0
            for s_dict, c_dict in zip(uploaded_sigmas, uploaded_counts):
                if c in s_dict:
                    weight_sq = float(c_dict[c] ** 2)
                    var_accum += s_dict[c] * weight_sq
                    denom += weight_sq

            if denom > 0:
                self.global_sigmas[c] = var_accum / denom
                self.global_covs[c] = torch.eye(self.z_dim, device=self.device) * self.global_sigmas[c]

    def generate_distilled_data(self):
        self.tc_model.eval()
        fake_images = []
        fake_labels = []

        with torch.no_grad():
            for c in range(self.num_classes):
                if self.global_protos[c] is None:
                    continue
                mean = self.global_protos[c]
                cov = self.global_covs[c] + 1e-4 * torch.eye(self.z_dim, device=self.device)

                dist = torch.distributions.MultivariateNormal(mean, cov)
                z_samples = dist.sample((self.num_fake_samples,))

                syn_x = self.tc_model(z_samples)
                fake_images.append(syn_x.cpu())
                fake_labels.append(torch.full((self.num_fake_samples,), c, dtype=torch.long))

        if len(fake_images) > 0:
            all_x = torch.cat(fake_images, dim=0)
            all_y = torch.cat(fake_labels, dim=0)
            return TensorDataset(all_x, all_y)
        return None