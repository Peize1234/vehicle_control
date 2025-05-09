# 这部分用于训练用于路径状态感知的 transformer 模型

import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import numpy as np
from ErrorEncoder import ErrorEncoder
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from Model_Trace_Interactor import ModelTraceInteractor
from utils import state_diff
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


delta_prefix = "sim_delta"
Fxf_prefix = "sim_Fxf"
init_state_prefix = "sim_init_state"
output_prefix = "sim_output"
reference_prefix = "sim_reference"
u_prefix = "sim_u"
u_aug_prefix = "sim_u_aug"
suffix = ".npy"


class SequenceDataSet(Dataset):
    def __init__(self, data_path, data_file_num=100, shuffle=True, train=True):
        super().__init__()
        data_path = Path(data_path)
        self.data_file_num = data_file_num
        self.shuffle = shuffle

        self.delta_path = [data_path / f"{delta_prefix}{i}{suffix}" for i in range(data_file_num)]
        self.Fxf_path = [data_path / f"{Fxf_prefix}{i}{suffix}" for i in range(data_file_num)]
        self.init_state_path = [data_path / f"{init_state_prefix}{i}{suffix}" for i in range(data_file_num)]
        self.output_path = [data_path / f"{output_prefix}{i}{suffix}" for i in range(data_file_num)]
        self.reference_path = [data_path / f"{reference_prefix}{i}{suffix}" for i in range(data_file_num)]
        self.u_path = [data_path / f"{u_prefix}{i}{suffix}" for i in range(data_file_num)]
        # self.u_aug_path = [data_path / f"{u_aug_prefix}{i}{suffix}" for i in range(data_file_num)]

        if train:
            self.delta_path = self.delta_path[:int(0.8 * data_file_num)]
            self.Fxf_path = self.Fxf_path[:int(0.8 * data_file_num)]
            self.init_state_path = self.init_state_path[:int(0.8 * data_file_num)]
            self.output_path = self.output_path[:int(0.8 * data_file_num)]
            self.reference_path = self.reference_path[:int(0.8 * data_file_num)]
            self.u_path = self.u_path[:int(0.8 * data_file_num)]
            # self.u_aug_path = self.u_aug_path[:int(0.8 * data_file_num)]
        else:
            self.delta_path = self.delta_path[int(0.8 * data_file_num):]
            self.Fxf_path = self.Fxf_path[int(0.8 * data_file_num):]
            self.init_state_path = self.init_state_path[int(0.8 * data_file_num):]
            self.output_path = self.output_path[int(0.8 * data_file_num):]
            self.reference_path = self.reference_path[int(0.8 * data_file_num):]
            self.u_path = self.u_path[int(0.8 * data_file_num):]
            # self.u_aug_path = self.u_aug_path[int(0.8 * data_file_num):]

    def __len__(self):
        return len(self.delta_path)

    def __getitem__(self, idx):
        delta = torch.from_numpy(np.load(self.delta_path[idx]))
        Fxf = torch.from_numpy(np.load(self.Fxf_path[idx]))
        init_state = np.load(self.init_state_path[idx])
        sim_output = np.load(self.output_path[idx])
        reference = np.load(self.reference_path[idx])
        u = torch.from_numpy(np.load(self.u_path[idx]))
        # u_aug = torch.from_numpy(np.load(self.u_aug_path[idx]))

        seq_len = reference.shape[1]

        sim_output = np.concatenate((init_state[:, None, :], sim_output), axis=1, dtype=np.float32)
        reference = np.concatenate((init_state[:, None, :], reference), axis=1, dtype=np.float32)
        out_reference_diff = torch.from_numpy(state_diff(sim_output, reference)) * 1e2  # * 1e2 is for scaling(not optimal but must)

        # out_reference_diff = (out_reference_diff - torch.min(out_reference_diff, dim=-1, keepdim=True)[0]) / \
        #                                             (torch.max(out_reference_diff, dim=-1, keepdim=True)[0] -
        #                                              torch.min(out_reference_diff, dim=-1, keepdim=True)[0])
        # print(out_reference_diff)
        # exit()
        input = torch.cat((out_reference_diff, delta[:, :, None] / torch.deg2rad(torch.tensor(40)),
                           Fxf[:, :, None] / 2000), dim=2)
        # print(input)
        # exit()
        output = u[:, 0]
        if self.shuffle:
            random_idx = torch.randperm(input.size()[0])
            input = input[random_idx]
            output = output[random_idx]

        return input, output


if __name__ == '__main__':

    train_suffix = "_test"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_dataset = SequenceDataSet(data_path=Path(r"D:\PythonCode\RL_exp\sim_model_data"),
                                    data_file_num=100, shuffle=True, train=True)
    test_dataset = SequenceDataSet(data_path=Path(r"D:\PythonCode\RL_exp\sim_model_data"),
                                   data_file_num=100, shuffle=True, train=False)

    train_dataloader = DataLoader(train_dataset, batch_size=3, shuffle=True, num_workers=4)
    test_dataloader = DataLoader(test_dataset, batch_size=3, shuffle=True, num_workers=4)

    width = 1
    model = ErrorEncoder(input_dim=train_dataset[0][0].shape[-1], out_dim=10, out_encoder_dim=1, seq_len=10,
                         num_heads=8, width=width, drop_prob=0.1).to(device)
    # model.load_state_dict(torch.load(r"D:\PythonCode\RL_exp\models\sim_model1.pth", map_location=device))
    # model.load_encoder_state_dict(f"encoder_state_dict_width{width}.pth")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.MSELoss()

    train_loss_list = []
    min_train_loss = torch.tensor(float('inf'))

    test_loss_list = []
    min_test_loss = torch.tensor(float('inf'))

    for epoch in range(10000):
        train_epoch_losses = []
        test_epoch_losses = []
        model.train()
        for i, (input, output) in enumerate(train_dataloader):
            batch_size, mini_batch, seq_len, state_dim = input.size()
            input = input.view(batch_size * mini_batch, seq_len, state_dim).to(device).float().contiguous()
            output = output.view(batch_size * mini_batch, 1).to(device).float().contiguous()

            encoder_out, decoder_out = model(input)
            loss = loss_func(encoder_out, output)
            train_loss_list.append(loss.cpu().detach().item())
            train_epoch_losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(encoder_out[:10, 0])
                print(output[:10, 0])
                print(f"Epoch: {epoch}, i: {i}, Current Train Loss: {loss.item()}, Min Train Loss: {min_train_loss}")
                print("=" * 100)

        if epoch % 1 == 0:
            model.eval()
            for i, (input, output) in enumerate(test_dataloader):
                batch_size, mini_batch, seq_len, state_dim = input.size()
                input = input.view(batch_size * mini_batch, seq_len, state_dim).to(device).float().contiguous()
                output = output.view(batch_size * mini_batch, 1).to(device).float().contiguous()

                encoder_out, decoder_out = model(input)

                loss = loss_func(encoder_out, output)
                # print(encoder_out)
                # print(output)
                # print(loss)
                # exit()

                test_loss_list.append(loss.cpu().detach().item())
                test_epoch_losses.append(loss.item())

                if i % 10 == 0:
                    print(encoder_out[:10, 0])
                    print(output[:10, 0])
                    print(f"Epoch: {epoch}, i: {i}, Current Test Loss: {loss.item()}, Min Test Loss: {min_test_loss}")
                    print("=" * 100)

        if torch.mean(torch.tensor(test_epoch_losses)) < min_test_loss:
            min_test_loss = torch.mean(torch.tensor(test_epoch_losses)).cpu()
            # torch.save(model.state_dict(), r"D:\PythonCode\RL_exp\models\sim_model1.pth")
            model.save_encoder_state_dict(f"encoder_state_dict_width{width}{train_suffix}.pth")
            # torch.save(model.state_dict(), f"./models/encoder_state_dict_width{width}_epoch_{epoch}.pth")
            print(f"Epoch: {epoch}, Save Model！")
            print("=" * 100)

        if torch.mean(torch.tensor(train_epoch_losses)) < min_train_loss:
            min_train_loss = torch.mean(torch.tensor(train_epoch_losses)).cpu()

        d = f"Epoch: {epoch}, Train Loss: {torch.mean(torch.tensor(train_epoch_losses)).cpu()}, " \
            f"Test Loss: {torch.mean(torch.tensor(test_epoch_losses)).cpu()}"
        print(d)
        print(d, file=open(f"./results/train_stage1_log_width{width}{train_suffix}.txt", "a"))
        print("=" * 100)

        plt.plot(train_loss_list, label="Train Loss")
        plt.plot(test_loss_list, label="Test Loss")
        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(f"./results/train_stage1_loss_plot_width{width}{train_suffix}.png")
        plt.clf()

        np.save(f"./results/train_stage1_loss_list_width{width}{train_suffix}.npy", np.array(train_loss_list))
        np.save(f"./results/test_stage1_loss_list_width{width}{train_suffix}.npy", np.array(test_loss_list))