import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from audio_data import AudioData
from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist
import matplotlib.pyplot as plt
import librosa.display
from torchaudio.transforms import MuLawDecoding
import soundfile as sf


def train(epoch, loader, model, optimizer, scheduler, device, tag):
    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 1

    mse_sum = 0
    mse_n = 0

    for i, img in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)
        recon_loss = criterion(out, img)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * img.shape[0]
        part_mse_n = img.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 100 == 0:
                model.eval()

                sample = img[:sample_size]

                with torch.no_grad():
                    out, _ = model(sample)

                sample = MuLawDecoding(256)(sample.view(-1))
                out = MuLawDecoding(256)(out.view(-1))

                path = f"sample/{tag}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_sample.wav"
                sf.write(path, sample.cpu().numpy(), 16000)
                path = f"sample/{tag}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}_out.wav"
                sf.write(path, out.cpu().numpy(), 16000)

                plt.figure(figsize=(16, 6))
                p = plt.subplot(2, 1, 1)
                p.set_ylim(-0.5, 0.5)
                librosa.display.waveplot(sample.cpu().numpy(), sr=16000)
                p = plt.subplot(2, 1, 2)
                p.set_ylim(-0.5, 0.5)
                librosa.display.waveplot(out.cpu().numpy(), sr=16000)
                plt.tight_layout()
                path = f"sample/{tag}/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png"
                plt.savefig(path, format="png")
                plt.close()

                model.train()


def test(epoch, loader, model, optimizer, scheduler, device, tag):
    ckpt = torch.load(os.path.join('checkpoint', 'musicnet/vqvae_160.pt'))
    model.load_state_dict(ckpt)
    model.eval()
    if dist.is_primary():
        loader = loader.__iter__()
    anc, neg = loader.next()[:1].to(device), loader.next()[:1].to(device)
    with torch.no_grad():
        vq_t_a, vq_b_a, _, _, _ = model.encode(anc)
        vq_t_n, vq_b_n, _, _, _ = model.encode(neg)
        recons_a = model.decode(vq_t_a, vq_b_a)[:1].view(-1)
        recons_n = model.decode(vq_t_n, vq_b_n)[:1].view(-1)
        swap1 = model.decode(vq_t_a, vq_b_n)[:1].view(-1)
        swap2 = model.decode(vq_t_a, vq_b_a)[:1].view(-1)
        recons_a = MuLawDecoding(256)(recons_a)
        recons_n = MuLawDecoding(256)(recons_n)
        swap1 = MuLawDecoding(256)(swap1)
        swap2 = MuLawDecoding(256)(swap2)
        gt_a = MuLawDecoding(256)(anc.squeeze())
        gt_n = MuLawDecoding(256)(neg.squeeze())
    os.makedirs(f"gen/{tag}/", exist_ok=True)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_gt_a.wav"
    sf.write(path, gt_a.cpu().numpy(), 16000)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_gt_n.wav"
    sf.write(path, gt_n.cpu().numpy(), 16000)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_recons_a.wav"
    sf.write(path, recons_a.cpu().numpy(), 16000)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_recons_n.wav"
    sf.write(path, recons_n.cpu().numpy(), 16000)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_swap1.wav"
    sf.write(path, swap1.cpu().numpy(), 16000)
    path = f"gen/{tag}/{str(epoch + 1).zfill(5)}_swap2.wav"
    sf.write(path, swap2.cpu().numpy(), 16000)


def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = AudioData('/groups/1/gcc50521/furukawa/musicnet_npy_10sec')
    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)
    loader = DataLoader(
        dataset, batch_size=128, sampler=sampler, num_workers=2
    )
    os.makedirs(f"sample/{args.tag}/", exist_ok=True)
    os.makedirs(f"checkpoint/{args.tag}/", exist_ok=True)

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, args.tag)

        if dist.is_primary() and (i + 1) % 20 == 0:
            torch.save(model.module.state_dict(), f"checkpoint/{args.tag}/vqvae_{str(i + 1).zfill(3)}.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
            2 ** 15
            + 2 ** 14
            + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--path", type=str)
    parser.add_argument("tag", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))
