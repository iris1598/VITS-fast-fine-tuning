import os
import json
import argparse
import itertools
import math
import socket
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

import librosa
import logging

logging.getLogger('numba').setLevel(logging.WARNING)

import commons
import utils
from compat import autocast, make_grad_scaler, silence_known_warnings
import monotonic_align
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch


torch.backends.cudnn.benchmark = True
global_step = 0

# AMP is only meaningful on CUDA.  Keeping this in one place avoids handing
# fp16 tensors to a CPU-only build.
USE_AMP = False


def _find_free_port():
  """Pick a free localhost port so parallel runs don't collide on 8000."""
  with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind(("127.0.0.1", 0))
    return s.getsockname()[1]


def _device(rank):
  """The device a given process should use."""
  if torch.cuda.is_available():
    return torch.device("cuda", rank)
  return torch.device("cpu")


def main():
  """Assume Single Node Multi GPUs Training Only"""
  global USE_AMP
  silence_known_warnings()

  hps = utils.get_hparams()

  n_gpus = torch.cuda.device_count()
  if n_gpus == 0:
    print(
      "=" * 72 + "\n"
      "WARNING: no CUDA device detected.\n"
      "Falling back to single-process CPU training. This is only practical for\n"
      "smoke tests / very small datasets — it is orders of magnitude slower.\n"
      + "=" * 72
    )
  elif n_gpus > 1:
    print(f"Found {n_gpus} CUDA devices, launching distributed training.")
  else:
    print("Found 1 CUDA device, launching single-GPU training.")

  USE_AMP = bool(hps.train.fp16_run) and torch.cuda.is_available()
  if hps.train.fp16_run and not torch.cuda.is_available():
    print("NOTE: train.fp16_run is enabled in the config but no GPU is present; "
          "running in fp32.")

  nprocs = max(1, n_gpus)
  os.environ["MASTER_ADDR"] = "localhost"
  os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

  mp.spawn(run, nprocs=nprocs, args=(nprocs, hps,))


def run(rank, n_gpus, hps):
  global global_step
  symbols = hps['symbols']
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    logger.info("monotonic_align backend: %s", monotonic_align.backend_name())

  device = _device(rank)

  # Use gloo on Windows (no NCCL) and on CPU-only machines.
  if torch.cuda.is_available() and os.name != 'nt':
    backend = 'nccl'
  else:
    backend = 'gloo'
  dist.init_process_group(backend=backend, init_method='env://',
                          world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  if torch.cuda.is_available():
    torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, symbols)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate()
  # num_workers>0 needs the dataset to be picklable and can be fragile on
  # Windows; keep the original behaviour on Linux, be conservative elsewhere.
  num_workers = 2 if os.name != 'nt' else 0
  train_loader = DataLoader(train_dataset, num_workers=num_workers, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, symbols)
    eval_loader = DataLoader(eval_dataset, num_workers=0, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)
  else:
    eval_loader = None

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      n_speakers=hps.data.n_speakers,
      **hps.model).to(device)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).to(device)

  # load existing model
  if hps.cont:
      try:
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_latest.pth"), net_g, None)
          _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_latest.pth"), net_d, None)
          global_step = (epoch_str - 1) * len(train_loader)
      except Exception as e:
          print(f"Failed to find latest checkpoint ({e}), loading G_0.pth...")
          if hps.train_with_pretrained_model:
              print("Train with pretrained model...")
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
              _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
          else:
              print("Train without pretrained model...")
          epoch_str = 1
          global_step = 0
  else:
      if hps.train_with_pretrained_model:
          print("Train with pretrained model...")
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/G_0.pth", net_g, None)
          _, _, _, epoch_str = utils.load_checkpoint("./pretrained_models/D_0.pth", net_d, None)
      else:
          print("Train without pretrained model...")
      epoch_str = 1
      global_step = 0
  # freeze all other layers except speaker embedding
  for p in net_g.parameters():
      p.requires_grad = True
  for p in net_d.parameters():
      p.requires_grad = True
  optim_g = torch.optim.AdamW(
      net_g.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate,
      betas=hps.train.betas,
      eps=hps.train.eps)

  if torch.cuda.is_available():
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])
  else:
    net_g = DDP(net_g)
    net_d = DDP(net_d)

  scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
  scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)

  scaler = make_grad_scaler(enabled=USE_AMP, device_type=device.type)

  # The shipped configs use a very large `train.epochs` (10000) as an upper
  # bound; `--max_epochs` is what the user actually controls.
  total_epochs = min(int(hps.train.epochs), int(hps.max_epochs))
  print(f"Training from epoch {epoch_str} to {total_epochs}.")

  for epoch in range(epoch_str, total_epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], device)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, device)
    scheduler_g.step()
    scheduler_d.step()

  if rank == 0:
    logger.info("Training finished.")
  dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, device):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths, speakers) in enumerate(tqdm(train_loader)):
    x, x_lengths = x.to(device, non_blocking=True), x_lengths.to(device, non_blocking=True)
    spec, spec_lengths = spec.to(device, non_blocking=True), spec_lengths.to(device, non_blocking=True)
    y, y_lengths = y.to(device, non_blocking=True), y_lengths.to(device, non_blocking=True)
    speakers = speakers.to(device, non_blocking=True)

    with autocast(enabled=USE_AMP, device_type=device.type):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths, speakers)

      mel = spec_to_mel_torch(
          spec,
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.mel_fmin,
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1),
          hps.data.filter_length,
          hps.data.n_mel_channels,
          hps.data.sampling_rate,
          hps.data.hop_length,
          hps.data.win_length,
          hps.data.mel_fmin,
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False, device_type=device.type):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=USE_AMP, device_type=device.type):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False, device_type=device.type):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])

        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = {
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()),
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step,
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval, device)

        utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "G_latest.pth"))

        utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                              os.path.join(hps.model_dir, "D_latest.pth"))
        # save to google drive
        if os.path.exists("/content/drive/MyDrive/"):
            utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "G_latest.pth"))

            utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join("/content/drive/MyDrive/", "D_latest.pth"))
        if hps.preserved > 0:
          utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
          utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                  os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
          old_g = utils.oldest_checkpoint_path(hps.model_dir, "G_[0-9]*.pth",
                                               preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
          old_d = utils.oldest_checkpoint_path(hps.model_dir, "D_[0-9]*.pth", preserved=hps.preserved)
          if os.path.exists(old_g):
            print(f"remove {old_g}")
            os.remove(old_g)
          if os.path.exists(old_d):
            print(f"remove {old_d}")
            os.remove(old_d)
          if os.path.exists("/content/drive/MyDrive/"):
              utils.save_checkpoint(net_g, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "G_{}.pth".format(global_step)))
              utils.save_checkpoint(net_d, None, hps.train.learning_rate, epoch,
                                    os.path.join("/content/drive/MyDrive/", "D_{}.pth".format(global_step)))
              old_g = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "G_[0-9]*.pth",
                                                   preserved=hps.preserved)  # Preserve 4 (default) historical checkpoints.
              old_d = utils.oldest_checkpoint_path("/content/drive/MyDrive/", "D_[0-9]*.pth", preserved=hps.preserved)
              if os.path.exists(old_g):
                  print(f"remove {old_g}")
                  os.remove(old_g)
              if os.path.exists(old_d):
                  print(f"remove {old_d}")
                  os.remove(old_d)
    global_step += 1

  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


def evaluate(hps, generator, eval_loader, writer_eval, device):
    generator.eval()
    with torch.no_grad():
      batch = None
      for batch_idx, batch in enumerate(eval_loader):
        break
      if batch is None:
        print("Validation set is empty, skipping evaluation.")
        generator.train()
        return
      x, x_lengths, spec, spec_lengths, y, y_lengths, speakers = batch
      x, x_lengths = x.to(device), x_lengths.to(device)
      spec, spec_lengths = spec.to(device), spec_lengths.to(device)
      y, y_lengths = y.to(device), y_lengths.to(device)
      speakers = speakers.to(device)

      # remove else
      x = x[:1]
      x_lengths = x_lengths[:1]
      spec = spec[:1]
      spec_lengths = spec_lengths[:1]
      y = y[:1]
      y_lengths = y_lengths[:1]
      speakers = speakers[:1]

      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, speakers, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec,
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.mel_fmin,
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step,
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()


if __name__ == "__main__":
  main()
