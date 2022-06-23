import argparse
from email.generator import Generator
import json
import os
import numpy as np

from sklearn.utils import shuffle

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import load_pretrain, get_param_num
from utils.tools import to_device, log, synth_one_sample, AttrDict
from model import AdaSpeechLoss
from dataset import Dataset

from evaluate import evaluate
from inference import get_reference_mel, synth_samples, synthesize
import sys
sys.path.append("vocoder")
from models.hifigan import Generator

import audio as Audio
from g2p_en import G2p
from text import text_to_sequence

import wandb

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_vocoder(config, checkpoint_path):
    config = json.load(open(config, 'r', encoding='utf-8'))
    config = AttrDict(config)
    checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")
    vocoder = Generator(config).to(device).eval()
    vocoder.load_state_dict(checkpoint_dict['generator'])
    vocoder.remove_weight_norm()

    return vocoder


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        args.train_file, preprocess_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 1  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size <= len(dataset)
    loader = DataLoader(
        dataset,
        shuffle = True,
        batch_size=batch_size * group_size,
        collate_fn=dataset.collate_fn,
    )

    # Prepare model
    model, optimizer = load_pretrain(args, configs, device, train=True)
    # model = nn.DataParallel(model)
    num_param, trainable_param = get_param_num(model)
    Loss = AdaSpeechLoss(preprocess_config, model_config).to(device)
    print(f"Number of AdaSpeech Parameters: {num_param}; trainable {trainable_param}")

    # Load vocoder
    #vocoder = get_vocoder(model_config, device)
    vocoder = get_vocoder(args.vocoder_config, args.vocoder_checkpoint)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # Training
    step = 1
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    phoneme_level_encoder_step = train_config["step"]["phoneme_level_encoder_step"]

    # synthesis utils
    g2p = G2p()
    def prepare_inputs(text, speaker_id, savefile, reference_mel):
        raw_texts = [text]
        ids = [savefile]
        raw_text = raw_texts[0]
        speakers = np.array([speaker_id])
        languages = np.array([0])

        text = ' '.join(g2p(raw_text)).replace(",", "sp")
        text = np.array(
                text_to_sequence(
                    text, preprocess_config["preprocessing"]["text"]["text_cleaners"]
            ))[None, ...]
        text_lens = np.array([len(text[0])])
        mel_spectrogram = np.array([reference_mel])
        batchs = [(ids, raw_texts, speakers, text, text_lens, max(text_lens), mel_spectrogram, languages)]
        
        return batchs

    SYNTH_TEXTS = {
        'phrase1': "This is a voicemod T T Speech trial of zero sample.",
        'phrase2': 'This is a test run to understand the voice quality of the audio.',
    }

    ds_name = preprocess_config['dataset']
    config = dict(
        dataset=ds_name,
        arch="adaspeech",
        task='fine-tune',
        ft_modules = ['speaker_emb', 'cln'],
        from_checkpoint = args.pretrain_dir.split('/')[-2],
        max_steps=total_step,
        eval_step=val_step,
        save_step=save_step,
        lr=None,
        bs=batch_size,
        grad_acc_step=grad_acc_step,
        total_params=num_param,
        trainable_params=trainable_param
    )
    config['num_speakers'] = 1

    wandb.init(project='voicemod', name=args.run_name, tags=['adaspeech', ds_name], config=config, sync_tensorboard=True)

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = 0
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device)
                # Forward
                if step >= phoneme_level_encoder_step:
                    phoneme_level_predictor = True
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch[2:]))
                else:
                    phoneme_level_predictor = False
                    exe_batch = batch + (phoneme_level_predictor, )
                    output = model(*(exe_batch[2:]))

                # Cal Loss
                if step >= phoneme_level_encoder_step:
                    losses = Loss(batch, output, phoneme_level_loss = True)
                else:
                    losses = Loss(batch, output, phoneme_level_loss = False)
                total_loss = losses[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if step % log_step == 0:
                    losses = [l.item() for l in losses]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Mel PostNet Loss: {:.4f}, Pitch Loss: {:.4f}, Energy Loss: {:.4f}, Duration Loss: {:.4f}, Phone_Level Loss: {:.4f}".format(
                        *losses
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses)

                if step % synth_step == 0:
                    model.eval()
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )
                    wandb.log({'train/sample':wandb.Audio(wav_prediction, sampling_rate)})

                    mel_len = batch[7][0].item()
                    ref_mel = batch[6][0, :mel_len].cpu().numpy().T
                    synth_inputs = []
                    for name, text in SYNTH_TEXTS.items():

                        synth_inputs += prepare_inputs(text, 0, f'{name}_{step}', ref_mel)
                    control = (1.,1.,1.,None)
                    synthesize(model, step, configs, vocoder, synth_inputs, control, train_config['path']['result_path'], )
                    wavs_to_log = {
                        k:wandb.Audio((os.path.join(train_config['path']['result_path'], f'{k}_{step}.wav')), caption=text) for k,text in SYNTH_TEXTS.items()
                    }
                    wandb.log(wavs_to_log)

                    model.train()

                if step % val_step == 0:
                    model.eval()
                    message = evaluate(model, step, configs, val_logger, vocoder, eval_file=args.eval_file)
                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", type=str, help="path to pretrained")
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--vocoder_checkpoint", type=str, default=None, required= True, help="path to vocoder checkpoint"
    )
    parser.add_argument(
        "--vocoder_config", type=str, default=None, required=True, help="path to vocoder config"
    )
    parser.add_argument(
        '--run_name', type=str, default=None, help="WandB run name"
    )
    parser.add_argument('--train_file', type=str, default='train.txt')
    parser.add_argument('--eval_file', type=str, default='val.txt')
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)


    main(args, configs)
