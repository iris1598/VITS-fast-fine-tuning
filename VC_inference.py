import os
import numpy as np
import torch
from torch import no_grad, LongTensor
import argparse
import commons
from audio_io import resample as _resample
from mel_processing import spectrogram_torch
import utils
from models import SynthesizerTrn
import gradio as gr

from text import text_to_sequence, _clean_text
device = "cuda:0" if torch.cuda.is_available() else "cpu"
import logging
logging.getLogger("PIL").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("markdown_it").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("asyncio").setLevel(logging.WARNING)

language_marks = {
    "Japanese": "",
    "日本語": "[JA]",
    "简体中文": "[ZH]",
    "English": "[EN]",
    "Mix": "",
}
lang = ['日本語', '简体中文', 'English', 'Mix']


def get_text(text, hps, is_symbol):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def _in_colab():
    return "google.colab" in os.environ.get("JPY_PARENT_PID", "") or os.path.isdir("/content")


def record_audio_to_array(audio):
    """Normalise whatever gradio hands us into float32 samples in [-1, 1].

    Gradio used to return int16 numpy arrays; since gradio 4 `type="numpy"`
    yields float32 already scaled to [-1, 1].  ``np.iinfo`` blows up on float
    dtypes, so detect the dtype instead of assuming int16.
    """
    if isinstance(audio, tuple):  # (sample_rate, samples)
        audio = audio[1]
    audio = np.asarray(audio)
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        audio = audio.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        audio = audio.astype(np.float32)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 1.0:  # some backends hand back un-normalised ints as floats
            audio = audio / peak
    if audio.ndim == 2:
        # gradio returns (samples, channels)
        audio = audio.mean(axis=1) if audio.shape[1] <= 8 else audio.mean(axis=0)
    return np.ascontiguousarray(audio, dtype=np.float32)


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, language, speed):
        if language is not None:
            text = language_marks[language] + text + language_marks[language]
        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, False)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn


def create_vc_fn(model, hps, speaker_ids):
    def vc_fn(original_speaker, target_speaker, record_audio, upload_audio):
        input_audio = record_audio if record_audio is not None else upload_audio
        if input_audio is None:
            return "You need to record or upload an audio", None
        sampling_rate, audio = input_audio
        original_speaker_id = speaker_ids[original_speaker]
        target_speaker_id = speaker_ids[target_speaker]

        audio = record_audio_to_array(audio)
        if sampling_rate != hps.data.sampling_rate:
            audio = _resample(torch.from_numpy(audio).unsqueeze(0),
                              sampling_rate, hps.data.sampling_rate)[0].numpy()
        with no_grad():
            y = torch.FloatTensor(np.ascontiguousarray(audio))
            y = y / max(-y.min(), y.max()) / 0.99
            y = y.to(device)
            y = y.unsqueeze(0)
            spec = spectrogram_torch(y, hps.data.filter_length,
                                     hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                                     center=False).to(device)
            spec_lengths = LongTensor([spec.size(-1)]).to(device)
            sid_src = LongTensor([original_speaker_id]).to(device)
            sid_tgt = LongTensor([target_speaker_id]).to(device)
            audio = model.voice_conversion(spec, spec_lengths, sid_src=sid_src, sid_tgt=sid_tgt)[0][
                0, 0].data.cpu().float().numpy()
        del y, spec, spec_lengths, sid_src, sid_tgt
        return "Success", (hps.data.sampling_rate, audio)

    return vc_fn


def build_app(model_dir, config_dir):
    hps = utils.get_hparams_from_file(config_dir)

    net_g = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    _ = net_g.eval()

    _ = utils.load_checkpoint(model_dir, net_g, None)
    speaker_ids = hps.speakers
    speakers = list(hps.speakers.keys())
    tts_fn = create_tts_fn(net_g, hps, speaker_ids)
    vc_fn = create_vc_fn(net_g, hps, speaker_ids)

    app = gr.Blocks(title="VITS fast fine-tuning")
    with app:
        with gr.Tab("Text-to-Speech"):
            with gr.Row():
                with gr.Column():
                    textbox = gr.Textbox(label="Text",
                                         placeholder="Type your sentence here",
                                         value="こんにちわ。", lines=3, elem_id="tts-input")
                    # select character
                    char_dropdown = gr.Dropdown(choices=speakers, value=speakers[0], label='character')
                    language_dropdown = gr.Dropdown(choices=lang, value=lang[0], label='language')
                    duration_slider = gr.Slider(minimum=0.1, maximum=5, value=1, step=0.1,
                                                label='速度 Speed')
                with gr.Column():
                    text_output = gr.Textbox(label="Message")
                    audio_output = gr.Audio(label="Output Audio", elem_id="tts-audio")
                    btn = gr.Button("Generate!")
                    btn.click(tts_fn,
                              inputs=[textbox, char_dropdown, language_dropdown, duration_slider,],
                              outputs=[text_output, audio_output])
        with gr.Tab("Voice Conversion"):
            gr.Markdown("""
                            录制或上传声音，并选择要转换的音色。
            """)
            with gr.Column():
                # gradio 4 renamed `source=` to `sources=[...]`
                record_audio = gr.Audio(label="record your voice", sources=["microphone"], type="numpy")
                upload_audio = gr.Audio(label="or upload audio here", sources=["upload"], type="numpy")
                source_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="source speaker")
                target_speaker = gr.Dropdown(choices=speakers, value=speakers[0], label="target speaker")
            with gr.Column():
                message_box = gr.Textbox(label="Message")
                converted_audio = gr.Audio(label='converted audio')
            btn = gr.Button("Convert!")
            btn.click(vc_fn, inputs=[source_speaker, target_speaker, record_audio, upload_audio],
                      outputs=[message_box, converted_audio])
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./G_latest.pth", help="directory to your fine-tuned model")
    parser.add_argument("--config_dir", default="./finetune_speaker.json", help="directory to your model config file")
    parser.add_argument("--share", default=False, type=utils.str2bool, nargs="?", const=True,
                        help="make link public (used in colab)")
    parser.add_argument("--server_name", default="127.0.0.1", help="bind address")

    args = parser.parse_args()

    app = build_app(args.model_dir, args.config_dir)

    if not _in_colab():
        try:
            import webbrowser
            webbrowser.open(f"http://{args.server_name}:7860")
        except Exception:  # noqa: BLE001 - headless machines have no browser
            pass

    app.launch(share=args.share, server_name=args.server_name)
