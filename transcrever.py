#!/usr/bin/env python3
"""
Transcrição + timestamps (+ diarização opcional) com WhisperX (CPU).

Aceita um arquivo OU uma pasta (processa todos os áudios da pasta em lote,
reaproveitando os modelos já carregados).

Uso via Docker (compatível com versões anteriores):
    docker compose run --rm transcritor /audio/meu-arquivo.mp3
    docker compose run --rm transcritor /audio                 # pasta inteira
    docker compose run --rm transcritor --no-diarize /audio/x.mp3

Uso local (após `pip install .` ou `pip install -r requirements.txt`):
    transcritor caminho/do/audio.mp3 --output ./output
    python transcrever.py caminho/da/pasta --no-diarize

Gera, para cada áudio (conforme --formats), nomeando a partir do stem do input:
    transcricao_<nome>.txt   — leitura humana, com falantes quando houver
    legenda_<nome>.srt       — legendas p/ Aegisub, Subtitle Edit, DaVinci, CapCut
    legenda_<nome>.vtt       — legendas p/ HTML5/web
    transcricao_<nome>.json  — segmentos + timestamps por palavra (uso programático)
"""
import argparse
import json
import os
import sys
from pathlib import Path

DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # quantização — muito mais rápido em CPU

# extensões que o ffmpeg lê e que consideramos "áudio" ao varrer uma pasta
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi",
}
VALID_FORMATS = ("txt", "srt", "vtt", "json")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="transcritor",
        description="Transcrição + timestamps (+ diarização opcional) com WhisperX (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  transcritor audio.mp3\n"
            "  transcritor ./audio --no-diarize\n"
            "  transcritor audio.mp3 --model medium --formats srt,json\n"
        ),
    )
    p.add_argument("input", help="arquivo de áudio/vídeo OU pasta com vários áudios")
    p.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"),
                   help="tamanho do Whisper: tiny|base|small|medium|large-v3 (padrão: small)")
    p.add_argument("--language", default=os.environ.get("LANGUAGE", "pt"),
                   help="idioma do áudio, código ISO 639-1 (padrão: pt)")
    p.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "."),
                   help="diretório de saída (padrão: diretório atual)")
    p.add_argument("--formats", default=os.environ.get("FORMATS", "txt,srt,vtt,json"),
                   help="formatos separados por vírgula: txt,srt,vtt,json (padrão: todos)")
    p.add_argument("--no-diarize", action="store_true",
                   help="não identificar falantes (dispensa o token do HuggingFace)")
    p.add_argument("--batch-size", type=int, default=int(os.environ.get("BATCH_SIZE", "8")),
                   help="batch size da transcrição; reduza se faltar RAM (padrão: 8)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="token do HuggingFace p/ diarização (ou variável de ambiente HF_TOKEN)")
    return p.parse_args(argv)


# --- formatação de timestamps ------------------------------------------------
def ts_srt(sec: float) -> str:
    h = int(sec // 3600); m = int((sec % 3600) // 60); s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}".replace(".", ",")


def ts_short(sec: float) -> str:
    m = int(sec // 60); s = int(sec % 60)
    return f"{m:02d}:{s:02d}"


def _json_default(o):
    """Torna escalares/arrays numpy serializáveis sem importar numpy."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


# --- escrita dos formatos ----------------------------------------------------
def write_txt(path, segs, diarized):
    with open(path, "w", encoding="utf-8") as f:
        if diarized:
            current = None
            for seg in segs:
                spk = seg.get("speaker", "SPEAKER_?")
                if spk != current:
                    f.write(f"\n[{spk}] ({ts_short(seg['start'])})\n")
                    current = spk
                f.write(seg["text"].strip() + " ")
        else:
            for seg in segs:
                f.write(f"[{ts_short(seg['start'])}] {seg['text'].strip()}\n")


def write_srt(path, segs, diarized):
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segs, 1):
            prefix = f"[{seg.get('speaker', '?')}] " if diarized else ""
            f.write(f"{i}\n{ts_srt(seg['start'])} --> {ts_srt(seg['end'])}\n"
                    f"{prefix}{seg['text'].strip()}\n\n")


def write_vtt(path, segs, diarized):
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segs:
            prefix = f"[{seg.get('speaker', '?')}] " if diarized else ""
            a = ts_srt(seg["start"]).replace(",", ".")
            b = ts_srt(seg["end"]).replace(",", ".")
            f.write(f"{a} --> {b}\n{prefix}{seg['text'].strip()}\n\n")


def write_json(path, segs):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(segs, f, ensure_ascii=False, indent=2, default=_json_default)


def write_outputs(stem, segs, output_dir, formats, diarized):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "txt" in formats:
        p = out / f"transcricao_{stem}.txt"; write_txt(p, segs, diarized); written.append(p)
    if "srt" in formats:
        p = out / f"legenda_{stem}.srt"; write_srt(p, segs, diarized); written.append(p)
    if "vtt" in formats:
        p = out / f"legenda_{stem}.vtt"; write_vtt(p, segs, diarized); written.append(p)
    if "json" in formats:
        p = out / f"transcricao_{stem}.json"; write_json(p, segs); written.append(p)
    return written


# --- pipeline ----------------------------------------------------------------
def build_pipeline(model_size, language, diarize, hf_token):
    """Carrega modelo, alinhador e (opcionalmente) diarizador — uma única vez."""
    import whisperx  # import tardio: mantém --help e validações rápidos sem a lib

    print(f"Carregando modelo Whisper ({model_size}, {language})...")
    model = whisperx.load_model(model_size, DEVICE, compute_type=COMPUTE_TYPE, language=language)
    align_model, align_meta = whisperx.load_align_model(language_code=language, device=DEVICE)
    diarizer = None
    if diarize:
        from whisperx.diarize import DiarizationPipeline
        diarizer = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)
    return model, align_model, align_meta, diarizer


def transcribe_one(audio_path, model, align_model, align_meta, diarizer,
                   language, batch_size, output_dir, formats):
    import whisperx

    stem = Path(audio_path).stem

    print("  -> transcrevendo...")
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size, language=language)

    print("  -> alinhando timestamps por palavra...")
    result = whisperx.align(result["segments"], align_model, align_meta, audio, DEVICE,
                            return_char_alignments=False)

    if diarizer is not None:
        print("  -> identificando falantes...")
        diarize_segments = diarizer(audio)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    return write_outputs(stem, result["segments"], output_dir, formats,
                         diarized=diarizer is not None)


def main(argv=None):
    args = parse_args(argv)

    formats = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    invalid = [x for x in formats if x not in VALID_FORMATS]
    if invalid:
        sys.exit(f"Formato(s) inválido(s): {', '.join(invalid)}. "
                 f"Use: {', '.join(VALID_FORMATS)}")
    if not formats:
        sys.exit("Nenhum formato selecionado em --formats.")

    in_path = Path(args.input)
    if in_path.is_dir():
        files = sorted(p for p in in_path.iterdir() if p.suffix.lower() in AUDIO_EXTS)
        if not files:
            sys.exit(f"Nenhum áudio encontrado em: {in_path} "
                     f"(extensões: {', '.join(sorted(AUDIO_EXTS))})")
    elif in_path.is_file():
        files = [in_path]
    else:
        sys.exit(f"Arquivo ou pasta não encontrado: {in_path}")

    diarize = not args.no_diarize
    if diarize and not args.hf_token:
        print("Aviso: HF_TOKEN não definido — seguindo SEM diarização de falantes.\n"
              "       Defina HF_TOKEN (veja o README) ou use --no-diarize para "
              "silenciar este aviso.")
        diarize = False

    model, align_model, align_meta, diarizer = build_pipeline(
        args.model, args.language, diarize, args.hf_token)

    total = len(files)
    ok = 0
    for idx, audio_path in enumerate(files, 1):
        header = f"[{idx}/{total}] {audio_path.name}" if total > 1 else audio_path.name
        print(f"\n=== {header} ===")
        try:
            written = transcribe_one(audio_path, model, align_model, align_meta, diarizer,
                                     args.language, args.batch_size, args.output, formats)
            print("OK: " + ", ".join(str(p) for p in written))
            ok += 1
        except Exception as e:  # não deixa um arquivo ruim derrubar o lote
            print(f"ERRO em {audio_path.name}: {e}", file=sys.stderr)

    if total > 1:
        print(f"\nConcluído: {ok}/{total} arquivo(s) transcrito(s).")
    if ok == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
