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
    transcricao_<nome>.md    — transcrição formatada em Markdown
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

DEVICE = "cpu"
COMPUTE_TYPE = "int8"  # quantização — muito mais rápido em CPU

# extensões que o ffmpeg lê e que consideramos "áudio" ao varrer uma pasta
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi",
}
VALID_FORMATS = ("txt", "srt", "vtt", "json", "md")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        prog="transcritor",
        description="Transcrição + timestamps (+ diarização opcional) com WhisperX (CPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exemplos:\n"
            "  transcritor audio.mp3\n"
            "  transcritor ./audio --no-diarize --skip-existing\n"
            "  transcritor audio.mp3 --speakers 'Alice,Bob' --max-speakers 2\n"
            "  transcritor audio.mp3 --language auto --formats srt,md\n"
        ),
    )
    p.add_argument("input", help="arquivo de áudio/vídeo OU pasta com vários áudios")
    p.add_argument("--model", default=os.environ.get("WHISPER_MODEL", "small"),
                   help="tamanho do Whisper: tiny|base|small|medium|large-v3 (padrão: small)")
    p.add_argument("--language", default=os.environ.get("LANGUAGE", "pt"),
                   help="idioma (ISO 639-1) ou 'auto' p/ detectar automaticamente (padrão: pt)")
    p.add_argument("--output", default=os.environ.get("OUTPUT_DIR", "."),
                   help="diretório de saída (padrão: diretório atual)")
    p.add_argument("--formats", default=os.environ.get("FORMATS", "txt,srt,vtt,json"),
                   help="formatos separados por vírgula: txt,srt,vtt,json,md (padrão: "
                        "txt,srt,vtt,json)")
    p.add_argument("--no-diarize", action="store_true",
                   help="não identificar falantes (dispensa o token do HuggingFace)")
    p.add_argument("--speakers", default=None,
                   help="nomes p/ renomear SPEAKER_00,01,... na ordem "
                        "(ex.: --speakers 'Alice,Bob')")
    p.add_argument("--min-speakers", type=int, default=None,
                   help="nº mínimo de falantes — melhora a diarização quando você sabe a conta")
    p.add_argument("--max-speakers", type=int, default=None,
                   help="nº máximo de falantes")
    p.add_argument("--skip-existing", action="store_true",
                   help="em modo pasta, pula áudios cujas saídas já existem no --output")
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


def fmt_dur(sec: float) -> str:
    if sec < 60:
        return f"{sec:.1f}s"
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def _json_default(o):
    """Torna escalares/arrays numpy serializáveis sem importar numpy."""
    if hasattr(o, "item"):
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def apply_speaker_names(segs, names):
    """Renomeia SPEAKER_00, SPEAKER_01, ... para os nomes dados, na ordem."""
    mapping = {f"SPEAKER_{i:02d}": n for i, n in enumerate(names)}
    for seg in segs:
        if seg.get("speaker") in mapping:
            seg["speaker"] = mapping[seg["speaker"]]
        for w in seg.get("words", []):
            if w.get("speaker") in mapping:
                w["speaker"] = mapping[w["speaker"]]
    return segs


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


def write_md(path, stem, segs, diarized):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# Transcrição — {stem}\n\n")
        if diarized:
            current = None
            for seg in segs:
                spk = seg.get("speaker", "SPEAKER_?")
                if spk != current:
                    f.write(f"\n**{spk}** — `{ts_short(seg['start'])}`\n\n")
                    current = spk
                f.write(seg["text"].strip() + " ")
            f.write("\n")
        else:
            for seg in segs:
                f.write(f"`[{ts_short(seg['start'])}]` {seg['text'].strip()}\n\n")


def output_paths(stem, output_dir, formats):
    out = Path(output_dir)
    names = {
        "txt": f"transcricao_{stem}.txt",
        "srt": f"legenda_{stem}.srt",
        "vtt": f"legenda_{stem}.vtt",
        "json": f"transcricao_{stem}.json",
        "md": f"transcricao_{stem}.md",
    }
    return {fmt: out / names[fmt] for fmt in formats}


def write_outputs(stem, segs, output_dir, formats, diarized):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = output_paths(stem, output_dir, formats)
    writers = {
        "txt": lambda p: write_txt(p, segs, diarized),
        "srt": lambda p: write_srt(p, segs, diarized),
        "vtt": lambda p: write_vtt(p, segs, diarized),
        "json": lambda p: write_json(p, segs),
        "md": lambda p: write_md(p, stem, segs, diarized),
    }
    written = []
    for fmt in formats:
        writers[fmt](paths[fmt])
        written.append(paths[fmt])
    return written


# --- pipeline ----------------------------------------------------------------
class Transcriber:
    """Carrega modelos uma vez e transcreve N arquivos reaproveitando-os."""

    def __init__(self, model_size, language, diarize, hf_token,
                 min_speakers=None, max_speakers=None):
        import whisperx  # import tardio: mantém --help e validações rápidos sem a lib

        self.wx = whisperx
        self.language = language
        self.auto = (language == "auto")
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers

        load_lang = None if self.auto else language
        print(f"Carregando modelo Whisper ({model_size}, {language})...")
        self.model = whisperx.load_model(model_size, DEVICE,
                                         compute_type=COMPUTE_TYPE, language=load_lang)

        self._align_cache = {}
        if not self.auto:
            self._align_cache[language] = whisperx.load_align_model(
                language_code=language, device=DEVICE)

        self.diarizer = None
        if diarize:
            from whisperx.diarize import DiarizationPipeline
            self.diarizer = DiarizationPipeline(use_auth_token=hf_token, device=DEVICE)

    def _align_model(self, lang):
        if lang not in self._align_cache:
            print(f"  -> carregando alinhador para '{lang}'...")
            self._align_cache[lang] = self.wx.load_align_model(
                language_code=lang, device=DEVICE)
        return self._align_cache[lang]

    def transcribe(self, audio_path, batch_size):
        wx = self.wx
        audio = wx.load_audio(str(audio_path))

        t = time.perf_counter()
        print("  -> transcrevendo...")
        lang_arg = None if self.auto else self.language
        result = self.model.transcribe(audio, batch_size=batch_size, language=lang_arg)
        lang = result.get("language", self.language)
        print(f"     ok ({fmt_dur(time.perf_counter() - t)}, idioma: {lang})")

        t = time.perf_counter()
        print("  -> alinhando timestamps por palavra...")
        align_model, align_meta = self._align_model(lang)
        result = wx.align(result["segments"], align_model, align_meta, audio, DEVICE,
                          return_char_alignments=False)
        print(f"     ok ({fmt_dur(time.perf_counter() - t)})")

        if self.diarizer is not None:
            t = time.perf_counter()
            print("  -> identificando falantes...")
            kw = {}
            if self.min_speakers is not None:
                kw["min_speakers"] = self.min_speakers
            if self.max_speakers is not None:
                kw["max_speakers"] = self.max_speakers
            diarize_segments = self.diarizer(audio, **kw)
            result = wx.assign_word_speakers(diarize_segments, result)
            print(f"     ok ({fmt_dur(time.perf_counter() - t)})")

        return result["segments"], self.diarizer is not None


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

    names = [n.strip() for n in args.speakers.split(",") if n.strip()] if args.speakers else None

    # pré-filtra o que já existe: não carrega os modelos se não há nada a fazer
    total = len(files)
    skipped = 0
    if args.skip_existing:
        kept = []
        for audio_path in files:
            paths = output_paths(audio_path.stem, args.output, formats)
            if all(p.exists() for p in paths.values()):
                skipped += 1
            else:
                kept.append(audio_path)
        files = kept
        if not files:
            print(f"Nada a fazer: {skipped}/{total} arquivo(s) já existiam (--skip-existing).")
            return

    diarize = not args.no_diarize
    if diarize and not args.hf_token:
        print("Aviso: HF_TOKEN não definido — seguindo SEM diarização de falantes.\n"
              "       Defina HF_TOKEN (veja o README) ou use --no-diarize para "
              "silenciar este aviso.")
        diarize = False
    if not diarize and (names or args.min_speakers or args.max_speakers):
        print("Aviso: --speakers/--min-speakers/--max-speakers são ignorados sem diarização.")

    tr = Transcriber(args.model, args.language, diarize, args.hf_token,
                     args.min_speakers, args.max_speakers)

    ok = 0
    n = len(files)
    for idx, audio_path in enumerate(files, 1):
        stem = audio_path.stem
        header = f"[{idx}/{n}] {audio_path.name}" if n > 1 else audio_path.name
        print(f"\n=== {header} ===")
        try:
            segs, diarized = tr.transcribe(audio_path, args.batch_size)
            if names and diarized:
                apply_speaker_names(segs, names)
            written = write_outputs(stem, segs, args.output, formats, diarized)
            print("OK: " + ", ".join(str(p) for p in written))
            ok += 1
        except Exception as e:  # não deixa um arquivo ruim derrubar o lote
            print(f"ERRO em {audio_path.name}: {e}", file=sys.stderr)

    if total > 1:
        errors = total - ok - skipped
        print(f"\nConcluído: {ok} transcrito(s), {skipped} pulado(s), {errors} com erro "
              f"(de {total}).")
    if ok == 0 and skipped == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
