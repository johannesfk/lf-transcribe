from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, BarColumn, TaskProgressColumn

from .logging_setup import setup_logging, get_console
from .config import AppConfig
from .audio import load_mono_16k, peak_normalize
from .asr import transcribe_with_whisperx
from .align import run_alignment
from .prose import paragraphs_from_segments, apply_polish_guarded
from .exporters import export_docx, export_json, export_markdown
from .metrics import compute_metrics

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.callback()
def _main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logs")):
    setup_logging(logging.DEBUG if verbose else logging.INFO)


@app.command()
def transcribe(
    audio_path: Path = typer.Argument(..., exists=True, help="Path to input audio file"),
    config: Path = typer.Option(..., "--config", "-c", exists=True, help="Path to YAML config"),
):
    """Transcribe a long-form sermon and export results."""
    cfg = AppConfig.load(config)
    console = get_console()

    # Use console for step-by-step progress instead of live progress bar
    console.print("[bold blue]Starting transcription pipeline...[/bold blue]\n")

    # Load audio
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Loading and normalizing audio...", total=None)
        audio, sr = load_mono_16k(audio_path)
        audio = peak_normalize(audio)
    console.print("[green]✓[/green] Audio loaded\n")

    # ASR - Let native tqdm show download progress, then show our progress for transcription
    sample_rate = 16000
    total_duration = len(audio) / sample_rate
    
    # Check if model is cached
    from pathlib import Path
    model_id = cfg.asr.model_id
    if cfg.asr.backend.lower() in {"ctranslate2", "faster_whisper", "faster-whisper"}:
        tail = model_id.split("/")[-1]
        if tail.startswith("whisper-"):
            tail = tail[len("whisper-"):]
        model_id = tail
    
    # Construct HF cache path
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub" / f"models--Systran--faster-whisper-{model_id}"
    model_cached = hf_cache.exists()
    
    if not model_cached:
        console.print(f"[cyan]Downloading Whisper model ({model_id})...[/cyan]")
        console.print("[dim]Download progress will appear below[/dim]\n")
    
    # Use a variable to store the progress context so we can create it lazily
    progress_obj = None
    task_id = None
    
    def update_progress(current_time: float, total: float):
        nonlocal progress_obj, task_id
        if progress_obj is None:
            # First callback - model is loaded, transcription has started
            # Now create the progress bar
            progress_obj = Progress(
                SpinnerColumn(),
                TextColumn("{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console
            )
            progress_obj.__enter__()
            task_id = progress_obj.add_task(
                f"[cyan]Transcribing {total_duration/60:.1f} min audio...",
                total=total_duration
            )
        progress_obj.update(task_id, completed=min(current_time, total))
    
    try:
        asr = transcribe_with_whisperx(audio, cfg, progress_callback=update_progress)
    finally:
        # Clean up progress bar if it was created
        if progress_obj is not None:
            progress_obj.__exit__(None, None, None)
    
    console.print(f"[green]✓[/green] Transcribed {len(asr.segments)} segments\n")

    # Alignment
    if cfg.alignment.enabled:
        # Check if alignment model is cached
        alignment_model_path = Path.home() / ".cache" / "torch" / "hub" / "checkpoints" / "wav2vec2_fairseq_base_ls960_asr_ls960.pth"
        alignment_cached = alignment_model_path.exists()
        
        if not alignment_cached:
            console.print("[cyan]⋯ Downloading alignment model...[/cyan]")
            console.print("[dim]This may take a few minutes on first run[/dim]")
        else:
            console.print("[cyan]⋯ Aligning word timestamps...[/cyan]")
        
        asr = run_alignment(audio, asr)
        console.print("[green]✓[/green] Alignment complete\n")

    # Paragraphs + optional polish
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Formatting paragraphs...", total=None)
        paras = paragraphs_from_segments(asr.segments, paragraph_gap_s=cfg.prose.format.paragraph_gap_s,
                                       max_sentences=cfg.prose.format.max_sentences_per_paragraph,
                                       discourse_starters=cfg.prose.format.discourse_starters)
        if cfg.prose.polish.enabled:
            for i, p in enumerate(paras):
                paras[i].text = apply_polish_guarded(p.text, cfg.prose.polish.max_diff_ratio)
    console.print(f"[green]✓[/green] Formatted {len(paras)} paragraphs\n")

    # Exports
    with Progress(SpinnerColumn(), TextColumn("{task.description}"), TimeElapsedColumn(), console=console) as progress:
        task = progress.add_task("[cyan]Exporting results...", total=None)
        out_dir = Path(cfg.export.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = out_dir / audio_path.stem

        if "json" in cfg.export.formats:
            export_json(str(base.with_suffix(".json")), asr)
        if "md" in cfg.export.formats:
            export_markdown(str(base.with_suffix(".md")), paras)
        if "docx" in cfg.export.formats:
            export_docx(str(base.with_suffix(".docx")), paras, title=audio_path.stem)
    console.print(f"[green]✓[/green] Exported to {base}\n")

    # Metrics (optional WER)
    ref_text = None
    if cfg.eval.reference_path:
        ref_text = Path(cfg.eval.reference_path).read_text()
    _ = compute_metrics(asr, ref_text)

    console.print(f"\n[bold green]✓ Transcription complete![/bold green]")
    typer.echo(json.dumps({
        "out_base": str(base),
        "segments": len(asr.segments),
        "language": asr.language,
    }, indent=2))


@app.command()
def compare(
    audio_path: Path = typer.Argument(..., exists=True),
    config_a: Path = typer.Option(..., "--config-a", exists=True),
    config_b: Path = typer.Option(..., "--config-b", exists=True),
):
    """Run two configs and print a simple side-by-side metrics report."""
    from time import perf_counter

    def run_one(cfg_path: Path):
        cfg = AppConfig.load(cfg_path)
        audio, sr = load_mono_16k(audio_path)
        audio = peak_normalize(audio)
        t0 = perf_counter()
        asr = transcribe_with_whisperx(audio, cfg)
        if cfg.alignment.enabled:
            asr = run_alignment(audio, asr)
        ref_text = None
        if cfg.eval.reference_path:
            ref_text = Path(cfg.eval.reference_path).read_text()
        m = compute_metrics(asr, ref_text)
        return m

    m1 = run_one(config_a)
    m2 = run_one(config_b)

    typer.echo("A: wall={:.1f}s peak_vram={:.2f}GB wer={}".format(m1.wall_time_s, m1.peak_vram_gb, f"{m1.wer:.2f}%" if m1.wer is not None else "-"))
    typer.echo("B: wall={:.1f}s peak_vram={:.2f}GB wer={}".format(m2.wall_time_s, m2.peak_vram_gb, f"{m2.wer:.2f}%" if m2.wer is not None else "-"))


if __name__ == "__main__":
    app()
