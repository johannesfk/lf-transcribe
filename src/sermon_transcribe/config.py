from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator


ComputeType = Literal["float16", "int8_float16", "int8"]


class AudioConfig(BaseModel):
    input_paths: list[str]


class VADConfig(BaseModel):
    enabled: bool = True
    segment_len_s: float = 28.0
    overlap_s: float = 1.5


class ASRConfig(BaseModel):
    model_id: str = "openai/whisper-large-v3"
    backend: Literal["ctranslate2", "transformers", "granite"] = "ctranslate2"
    compute_type: ComputeType = "float16"
    batch_size: int = 8
    language: str = "en"
    task: Literal["transcribe", "translate"] = "transcribe"
    beam_size: int = 1
    auto_tune_batch: bool = True
    vram_target_gb: float = 0.8

    @field_validator("batch_size")
    @classmethod
    def _pos_batch(cls, v: int) -> int:
        assert v >= 1, "batch_size must be >= 1"
        return v


class AlignmentConfig(BaseModel):
    enabled: bool = True


class DiarizationConfig(BaseModel):
    enabled: bool = False
    hf_token_env: Optional[str] = None


class ProseFormatConfig(BaseModel):
    enabled: bool = True
    sentence_gap_s: float = 0.5
    paragraph_gap_s: float = 2.5
    itn_mode: Literal["off", "safe"] = "safe"
    max_sentences_per_paragraph: int = 6
    discourse_starters: list[str] = Field(default_factory=lambda: ["Now", "So", "Let’s", "In closing"])


class ProsePolishConfig(BaseModel):
    enabled: bool = False
    engine: Literal["local_llm", "none"] = "none"
    max_diff_ratio: float = 0.07


class ProseConfig(BaseModel):
    format: ProseFormatConfig = ProseFormatConfig()
    polish: ProsePolishConfig = ProsePolishConfig()


class DictionaryConfig(BaseModel):
    path: Optional[str] = None


class ExportConfig(BaseModel):
    formats: list[Literal["md", "docx", "json"]] = Field(default_factory=lambda: ["md", "json"])
    out_dir: str = "out"


class EvalConfig(BaseModel):
    reference_path: Optional[str] = None


class PerfConfig(BaseModel):
    vram_target_gb: float = 0.8


class AppConfig(BaseModel):
    audio: AudioConfig
    vad: VADConfig = VADConfig()
    asr: ASRConfig = ASRConfig()
    alignment: AlignmentConfig = AlignmentConfig()
    diarization: DiarizationConfig = DiarizationConfig()
    prose: ProseConfig = ProseConfig()
    dictionary: DictionaryConfig = DictionaryConfig()
    export: ExportConfig = ExportConfig()
    eval: EvalConfig = EvalConfig()
    perf: PerfConfig = PerfConfig()

    @staticmethod
    def load(path: str | Path) -> "AppConfig":
        p = Path(path)
        assert p.exists(), f"Config file not found: {p}"
        data = yaml.safe_load(p.read_text())
        try:
            return AppConfig(**data)
        except ValidationError as e:
            raise SystemExit(f"Invalid config: {e}")

    def dumps_json(self) -> str:
        return json.dumps(self.model_dump(), indent=2)
