"""AI-детекция оффтопа: провайдеры/шаблоны + LLM-классификатор.

Отдельный пакет от llm/ (тот — offline NotebookLM для highlights). Здесь —
онлайн-классификатор оффтоп-сегментов через настраиваемого AI-провайдера
(anthropic | openai-совместимый).

Два пути классификации:

  окна      — `classify_transcript`: транскрипт стрима нарезается на окна и
              каждое судится отдельно. Работает без исходника, но судит СМЕСЬ
              речи видео и стримера — отсюда его потолок точности.
  исходник  — `classify_with_source`: имея транскрипт исходного видео
              (`transcribe_source`), смесь расщепляется на дорожки
              (`align`/`turns`), и классификатор получает только собственную
              речь стримера + что играло в видео в этот момент (`pause`).
              Без исходника есть деградация `classify_single_pass`: один проход
              по ASR-сегментам, границы всё равно задаёт модель.

Секреты (API-ключи) сюда НЕ попадают из конфигов: они живут в Electron
safeStorage, приходят в backend по loopback и передаются в LLMClient на время
джобы. Ни один модуль пакета не пишет ключ на диск.

Тяжёлые зависимости (SDK провайдеров, GigaAM/torch) импортируются лениво внутри
функций: пакет должен оставаться импортируемым там, где их нет — например, при
валидации settings.json.
"""

from .align import (
    SHINGLE_N,
    Alignment,
    Word,
    align_transcripts,
    align_words,
    normalize_word,
    transcript_words,
)
from .classifier import (
    classify_single_pass,
    classify_transcript,
    classify_with_source,
)
from .client import Completion, LLMClient, LLMError
from .merger import merge_into_decisions
from .model import OfftopicSegment, OfftopicSet
from .pause import PAUSE_RATE_PER_S, VideoContext, turn_context, video_context
from .prompt import (
    ClassificationVerdict,
    RawSegment,
    SegmentsAnswer,
    SourceOutline,
    format_asr_lines,
    format_timecode,
    format_turn_lines,
    parse_classification_response,
    parse_llm_timecode,
    parse_outline_response,
    parse_segments_response,
    parse_topic_response,
    render_classification_prompt,
    render_single_pass_prompt,
    render_source_classification_prompt,
    render_source_outline_prompt,
    render_topic_prompt,
)
from .providers import (
    BUILTIN_PROVIDERS,
    AIProvider,
    AuthScheme,
    ProviderFormat,
    builtin_provider,
)
from .source import SourceError, SourceTranscript, transcribe_source
from .templates import (
    BUILTIN_TEMPLATES,
    CLASSIFICATION_PLACEHOLDERS,
    SINGLE_PASS_PLACEHOLDERS,
    SOURCE_CLASSIFICATION_PLACEHOLDERS,
    SOURCE_OUTLINE_PLACEHOLDERS,
    TOPIC_PLACEHOLDERS,
    PromptTemplate,
    TemplateKind,
    builtin_template,
    builtin_template_for_kind,
)
from .turns import (
    SnappedSpan,
    Turn,
    TrackSplit,
    smooth,
    snap_span,
    split_tracks,
    turns_from_segments,
)

__all__ = [
    # providers
    "AIProvider",
    "ProviderFormat",
    "AuthScheme",
    "BUILTIN_PROVIDERS",
    "builtin_provider",
    # templates
    "PromptTemplate",
    "TemplateKind",
    "BUILTIN_TEMPLATES",
    "TOPIC_PLACEHOLDERS",
    "CLASSIFICATION_PLACEHOLDERS",
    "SOURCE_OUTLINE_PLACEHOLDERS",
    "SOURCE_CLASSIFICATION_PLACEHOLDERS",
    "SINGLE_PASS_PLACEHOLDERS",
    "builtin_template",
    "builtin_template_for_kind",
    # model
    "OfftopicSegment",
    "OfftopicSet",
    # prompt
    "render_topic_prompt",
    "render_classification_prompt",
    "render_source_outline_prompt",
    "render_source_classification_prompt",
    "render_single_pass_prompt",
    "format_timecode",
    "format_asr_lines",
    "format_turn_lines",
    "parse_topic_response",
    "parse_classification_response",
    "parse_outline_response",
    "parse_segments_response",
    "parse_llm_timecode",
    "ClassificationVerdict",
    "SourceOutline",
    "RawSegment",
    "SegmentsAnswer",
    # client
    "LLMClient",
    "LLMError",
    "Completion",
    # classifier + merge
    "classify_transcript",
    "classify_with_source",
    "classify_single_pass",
    "merge_into_decisions",
    # исходник как ground truth
    "SourceTranscript",
    "SourceError",
    "transcribe_source",
    "Word",
    "Alignment",
    "SHINGLE_N",
    "normalize_word",
    "transcript_words",
    "align_words",
    "align_transcripts",
    "Turn",
    "TrackSplit",
    "smooth",
    "split_tracks",
    "snap_span",
    "SnappedSpan",
    "turns_from_segments",
    "VideoContext",
    "PAUSE_RATE_PER_S",
    "video_context",
    "turn_context",
]
