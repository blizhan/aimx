"""Inline terminal image rendering for ``aimx query images``.

This module is the sole owner of:
- Terminal capability detection (passive, env-var-based, no DA queries).
- ``ImageRenderPlan`` construction (pure data, no I/O).
- Inline rendering via ``rich.Console`` + ``textual_image.renderable.Image``.
- A one-shot stderr warning for dependency / capability problems.

It MUST NOT be imported or called from the ``--json`` or ``--plain`` code
paths; those must remain byte-identical to the pre-003 behaviour.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Literal

from aimx.rendering import colors

if TYPE_CHECKING:
    from PIL import Image as PILImage

# ---------------------------------------------------------------------------
# One-shot stderr warning
# ---------------------------------------------------------------------------

_WARNED: bool = False


def warn_once(message: str) -> None:
    """Print a single warning to stderr; idempotent for the process lifetime."""
    global _WARNED  # noqa: PLW0603
    if not _WARNED:
        print(f"aimx: inline image rendering unavailable: {message}", file=sys.stderr)
        _WARNED = True


# ---------------------------------------------------------------------------
# TerminalCapability
# ---------------------------------------------------------------------------

ProtocolTag = Literal["auto", "fallback_text", "disabled"]


@dataclass(frozen=True)
class TerminalCapability:
    """Snapshot of the current stdout terminal properties."""

    is_tty: bool
    columns: int
    rows: int
    protocol: ProtocolTag
    reason: str | None = None


def _classify_protocol(is_tty: bool, columns: int) -> tuple[ProtocolTag, str | None]:
    """Return (protocol, reason) based on passive env-var inspection."""
    if not is_tty:
        return "disabled", "stdout is not a TTY"
    if columns < 20:
        return "disabled", f"terminal width {columns} < 20 columns"

    # Try importing textual_image; if unavailable, fall back to text.
    try:
        import textual_image.renderable  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        warn_once(str(exc))
        return "disabled", str(exc)

    # Passive protocol-support classification.
    term_program = os.environ.get("TERM_PROGRAM", "")
    if term_program in ("iTerm.app", "WezTerm"):
        return "auto", None
    if "KITTY_WINDOW_ID" in os.environ:
        return "auto", None
    if "GHOSTTY_RESOURCES_DIR" in os.environ:
        return "auto", None
    if "WEZTERM_EXECUTABLE" in os.environ:
        return "auto", None
    # VTE-based terminals (GNOME Terminal, etc.) support some image protocols.
    if "VTE_VERSION" in os.environ:
        return "auto", None

    # Plain ANSI / tmux / xterm — textual_image will use half-block fallback.
    return "fallback_text", None


def detect_capability() -> TerminalCapability:
    """Probe the current terminal once and return a frozen capability object."""
    is_tty = sys.stdout.isatty()
    size = shutil.get_terminal_size(fallback=(120, 24))
    columns, rows = size.columns, size.lines
    protocol, reason = _classify_protocol(is_tty, columns)
    return TerminalCapability(
        is_tty=is_tty,
        columns=columns,
        rows=rows,
        protocol=protocol,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# ImageRenderPlan
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageRenderPlan:
    """Pure data describing what the renderer should draw."""

    capability: TerminalCapability
    rendered_rows: list[dict[str, Any]] = field(default_factory=list)
    skipped_rows: list[dict[str, Any]] = field(default_factory=list)
    target_width: int = 80
    max_height: int = 8
    max_images: int = 6


def plan_render(
    rows: list[dict[str, Any]],
    capability: TerminalCapability,
    max_images: int,
) -> ImageRenderPlan:
    """Build an ``ImageRenderPlan`` without performing any I/O.

    ``max_images == 0`` means unlimited.
    All row splitting happens here so ``render_inline`` stays pure output.
    """
    target_width = capability.columns
    max_height = max(capability.rows // 3, 3)

    if capability.protocol == "disabled":
        return ImageRenderPlan(
            capability=capability,
            rendered_rows=[],
            skipped_rows=[],
            target_width=target_width,
            max_height=max_height,
            max_images=max_images,
        )

    if max_images == 0 or len(rows) <= max_images:
        rendered = list(rows)
        skipped: list[dict[str, Any]] = []
    else:
        rendered = list(rows[:max_images])
        skipped = list(rows[max_images:])

    return ImageRenderPlan(
        capability=capability,
        rendered_rows=rendered,
        skipped_rows=skipped,
        target_width=target_width,
        max_height=max_height,
        max_images=max_images,
    )


# ---------------------------------------------------------------------------
# render_inline
# ---------------------------------------------------------------------------

def _short_hash(h: str) -> str:
    return h[:8]


def _fmt_context(ctx: dict[str, Any]) -> str:
    if not ctx:
        return ""
    return " ".join(f"{k}={v}" for k, v in sorted(ctx.items()))


def _render_pil_image(
    pil_img: "PILImage.Image",
    _target_width: int,
    _max_height: int,
) -> "PILImage.Image":
    """Prepare *pil_img* for terminal rendering without pre-shrinking it.

    The actual render size is delegated to ``textual_image`` via cell-based
    width/height arguments. Pre-resizing here made images tiny in real TTYs.
    """

    orig_w, orig_h = pil_img.size
    if orig_w == 0 or orig_h == 0:
        return pil_img

    # Convert palette / RGBA to RGB for broadest terminal compatibility.
    if pil_img.mode not in ("RGB", "L"):
        try:
            pil_img = pil_img.convert("RGB")
        except Exception:  # noqa: BLE001
            pass

    return pil_img


def render_inline(plan: ImageRenderPlan) -> str:
    """Render the section blocks for all ``plan.rendered_rows``.

    Returns an empty string when ``capability.protocol == "disabled"``.
    The truncation footer is included when ``plan.skipped_rows`` is non-empty.
    """
    if plan.capability.protocol == "disabled" or not plan.rendered_rows:
        return ""

    import io  # noqa: PLC0415

    from rich.console import Console  # noqa: PLC0415
    from rich.text import Text  # noqa: PLC0415

    buf = io.StringIO()
    console = Console(
        file=buf,
        force_terminal=True,
        width=plan.capability.columns,
        highlight=False,
    )

    for index, row in enumerate(plan.rendered_rows):
        run = row["run"]
        name: str = row["name"]
        ctx_str = _fmt_context(row.get("context", {}))

        # Section header.
        if index > 0:
            console.print()
        label = Text()
        label.append("▌ ", style=colors.RULE_LINE)
        label.append(_short_hash(run.hash), style=colors.RUN_HASH)
        exp = run.experiment or run.name or ""
        if exp:
            label.append(f"  {exp}", style=colors.EXPERIMENT)
        label.append(f"  {name}", style=colors.METRIC_NAME)
        if ctx_str:
            label.append(f"  {ctx_str}", style=colors.CONTEXT_VAL)
        console.print(label)

        # Image renderable.
        accessor: Callable[[], "PILImage.Image"] | None = row.get("_image_accessor")
        if accessor is None:
            console.print("[image unavailable: no accessor]", markup=False)
            continue

        try:
            pil_img = accessor()
            pil_img = _render_pil_image(pil_img, plan.target_width, plan.max_height)

            from textual_image.renderable import Image as TxImage  # noqa: PLC0415

            renderable = TxImage(pil_img, width="auto", height=plan.max_height)

            console.print(renderable)
        except Exception as exc:  # noqa: BLE001
            short_reason = str(exc)[:120]
            console.print(f"[image unavailable: {short_reason}]", markup=False)

    # Truncation footer.
    if plan.skipped_rows:
        total = len(plan.rendered_rows) + len(plan.skipped_rows)
        console.print(
            f"... rendered {len(plan.rendered_rows)} of {total} images, "
            "use --max-images=0 for all"
        )

    return buf.getvalue()
