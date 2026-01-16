"""Recording and export utilities for thesis figures."""

from pathlib import Path
from typing import Optional
from datetime import datetime
import pygame


class Recorder:
    """
    Record simulation frames for thesis figures.

    Supports:
    - Single frame screenshots
    - Frame sequences for video export
    """

    def __init__(self, output_dir: Path | str = "recordings") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._recording = False
        self._frame_count = 0
        self._session_dir: Optional[Path] = None

    def screenshot(self, surface: pygame.Surface, name: Optional[str] = None) -> Path:
        """
        Take a single screenshot.

        Args:
            surface: Surface to capture
            name: Optional filename (without extension)

        Returns:
            Path to saved image
        """
        if name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            name = f"screenshot_{timestamp}"

        path = self.output_dir / f"{name}.png"
        pygame.image.save(surface, str(path))
        return path

    def start_recording(self, session_name: Optional[str] = None) -> None:
        """Start recording frames."""
        if session_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            session_name = f"recording_{timestamp}"

        self._session_dir = self.output_dir / session_name
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._frame_count = 0
        self._recording = True

    def stop_recording(self) -> Path | None:
        """Stop recording and return session directory."""
        self._recording = False
        return self._session_dir

    def record_frame(self, surface: pygame.Surface) -> None:
        """Record a single frame (if recording active)."""
        if not self._recording or self._session_dir is None:
            return

        path = self._session_dir / f"frame_{self._frame_count:06d}.png"
        pygame.image.save(surface, str(path))
        self._frame_count += 1

    @property
    def is_recording(self) -> bool:
        """Check if currently recording."""
        return self._recording

    @property
    def frame_count(self) -> int:
        """Get number of recorded frames."""
        return self._frame_count
