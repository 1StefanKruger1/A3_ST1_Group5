from dataclasses import dataclass
from pathlib import Path

@dataclass
class ImageRecord:
    image_path: Path
    width: int
    height: int
    label: str
    channels: int

