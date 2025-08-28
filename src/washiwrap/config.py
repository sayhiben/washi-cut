from dataclasses import dataclass

@dataclass
class AppConfig:
    """
    Application configuration; single immutable bag of options passed around.
    """
    stl_path: str
    tape_width_mm: float
    out_svg_path: str = "washi_wrap.svg"
    stl_unit: str = "mm"  # "mm" or "inch"
    shrink_mm: float = 0.0
    gap_mm: float = 2.0
    margin_mm: float = 1.0
    duplicates: int = 1
    mode: str = "bfs"  # "bfs" or "hamiltonian"
    seed: int = 0

    # Hamiltonian search tuning
    ham_beam: int = 24
    ham_timeout_s: float = 2.0
    ham_allow_fallback: bool = True

    def validate(self) -> None:
        assert self.tape_width_mm > 0
        assert self.duplicates >= 1
        assert self.mode in ("bfs", "hamiltonian")
        assert self.stl_unit in ("mm", "inch")
