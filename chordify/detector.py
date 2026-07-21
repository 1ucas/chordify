"""Chord detection using librosa chroma features."""

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from .notation import note_name, prefer_flats_for_key, respell_chord_name


@dataclass
class Chord:
    """A detected chord with timing information."""

    start_time: float
    end_time: float
    name: str

    @property
    def duration(self) -> float:
        """Duration of the chord in seconds."""
        return self.end_time - self.start_time


@dataclass(frozen=True)
class KeyContext:
    """Estimated song key used as a soft chord prior."""

    tonic: int
    mode: str
    confidence: float

    @property
    def name(self) -> str:
        suffix = "m" if self.mode == "minor" else ""
        return f"{NOTE_NAMES_SHARP[self.tonic]}{suffix}"


# Chord templates for major and minor chords only
# Each template is a 12-element array representing pitch classes (C, C#, D, ..., B)
# Equal weights: root-heavy templates systematically mis-vote for whichever
# chord's root matches the loudest pitch class (usually the bass note — and pop
# bass lines spend half their time on the fifth), which was the single largest
# accuracy loss on the ground-truth benchmark (scripts/chord_eval.py).
NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
ROOT_WEIGHT = 1.0
THIRD_WEIGHT = 1.0
FIFTH_WEIGHT = 1.0
NON_CHORD_ENERGY_WEIGHT = 0.5  # Penalty applied to energy on non-chord tones
BASS_ROOT_BOOST = 0.15  # Boost when bass supports the chord root
BASS_ROOT_PROMINENCE_BOOST = 0.1  # Extra boost if root stands out in bass
BASS_ROOT_MISMATCH_PENALTY = 0.15  # Mild penalty if bass favors a non-chord tone
BASS_MISMATCH_MARGIN = 0.05  # Ignore tiny bass differences
PRIMARY_KEY_BONUS = 0.05
SECONDARY_DIATONIC_PENALTY = 0.02
NON_DIATONIC_KEY_PENALTY = 0.12
CHORD_CHANGE_PENALTY = 0.18
NO_CHORD_CHANGE_PENALTY = 0.10
BEATS_PER_CHORD_SEGMENT = 1
MIN_SEGMENT_SECONDS = 0.25  # absorb beats shorter than this (handles variable tempo)
SONG_PALETTE_MIN_SHARE = 0.025
SONG_PALETTE_RARE_PENALTY = 0.05
SONG_PALETTE_COMMON_BONUS = 0.02
EXTENSION_MIN_ENERGY = 0.56
SLASH_BASS_MARGIN = 0.20
CENS_WEIGHT = 0.7  # Blend of smoothed CENS vs raw CQT chroma for chord scoring

LOW_C_HZ = librosa.note_to_hz("C1")
BASS_C_HZ = librosa.note_to_hz("C2")

# Krumhansl-Schmuckler key profiles. They are not a hard rule; they only bias
# Viterbi away from chromatic outliers when the audio evidence is ambiguous.
MAJOR_KEY_PROFILE = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float32,
)
MINOR_KEY_PROFILE = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float32,
)


def _build_chord_templates():
    """Build chord templates with weighted notes - major and minor only."""
    templates = {}
    root_indices = {}
    # Use sharps for note names (standard), add flat aliases
    note_names_flat = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    for i, (sharp, flat) in enumerate(zip(NOTE_NAMES_SHARP, note_names_flat)):
        # Major chord: root, major third (+4), perfect fifth (+7)
        major = [0.0] * 12
        major[i] = ROOT_WEIGHT
        major[(i + 4) % 12] = THIRD_WEIGHT
        major[(i + 7) % 12] = FIFTH_WEIGHT
        templates[sharp] = major
        root_indices[sharp] = i
        if flat != sharp:
            templates[flat] = major  # Flat alias
            root_indices[flat] = i

        # Minor chord: root, minor third (+3), perfect fifth (+7)
        minor = [0.0] * 12
        minor[i] = ROOT_WEIGHT
        minor[(i + 3) % 12] = THIRD_WEIGHT
        minor[(i + 7) % 12] = FIFTH_WEIGHT
        templates[f"{sharp}m"] = minor
        root_indices[f"{sharp}m"] = i
        if flat != sharp:
            templates[f"{flat}m"] = minor  # Flat alias
            root_indices[f"{flat}m"] = i

    return templates, root_indices

CHORD_TEMPLATES, CHORD_ROOT_INDEX = _build_chord_templates()

# Convert templates to numpy arrays, build masks, and pre-normalize
CHORD_TEMPLATE_ARRAYS = {}
CHORD_TEMPLATE_MASKS = {}
for name, template in CHORD_TEMPLATES.items():
    arr = np.array(template, dtype=np.float32)
    mask = arr > 0
    norm = np.linalg.norm(arr)
    if norm == 0:
        continue
    CHORD_TEMPLATE_ARRAYS[name] = arr / norm
    CHORD_TEMPLATE_MASKS[name] = mask

CANONICAL_CHORD_NAMES = (
    ["N"]
    + NOTE_NAMES_SHARP
    + [f"{note}m" for note in NOTE_NAMES_SHARP]
)


def _normalize_vector(vector: np.ndarray) -> np.ndarray | None:
    """Return an L2-normalized copy, or None for silent vectors."""
    norm = np.linalg.norm(vector)
    if norm <= 0:
        return None
    return vector / norm


def _score_chord(
    chord_name: str,
    chroma_normalized: np.ndarray,
    bass_normalized: np.ndarray | None = None,
    bass_mean: float = 0.0,
    bass_max: float = 0.0,
) -> float:
    """Score one chord template against normalized chroma inputs."""
    template_norm = CHORD_TEMPLATE_ARRAYS[chord_name]
    mask = CHORD_TEMPLATE_MASKS[chord_name]

    score = float(np.dot(chroma_normalized, template_norm))

    # Penalize energy that lands on non-chord tones to avoid ambiguous matches.
    non_chord_energy = float(np.sum(np.square(chroma_normalized[~mask])))
    score -= NON_CHORD_ENERGY_WEIGHT * non_chord_energy

    if bass_normalized is not None:
        root_idx = CHORD_ROOT_INDEX.get(chord_name)
        if root_idx is not None:
            root_energy = float(bass_normalized[root_idx])
            bass_prominence = root_energy - bass_mean
            score += BASS_ROOT_BOOST * root_energy
            score += BASS_ROOT_PROMINENCE_BOOST * bass_prominence
            # Pop bass lines constantly alternate root and fifth (and walk
            # through the third), so a chord-tone bass is never evidence
            # against the chord. Only penalize a dominant NON-chord-tone bass.
            bass_idx = int(np.argmax(bass_normalized))
            if (
                not mask[bass_idx]
                and bass_max > root_energy + BASS_MISMATCH_MARGIN
            ):
                score -= BASS_ROOT_MISMATCH_PENALTY * (bass_max - root_energy)

    return score


def match_chord(
    chroma_vector: np.ndarray,
    chroma_bass: np.ndarray | None = None,
) -> tuple[str, float]:
    """
    Match a chroma vector to the best fitting chord template.

    Args:
        chroma_vector: 12-element array of chroma features
        chroma_bass: 12-element array of bass-focused chroma (optional)

    Returns:
        Tuple of (chord_name, correlation_score)
    """
    best_chord = "N"
    best_score = 0.0

    chroma_normalized = _normalize_vector(chroma_vector)
    if chroma_normalized is None:
        return "N", 0.0

    bass_normalized = None
    bass_mean = 0.0
    bass_max = 0.0
    if chroma_bass is not None:
        bass_normalized = _normalize_vector(chroma_bass)
        if bass_normalized is not None:
            bass_mean = float(np.mean(bass_normalized))
            bass_max = float(np.max(bass_normalized))

    for chord_name, template_norm in CHORD_TEMPLATE_ARRAYS.items():
        score = _score_chord(
            chord_name,
            chroma_normalized,
            bass_normalized,
            bass_mean,
            bass_max,
        )

        if score > best_score:
            best_score = score
            best_chord = chord_name

    return best_chord, best_score


def estimate_key(chroma: np.ndarray) -> KeyContext | None:
    """Estimate the global key from song-level chroma."""
    global_chroma = np.mean(chroma, axis=1)
    global_chroma = _normalize_vector(global_chroma)
    if global_chroma is None:
        return None

    major_profile = _normalize_vector(MAJOR_KEY_PROFILE)
    minor_profile = _normalize_vector(MINOR_KEY_PROFILE)
    if major_profile is None or minor_profile is None:
        return None

    candidates: list[tuple[float, int, str]] = []
    for tonic in range(12):
        major_score = float(np.dot(global_chroma, np.roll(major_profile, tonic)))
        minor_score = float(np.dot(global_chroma, np.roll(minor_profile, tonic)))
        candidates.append((major_score, tonic, "major"))
        candidates.append((minor_score, tonic, "minor"))

    candidates.sort(reverse=True, key=lambda item: item[0])
    best_score, tonic, mode = candidates[0]
    second_score = candidates[1][0] if len(candidates) > 1 else 0.0
    return KeyContext(tonic=tonic, mode=mode, confidence=best_score - second_score)


def _is_diatonic_chord(chord_name: str, key: KeyContext) -> bool:
    """Return whether a major/minor triad fits a simple major/minor key model."""
    root = CHORD_ROOT_INDEX.get(chord_name)
    if root is None:
        return True

    degree = (root - key.tonic) % 12
    is_minor = chord_name.endswith("m")

    if key.mode == "major":
        return (
            (not is_minor and degree in {0, 5, 7})
            or (is_minor and degree in {2, 4, 9})
        )

    return (
        (is_minor and degree in {0, 5, 7})
        or (not is_minor and degree in {3, 8, 10})
        or (not is_minor and degree == 7)  # common major V in minor keys
    )


def _key_prior(chord_name: str, key: KeyContext | None) -> float:
    """Softly bias ambiguous chord choices toward the estimated key."""
    if key is None or chord_name == "N":
        return 0.0

    root = CHORD_ROOT_INDEX.get(chord_name)
    if root is None:
        return 0.0

    degree = (root - key.tonic) % 12
    is_minor = chord_name.endswith("m")

    if key.mode == "major":
        if (not is_minor and degree in {0, 5, 7}) or (is_minor and degree == 9):
            return PRIMARY_KEY_BONUS
        if _is_diatonic_chord(chord_name, key):
            return -SECONDARY_DIATONIC_PENALTY
        return -NON_DIATONIC_KEY_PENALTY

    if (is_minor and degree in {0, 5, 7}) or (not is_minor and degree in {3, 8, 10}):
        return PRIMARY_KEY_BONUS
    if _is_diatonic_chord(chord_name, key):
        return -SECONDARY_DIATONIC_PENALTY
    return -NON_DIATONIC_KEY_PENALTY


def _build_beat_boundaries(
    y: np.ndarray,
    sr: int,
    hop_length: int,
    duration: float,
    beats_per_segment: int = BEATS_PER_CHORD_SEGMENT,
    min_segment_seconds: float = MIN_SEGMENT_SECONDS,
) -> np.ndarray:
    """Build bar-ish segment boundaries, with a fixed-grid fallback.

    Groups every `beats_per_segment` beats; if that still leaves a segment
    shorter than `min_segment_seconds`, additional beats are absorbed until
    the minimum is met. This compensates for librosa over-estimating tempo
    on slow songs (where 2 beats may only span ~0.7 s).
    """
    try:
        _tempo, beat_frames = librosa.beat.beat_track(
            y=y,
            sr=sr,
            hop_length=hop_length,
            trim=False,
        )
    except Exception:
        beat_frames = np.array([], dtype=np.int64)

    if len(beat_frames) >= 4:
        beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)
        beat_boundaries = np.unique(
            np.clip(np.concatenate(([0.0], beat_times, [duration])), 0.0, duration)
        )
        if beats_per_segment > 1 and len(beat_boundaries) > beats_per_segment + 1:
            kept = [0]
            i = 0
            while i < len(beat_boundaries) - 1:
                j = min(i + beats_per_segment, len(beat_boundaries) - 1)
                # Extend until min segment seconds is satisfied (or we run out).
                while (
                    j < len(beat_boundaries) - 1
                    and beat_boundaries[j] - beat_boundaries[i] < min_segment_seconds
                ):
                    j += 1
                kept.append(j)
                i = j
            boundaries = beat_boundaries[np.array(kept, dtype=int)]
        else:
            boundaries = beat_boundaries
    else:
        boundaries = np.arange(0.0, duration + 1.0, 1.0)
        if boundaries[-1] < duration:
            boundaries = np.append(boundaries, duration)

    boundaries = np.unique(np.clip(boundaries, 0.0, duration))
    if len(boundaries) < 2:
        return np.array([0.0, duration], dtype=np.float32)
    return boundaries


def _aggregate_chroma(
    chroma: np.ndarray,
    times: np.ndarray,
    boundaries: np.ndarray,
) -> np.ndarray:
    """Aggregate frame-level chroma into timing segments."""
    aggregated = np.zeros((12, len(boundaries) - 1), dtype=np.float32)

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        start_idx = int(np.searchsorted(times, start, side="left"))
        end_idx = int(np.searchsorted(times, end, side="left"))
        if end_idx <= start_idx:
            end_idx = min(start_idx + 1, chroma.shape[1])

        segment = chroma[:, start_idx:end_idx]
        if segment.size:
            aggregated[:, i] = np.median(segment, axis=1)

    norms = np.linalg.norm(aggregated, axis=0, keepdims=True)
    norms[norms == 0.0] = 1.0
    return aggregated / norms


def _smooth_segment_chroma(chroma: np.ndarray) -> np.ndarray:
    """Apply light temporal smoothing so one noisy segment cannot dominate."""
    if chroma.shape[1] < 3:
        return chroma

    padded = np.pad(chroma, ((0, 0), (1, 1)), mode="edge")
    smoothed = (0.25 * padded[:, :-2]) + (0.5 * padded[:, 1:-1]) + (0.25 * padded[:, 2:])
    norms = np.linalg.norm(smoothed, axis=0, keepdims=True)
    norms[norms == 0.0] = 1.0
    return smoothed / norms


def _build_emissions(
    chroma: np.ndarray,
    chroma_bass: np.ndarray,
    chord_names: list[str],
    key: KeyContext | None,
    min_confidence: float,
) -> np.ndarray:
    """Score every chord state for every beat-synchronous segment."""
    emissions = np.full((chroma.shape[1], len(chord_names)), -np.inf, dtype=np.float32)

    for i in range(chroma.shape[1]):
        chroma_normalized = _normalize_vector(chroma[:, i])
        if chroma_normalized is None:
            emissions[i, 0] = min_confidence
            continue

        bass_normalized = _normalize_vector(chroma_bass[:, i])
        bass_mean = float(np.mean(bass_normalized)) if bass_normalized is not None else 0.0
        bass_max = float(np.max(bass_normalized)) if bass_normalized is not None else 0.0

        emissions[i, 0] = min_confidence - 0.08
        for state_idx, chord_name in enumerate(chord_names[1:], start=1):
            score = _score_chord(
                chord_name,
                chroma_normalized,
                bass_normalized,
                bass_mean,
                bass_max,
            )
            emissions[i, state_idx] = score + _key_prior(chord_name, key)

    return emissions


def _viterbi_decode(emissions: np.ndarray, chord_names: list[str]) -> np.ndarray:
    """Decode the most stable chord path through beat-synchronous emissions."""
    segment_count, state_count = emissions.shape
    if segment_count == 0:
        return np.array([], dtype=np.int32)

    transition = np.full((state_count, state_count), -CHORD_CHANGE_PENALTY, dtype=np.float32)
    np.fill_diagonal(transition, 0.0)
    transition[:, 0] -= NO_CHORD_CHANGE_PENALTY
    transition[0, :] -= NO_CHORD_CHANGE_PENALTY

    scores = np.full((segment_count, state_count), -np.inf, dtype=np.float32)
    backpointers = np.zeros((segment_count, state_count), dtype=np.int32)
    scores[0] = emissions[0]

    for i in range(1, segment_count):
        previous = scores[i - 1][:, np.newaxis] + transition
        backpointers[i] = np.argmax(previous, axis=0)
        scores[i] = emissions[i] + np.max(previous, axis=0)

    path = np.zeros(segment_count, dtype=np.int32)
    path[-1] = int(np.argmax(scores[-1]))
    for i in range(segment_count - 2, -1, -1):
        path[i] = backpointers[i + 1, path[i + 1]]

    return path


def _apply_song_palette_prior(
    emissions: np.ndarray,
    initial_path: np.ndarray,
) -> np.ndarray:
    """Use a first pass to mildly penalize chords that barely appear in the song."""
    if len(initial_path) == 0:
        return emissions

    counts = np.bincount(initial_path, minlength=emissions.shape[1]).astype(np.float32)
    counts[0] = 0.0
    total = float(np.sum(counts))
    if total <= 0.0:
        return emissions

    shares = counts / total
    adjusted = emissions.copy()
    for state_idx, share in enumerate(shares):
        if state_idx == 0:
            continue
        if share < SONG_PALETTE_MIN_SHARE:
            adjusted[:, state_idx] -= SONG_PALETTE_RARE_PENALTY
        elif share >= 0.08:
            adjusted[:, state_idx] += SONG_PALETTE_COMMON_BONUS
    return adjusted


def _suppress_isolated_path_changes(path: np.ndarray) -> np.ndarray:
    """Absorb single-segment chord flickers between two matching neighbours."""
    if len(path) < 3:
        return path

    smoothed = path.copy()
    for i in range(1, len(path) - 1):
        if path[i - 1] == path[i + 1] and path[i] != path[i - 1]:
            smoothed[i] = path[i - 1]
    return smoothed


def _path_to_chords(
    path: np.ndarray,
    chord_names: list[str],
    boundaries: np.ndarray,
    min_duration: float,
) -> list[Chord]:
    """Convert a decoded state path into merged chord spans."""
    if len(path) == 0:
        return []

    chords: list[Chord] = []
    current_idx = int(path[0])
    start_time = float(boundaries[0])

    for i in range(1, len(path)):
        state_idx = int(path[i])
        if state_idx == current_idx:
            continue

        if chord_names[current_idx] != "N":
            chords.append(
                Chord(
                    start_time=start_time,
                    end_time=float(boundaries[i]),
                    name=chord_names[current_idx],
                )
            )
        current_idx = state_idx
        start_time = float(boundaries[i])

    if chord_names[current_idx] != "N":
        chords.append(
            Chord(
                start_time=start_time,
                end_time=float(boundaries[-1]),
                name=chord_names[current_idx],
            )
        )

    return [chord for chord in chords if chord.duration >= min_duration]


def _segment_slice_for_chord(boundaries: np.ndarray, chord: Chord) -> slice:
    """Return the segment slice overlapping a chord span."""
    start_idx = int(np.searchsorted(boundaries, chord.start_time, side="right") - 1)
    end_idx = int(np.searchsorted(boundaries, chord.end_time, side="left"))
    start_idx = max(start_idx, 0)
    end_idx = max(end_idx, start_idx + 1)
    return slice(start_idx, end_idx)


def _decorate_chord_name(
    chord_name: str,
    chroma_vector: np.ndarray,
    bass_vector: np.ndarray,
    prefer_flats: bool,
) -> str:
    """Add conservative extensions/slash bass and apply display spelling."""
    if chord_name == "N":
        return chord_name

    root = CHORD_ROOT_INDEX.get(chord_name)
    if root is None:
        return respell_chord_name(chord_name, prefer_flats)

    chroma_normalized = _normalize_vector(chroma_vector)
    if chroma_normalized is None:
        return respell_chord_name(chord_name, prefer_flats)

    is_minor = chord_name.endswith("m")
    third = (root + (3 if is_minor else 4)) % 12
    fifth = (root + 7) % 12
    second = (root + 2) % 12
    fourth = (root + 5) % 12
    flat_seventh = (root + 10) % 12

    root_text = note_name(root, prefer_flats)
    suffix = "m" if is_minor else ""
    third_energy = float(chroma_normalized[third])

    if (
        not is_minor
        and float(chroma_normalized[fourth]) >= EXTENSION_MIN_ENERGY
        and third_energy < 0.22
    ):
        suffix = "sus4"
    elif (
        not is_minor
        and float(chroma_normalized[second]) >= EXTENSION_MIN_ENERGY
        and third_energy < 0.22
    ):
        suffix = "sus2"
    elif not is_minor and float(chroma_normalized[second]) >= EXTENSION_MIN_ENERGY:
        suffix = "9"

    if (
        suffix in {"", "m"}
        and float(chroma_normalized[flat_seventh]) >= EXTENSION_MIN_ENERGY
        and float(chroma_normalized[flat_seventh]) > third_energy * 0.9
    ):
        suffix = f"{suffix}7"

    slash = ""
    bass_normalized = _normalize_vector(bass_vector)
    if bass_normalized is not None:
        bass_idx = int(np.argmax(bass_normalized))
        chord_tones = {root, third, fifth}
        if (
            bass_idx in chord_tones
            and bass_idx != root
            and float(bass_normalized[bass_idx]) > float(bass_normalized[root]) + SLASH_BASS_MARGIN
        ):
            slash = f"/{note_name(bass_idx, prefer_flats)}"

    return f"{root_text}{suffix}{slash}"


def _decorate_chords(
    chords: list[Chord],
    chroma: np.ndarray,
    chroma_bass: np.ndarray,
    boundaries: np.ndarray,
    prefer_flats: bool,
) -> list[Chord]:
    """Apply notation-level improvements after the stable chord path is decoded."""
    decorated: list[Chord] = []
    for chord in chords:
        segment_slice = _segment_slice_for_chord(boundaries, chord)
        chord_chroma = np.median(chroma[:, segment_slice], axis=1)
        chord_bass = np.median(chroma_bass[:, segment_slice], axis=1)
        decorated.append(
            Chord(
                start_time=chord.start_time,
                end_time=chord.end_time,
                name=_decorate_chord_name(
                    chord_name=chord.name,
                    chroma_vector=chord_chroma,
                    bass_vector=chord_bass,
                    prefer_flats=prefer_flats,
                ),
            )
        )
    return decorated


def detect_chords(
    audio_path: Path,
    verbose: bool = False,
    hop_length: int = 512,
    min_confidence: float = 0.52,
    min_chord_duration: float = 0.7,
) -> list[Chord]:
    """
    Detect chords in an audio file using librosa chroma features.

    Args:
        audio_path: Path to the audio file (WAV format recommended)
        verbose: Whether to show progress information
        hop_length: Number of samples between frames (larger = faster, less precise)
        min_confidence: Minimum score needed before "N" becomes competitive
        min_chord_duration: Minimum duration for a reported chord

    Returns:
        List of Chord objects with timing information
    """
    if verbose:
        print(f"Analyzing chords in: {audio_path.name}")

    # Load audio
    y, sr = librosa.load(str(audio_path), sr=22050)
    duration = len(y) / sr

    if verbose:
        print(f"Audio duration: {duration:.1f} seconds")

    # Harmonic-Percussive Source Separation: filter out drums/percussion
    # This significantly improves chroma accuracy by removing percussive noise
    y_harmonic, _ = librosa.effects.hpss(y, margin=3.0)

    # Estimate tuning to correct for slightly detuned recordings
    tuning = librosa.estimate_tuning(y=y_harmonic, sr=sr)
    if verbose:
        print(f"Estimated tuning offset: {tuning:.2f} semitones")

    # Extract CQT chroma features with tuning correction
    # Use full-range chroma for chord tones
    chroma_full = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        tuning=tuning,
        n_chroma=12,
        fmin=BASS_C_HZ,
        n_octaves=5,
    )

    # Extract bass-focused chroma (lower octaves where root notes are strongest)
    chroma_bass = librosa.feature.chroma_cqt(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        tuning=tuning,
        n_chroma=12,
        fmin=BASS_C_HZ,
        n_octaves=3,
    )

    chroma_cens = librosa.feature.chroma_cens(
        y=y_harmonic,
        sr=sr,
        hop_length=hop_length,
        tuning=tuning,
        n_chroma=12,
        fmin=BASS_C_HZ,
        n_octaves=5,
        win_len_smooth=11,
    )

    frame_count = min(chroma_full.shape[1], chroma_bass.shape[1], chroma_cens.shape[1])
    chroma_full = chroma_full[:, :frame_count]
    chroma_bass = chroma_bass[:, :frame_count]
    chroma_cens = chroma_cens[:, :frame_count]

    # Re-normalize each frame so scoring compares relative pitch content, not loudness
    frame_norms_full = np.linalg.norm(chroma_full, axis=0, keepdims=True)
    frame_norms_full[frame_norms_full == 0.0] = 1.0
    chroma_full = chroma_full / frame_norms_full

    frame_norms_cens = np.linalg.norm(chroma_cens, axis=0, keepdims=True)
    frame_norms_cens[frame_norms_cens == 0.0] = 1.0
    chroma_cens = chroma_cens / frame_norms_cens

    frame_norms_bass = np.linalg.norm(chroma_bass, axis=0, keepdims=True)
    frame_norms_bass[frame_norms_bass == 0.0] = 1.0
    chroma_bass = chroma_bass / frame_norms_bass

    chroma = (CENS_WEIGHT * chroma_cens) + ((1.0 - CENS_WEIGHT) * chroma_full)
    frame_norms = np.linalg.norm(chroma, axis=0, keepdims=True)
    frame_norms[frame_norms == 0.0] = 1.0
    chroma = chroma / frame_norms

    # Calculate time for each frame
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    boundaries = _build_beat_boundaries(
        y,
        sr,
        hop_length,
        duration,
        beats_per_segment=BEATS_PER_CHORD_SEGMENT,
        min_segment_seconds=MIN_SEGMENT_SECONDS,
    )
    beat_chroma = _smooth_segment_chroma(_aggregate_chroma(chroma, times, boundaries))
    beat_bass = _smooth_segment_chroma(_aggregate_chroma(chroma_bass, times, boundaries))

    key = estimate_key(beat_chroma)
    if verbose and key is not None:
        print(f"Estimated key: {key.name} (confidence {key.confidence:.3f})")

    chord_names = CANONICAL_CHORD_NAMES
    emissions = _build_emissions(
        chroma=beat_chroma,
        chroma_bass=beat_bass,
        chord_names=chord_names,
        key=key,
        min_confidence=min_confidence,
    )
    initial_path = _viterbi_decode(emissions, chord_names)
    emissions = _apply_song_palette_prior(emissions, initial_path)
    path = _suppress_isolated_path_changes(_viterbi_decode(emissions, chord_names))
    chords = _path_to_chords(
        path=path,
        chord_names=chord_names,
        boundaries=boundaries,
        min_duration=min_chord_duration,
    )
    if key is not None:
        chords = _decorate_chords(
            chords=chords,
            chroma=beat_chroma,
            chroma_bass=beat_bass,
            boundaries=boundaries,
            prefer_flats=prefer_flats_for_key(key.tonic, key.mode),
        )

    if verbose:
        print(f"Detected {len(chords)} chord changes")

    return chords


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def format_chords_only(chords: list[Chord]) -> str:
    """Format chords as timestamped list."""
    lines = []
    for chord in chords:
        timestamp = format_timestamp(chord.start_time)
        lines.append(f"{timestamp}\t{chord.name}")
    return "\n".join(lines)
