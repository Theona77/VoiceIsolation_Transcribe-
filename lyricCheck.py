
from pathlib import Path
import re
from jiwer import wer

# 1. Paste the REAL lyrics here
ACTUAL_LYRICS = """
Kau datang tak kala sinar senjaku telah redup
Dan pamit ketika purnamaku penuh seutuhnya
Kau yang singgah tapi tak sungguh
Kau yang singgah tapi tak sungguh
Kukira kau rumah
Nyatanya kau cuma aku sewa
Dari tubuh seorang perempuan
Yang memintamu untuk pulang
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kukira kau rumah
Nyatanya kau cuma aku sewa
Dari tubuh seorang perempuan
Yang memintamu untuk pulang
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
Kau bukan rumah
"""

# 2. Paths to Whisper transcription files
# ORIGINAL_FILE = Path("results/original/KukiraKauRumah.txt")
# VOCALS_FILE = Path("results/vocals/vocals.txt")

WHISPER_DIR = Path("../whisper.cpp")

QUANTIZED_FILE = (
    WHISPER_DIR / "results/quantized5_small/KuNyanyiHaleluya.txt"
)

NORMAL_FILE = (
    WHISPER_DIR / "results/normal_small/KuNyanyiHaleluya.txt"
)

# NORMAL_FILE = Path("results/vocals/vocals.txt")

def normalize(text):
    """Lowercase, remove punctuation, and normalize whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def compare_lyrics(reference, hypothesis):
    reference = normalize(reference)
    hypothesis = normalize(hypothesis)

    error_rate = wer(reference, hypothesis)
    similarity = max(0, (1 - error_rate) * 100)

    return error_rate * 100, similarity


def read_transcript(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Transcript not found: {path}\n"
            "Check the file path and run Whisper first."
        )
    return path.read_text(encoding="utf-8")


def main():
    actual = normalize(ACTUAL_LYRICS)

    if not actual:
        print("Please paste the actual lyrics first.")
        return

    # original = read_transcript(ORIGINAL_FILE)
    # vocals = read_transcript(VOCALS_FILE)

    # original_wer, original_similarity = compare_lyrics(
    #     actual, original
    # )
    # vocals_wer, vocals_similarity = compare_lyrics(
    #     actual, vocals
    # )

    # print("\n===== LYRIC TRANSCRIPTION EVALUATION =====")
    # print(f"Original audio WER: {original_wer:.2f}%")
    # print(f"Original audio similarity: {original_similarity:.2f}%")
    # print()
    # print(f"Vocal-only WER: {vocals_wer:.2f}%")
    # print(f"Vocal-only similarity: {vocals_similarity:.2f}%")
    # print()

    # difference = vocals_similarity - original_similarity
    # print(f"Similarity difference: {difference:+.2f} percentage points")

    # if difference > 0:
    #     print("Result: Vocal isolation performed better.")
    # elif difference < 0:
    #     print("Result: Original audio performed better.")
    # else:
    #     print("Result: Both achieved the same similarity score.")

    quantized = read_transcript(QUANTIZED_FILE)
    normal = read_transcript(NORMAL_FILE)

    quantized_wer, quantized_similarity = compare_lyrics(
        actual, quantized
    )
    normal_wer, normal_similarity = compare_lyrics(
        actual, normal
    )

    print("\n===== WHISPER MODEL COMPARISON: ISOLATED VOCALS =====")
    print(f"Normal small WER: {normal_wer:.2f}%")
    print(f"Normal small score: {normal_similarity:.2f}%")
    print()
    print(f"Quantized small Q5_0 WER: {quantized_wer:.2f}%")
    print(f"Quantized small Q5_0 score: {quantized_similarity:.2f}%")
    print()

    difference = quantized_similarity - normal_similarity
    print(f"Quantized - Normal: {difference:+.2f} percentage points")

    if difference > 0:
        print("Result: Quantized model scored better.")
    elif difference < 0:
        print("Result: Normal model scored better.")
    else:
        print("Result: Both models scored equally.")

if __name__ == "__main__":
    main()
