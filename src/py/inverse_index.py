import codecs
import json

from pathlib import Path
from typing import List, Dict

from tqdm import tqdm


def save_json(js: dict, save_path: Path):
    f = codecs.open(str(save_path), mode="w", encoding="utf-8")
    json.dump(js, f, indent=4, ensure_ascii=False, cls=json.JSONEncoder)
    f.close()


def create_inverse_index(words: List[str],
                         index: Dict[str, List[int]],
                         length: int = 3) -> Dict[str, List[int]]:
    for i, word in enumerate(
            tqdm(words, leave=False, desc=f"length: {length}")):
        if len(word) < length:
            continue

        prefix = word[:length]
        if index.get(prefix) is None:
            index[prefix] = [i]
        else:
            index[prefix].append(i)
    return index


if __name__ == "__main__":
    with open("data/words.txt", "r") as f:
        words = f.read().splitlines()

    index: Dict[str, List[int]] = {}
    for rng in range(3, 7):
        index = create_inverse_index(words, index, length=rng)

    save_json(index, Path("data/index.json"))
