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

    with open("data/wordfreq.txt", "r") as f:
        wordsfreq = [tuple(x.split()) for x in f.read().splitlines()]

    wordsfreq_dict = dict(map(lambda x: (x[0], int(x[1])), wordsfreq))
    wordsfreq_index_map = {}
    for i, word in enumerate(words):
        if wordsfreq_dict.get(word):
            wordsfreq_index_map[i] = wordsfreq_dict.get(word)
        else:
            wordsfreq_index_map[i] = 0

    index: Dict[str, List[int]] = {}
    for rng in range(3, 7):
        index = create_inverse_index(words, index, length=rng)

    for key, val in index.items():
        index[key] = sorted(val, key=lambda x: wordsfreq_index_map[x], reverse=True)

    save_json(index, Path("data/index.json"))
