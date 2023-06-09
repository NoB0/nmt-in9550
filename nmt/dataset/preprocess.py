"""Data preprocessing script."""
import random
import xml.etree.ElementTree as ET


def get_sentences(path):
    tree = ET.parse(path)
    root = tree.getroot()

    sentences, n_words = [], 0
    for sentence in root[-1]:
        assert sentence[0].get("{http://www.w3.org/XML/1998/namespace}lang") == "en"
        sentences.append((sentence[1][0].text, sentence[0][0].text))
        n_words += len(sentence[1][0].text.split(" "))

    print(f"{n_words} words found in {path}")
    return sentences


def deduplicate(sentences, no_set=set()):
    deduplicated = []
    for no, en in sentences:
        if no in no_set:
            continue
        deduplicated.append((no, en))
        no_set.add(no)

    print(f"{len(sentences)} -> {len(deduplicated)}")
    return deduplicated, no_set


def write(filename: str, sentences: list, filtered=False, limit_size=None):
    n_skipped, n_written = 0, 0
    with open(filename, "w") as f:
        for no, en in sentences:
            no = " ".join(no.strip().split())
            en = " ".join(en.strip().split())

            if (no == en and len(no) > 32) or (
                (len(no) > 16 or len(en) > 16)
                and (len(no) > 2 * len(en) or len(en) > 2 * len(no))
            ):
                n_skipped += 1
                continue

            if filtered and (len(no) + len(en) > 384 or len(no) + len(en) < 8):
                n_skipped += 1
                continue

            if filtered and (
                no.count(".") != en.count(".")
                or no.count("!") != en.count("!")
                or no.count("?") != en.count("?")
                or no.count("(") != en.count("(")
                or no.count(")") != en.count(")")
                or no.count(":") != en.count(":")
            ):
                n_skipped += 1
                continue

            if filtered:
                no = no.lstrip("-").strip()
                en = en.lstrip("-").strip()

            f.write(f"{no}\t{en}\n")
            n_written += 1

            if limit_size and n_written >= limit_size:
                break

    print(
        f"Dumped {n_written} lines into {filename}, skipped {n_skipped} dubious sentences"
    )


if __name__ == "__main__":
    random.seed("IN5550")

    sentences_europarl = get_sentences(
        "/fp/projects01/ec30/IN5550/ec-nmbernar/nmt/data/europarl-en-fr.tmx"
    )
    random.shuffle(sentences_europarl)
    sentences_europarl, _ = deduplicate(sentences_europarl)
    write(
        "data/train_government.txt", sentences_europarl, filtered=True, limit_size=50000
    )
    write(
        "data/valid_government.txt",
        sentences_europarl[-100000:],
        filtered=True,
        limit_size=2500,
    )
    del sentences_europarl

    sentences_book = get_sentences(
        "/fp/projects01/ec30/IN5550/ec-nmbernar/nmt/data/book-en-fr.tmx"
    )
    random.shuffle(sentences_book)
    sentences_book, _ = deduplicate(sentences_book)
    write("data/valid_book.txt", sentences_book[1:], filtered=True, limit_size=2500)
    del sentences_book

    sentences_subtitles = get_sentences(
        "/fp/projects01/ec30/IN5550/ec-nmbernar/nmt/data/sample-open-subtitles-en-fr.tmx"
    )
    random.shuffle(sentences_subtitles)
    sentences_subtitles, _ = deduplicate(sentences_subtitles)
    write(
        "data/train_subtitles.txt",
        sentences_subtitles,
        filtered=True,
        limit_size=250000,
    )
    write(
        "data/valid_subtitles.txt",
        sentences_subtitles[-100000:],
        filtered=True,
        limit_size=2500,
    )
