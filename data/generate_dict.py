"""
Generate a bilingual dictionary in order to help align two semantic vector spaces.
This is done using two resources:

    1. A (complete) English - Dutch dictionary from dict.cc.
    2. A list of 1000 most common words in Dutch.
"""

# STD
import codecs
import re
from typing import Dict, List


def read_dictionary(path: str) -> Dict[str, str]:
    """


    """
    entries = {}

    with codecs.open(path, "r", "utf-8") as dict_file:
        for line in dict_file:
            parts = line.strip().split("\t")

            if len(parts) == 1:
                continue

            en, nl = parts[:2]

            # Remove "to" for English verb entries
            if en.startswith("to "):
                en = en[2:]

            # Replace anything in parentheses
            # dict.cc adds extra information there
            parenthesis = [("(", ")"), ("{", "}"), ("<", ">"), ("[", "]")]

            # Replace other kinds of extra info
            en = en.replace("sb.", "").replace("sth.", "").replace("/", "")
            nl = nl.replace("iem.", "").replace("iets", "").replace("/", "")

            for opening, closing in parenthesis:
                en = re.sub(f"\\{opening}.+\\{closing}", "", en)
                nl = re.sub(f"\\{opening}.+\\{closing}", "", nl)

            en, nl = en.strip(), nl.strip()

            if len(en.split(" ")) > 1 or len(nl.split(" ")) > 1:
                print(f"Entry '{en} - {nl}' not generated because of multi-word expression.")
                continue

            entries[nl] = en

    return entries


def read_wordlist(path: str) -> List[str]:

    entries = []

    with codecs.open(path, "r", "utf-8") as word_file:
        for line in word_file.readlines()[1:]:
            entries.append(line.split("\t")[1].strip())

    return entries


def filter_dictionary(dictionary: Dict[str, str],
                      wordlist: List[str]) -> Dict[str, str]:
    filtered_dict = {}

    for word in wordlist:
        try:
            filtered_dict[word] = dictionary[word]
        except KeyError:
            print(f"No dictionary entry found for '{word}'.")

    return filtered_dict


def dump_dictionary(dictionary: Dict[str, str], path: str) -> None:
    with codecs.open(path, "w", "utf-8") as dict_file:
        for nl, en in dictionary.items():

            dict_file.write(f"{nl}\t{en}\n")


if __name__ == "__main__":
    dict_path = "./en_nl_dict.txt"
    filtered_dict_path = "./filtered_en_nl_dict.txt"
    wordlist_path = "./dutch_top1k.txt"

    dictionary = read_dictionary(dict_path)
    print(f"{len(dictionary)} entries found in EN - NL dictionary.")
    wordlist = read_wordlist(wordlist_path)
    filtered_dict = filter_dictionary(dictionary, wordlist)
    print(f"Final dictionary contains {len(filtered_dict)} entries.")
    dump_dictionary(filtered_dict, filtered_dict_path)
