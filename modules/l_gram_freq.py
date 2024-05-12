def get_all_l_grams(seqs: [str], l: int) -> [str]:
    l_grams = set()
    for seq in seqs:
        seq_len = len(seq)
        for i in range(seq_len - l + 1):
            l_gram = seq[i: i + l]
            l_grams.add(l_gram)
    return sorted(list(l_grams))


def calculate_frequencies(seqs: [str], l: int) -> list:
    all_l_grams = get_all_l_grams(seqs, l)
    freq_dicts = []
    for seq in seqs:
        l_gram_count = dict.fromkeys(all_l_grams, 0)
        seq_len = len(seq)
        for i in range(seq_len - l + 1):
            l_gram = seq[i: i + l]
            l_gram_count[l_gram] += 1
        l_gram_freq = {k: v / (seq_len - l + 1) for k, v in l_gram_count.items()}
        freq_dicts.append(l_gram_freq)

    freq_lists = [[] for _ in seqs]
    for l_gram in all_l_grams:
        for i in range(len(seqs)):
            freq_lists[i].append(freq_dicts[i][l_gram])

    return freq_lists
