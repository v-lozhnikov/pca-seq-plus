def read_seq(filename: str) -> str:
    with open(filename) as f:
        lines = [s.strip() for s in f.readlines()]
    s = ''.join(lines[1:])
    return s


def read_name(filename: str) -> str:
    with open(filename) as f:
        lines = [s.strip() for s in f.readlines()]
    s = lines[0]
    name = ' '.join(s[s.find('OS=') + len('OS='):s.rfind('OX=') - 1].split()[:2])
    return name


def read_seqs(filenames: [str]) -> [str]:
    return [read_seq(filename) for filename in filenames]


def read_seqs(filename: str, window: int, step: int) -> [str]:
    s = read_seq(filename)
    return [s[i: i + window] for i in range(0, len(s) - window + 1, step)]


def read_seqs_from_str(s: str, window: int, step: int) -> [str]:
    return [s[i: i + window] for i in range(0, len(s) - window + 1, step)]

