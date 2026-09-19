#!/usr/bin/env python3
"""Select evenly spaced alignment columns for CASTER scaling benchmarks."""
import argparse
import itertools
import os


def read_fasta(path):
    name = None
    sequence = []
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    yield name, "".join(sequence)
                name = line
                sequence = []
            else:
                if name is None:
                    raise ValueError("sequence data precedes the first header")
                sequence.append(line)
    if name is not None:
        yield name, "".join(sequence)


def sample_alignment(input_path, output_path, site_count):
    records = iter(read_fasta(input_path))
    try:
        first_name, first_sequence = next(records)
    except StopIteration as error:
        raise ValueError("alignment is empty") from error
    alignment_length = len(first_sequence)
    if site_count <= 0 or site_count > alignment_length:
        raise ValueError(
            f"sites must be in 1..{alignment_length}, received {site_count}")
    indices = [
        (index * alignment_length) // site_count
        for index in range(site_count)
    ]

    output_directory = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_directory, exist_ok=True)
    temporary_path = f"{output_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w") as output:
            for name, sequence in itertools.chain(
                    ((first_name, first_sequence),), records):
                if len(sequence) != alignment_length:
                    raise ValueError(
                        f"{name} has {len(sequence)} sites; "
                        f"expected {alignment_length}")
                output.write(name)
                output.write("\n")
                output.write("".join(sequence[index] for index in indices))
                output.write("\n")
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sites", required=True, type=int)
    args = parser.parse_args()
    sample_alignment(args.input, args.output, args.sites)


if __name__ == "__main__":
    main()
