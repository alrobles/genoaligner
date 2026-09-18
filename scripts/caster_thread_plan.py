#!/usr/bin/env python3
"""Plan meaningful CASTER CPU thread counts for a FASTA alignment."""

import argparse
import math


def fasta_sites(path):
    sequence_lengths = []
    current_length = 0
    saw_header = False
    with open(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if saw_header:
                    if current_length == 0:
                        raise ValueError("benchmark FASTA has an empty sequence")
                    sequence_lengths.append(current_length)
                saw_header = True
                current_length = 0
                continue
            if not saw_header:
                raise ValueError("benchmark input is not FASTA")
            current_length += len(line)
    if not saw_header or current_length == 0:
        raise ValueError("benchmark FASTA has no sequence")
    sequence_lengths.append(current_length)
    if len(set(sequence_lengths)) != 1:
        raise ValueError("benchmark FASTA sequences have unequal lengths")
    return sequence_lengths[0]


def effective_chunks(sites, chunk_size):
    if sites <= 0 or chunk_size <= 0:
        raise ValueError("sites and chunk size must be positive")
    return math.ceil(sites / chunk_size)


def default_threads(chunks, maximum):
    limit = min(chunks, maximum)
    threads = []
    value = 1
    while value <= limit:
        threads.append(value)
        value *= 2
    if threads[-1] != limit:
        threads.append(limit)
    return threads


def requested_threads(value, chunks, maximum):
    threads = []
    for token in value.replace(",", " ").split():
        try:
            thread = int(token)
        except ValueError as error:
            raise ValueError(f"invalid thread count: {token}") from error
        if thread < 1 or thread > maximum:
            raise ValueError(
                f"thread count {thread} is outside 1..{maximum}"
            )
        if thread > chunks:
            raise ValueError(
                f"thread count {thread} exceeds {chunks} scoring chunks; "
                "extra workers would be idle"
            )
        if thread not in threads:
            threads.append(thread)
    if not threads:
        raise ValueError("no thread counts requested")
    return threads


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--chunk", required=True, type=int)
    parser.add_argument("--max-threads", default=32, type=int)
    parser.add_argument("--threads")
    args = parser.parse_args()

    sites = fasta_sites(args.input)
    chunks = effective_chunks(sites, args.chunk)
    if args.max_threads < 1:
        parser.error("--max-threads must be positive")
    try:
        if args.threads:
            threads = requested_threads(
                args.threads,
                chunks,
                args.max_threads,
            )
        else:
            threads = default_threads(chunks, args.max_threads)
    except ValueError as error:
        parser.error(str(error))

    print(f"{sites}\t{chunks}\t{' '.join(map(str, threads))}")


if __name__ == "__main__":
    main()
