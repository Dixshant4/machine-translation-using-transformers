""""
This code is provided solely for the personal and private use of students
taking the CSC401H/2511H course at the University of Toronto. Copying for
purposes other than this use is expressly prohibited. All forms of
distribution of this code, including but not limited to public repositories on
GitHub, GitLab, Bitbucket, or any other online platform, whether as given or
with any changes, are expressly prohibited.

Authors: Sean Robertson, Jingcheng Niu, Zining Zhu, and Mohamed Abdall
Updated by: Arvie Frydenlund, Raeid Saqur and Jingcheng Niu

All of the files in this directory and all subdirectories are:
Copyright (c) 2024 University of Toronto
"""

"""
Calculate BLEU score for one reference and one hypothesis

You do not need to import anything more than what is here
"""

from math import exp  # exp(x) gives e^x
from collections.abc import Sequence


def grouper(seq: Sequence[str], n: int) -> list:
    """
    Extract all n-grams from a sequence

    An n-gram is a contiguous sub-sequence within `seq` of length `n`. This
    function extracts them (in order) from `seq`.

    Parameters
    ----------
    seq : sequence
        A sequence of words or token ids representing a transcription.
    n : int
        The size of sub-sequence to extract.

    Returns
    -------
    ngrams : list
    """
    ngrams = []
    for i in range(len(seq)):
        if n+i <= len(seq):
            n_gram = seq[i:n+i]
            ngrams.append(n_gram)
    return ngrams
    # assert False, "Fill me"


def n_gram_precision(
    reference: Sequence[str], candidate: Sequence[str], n: int
) -> float:
    """
    Calculate the precision for a given order of n-gram

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The order of n-gram precision to calculate

    Returns
    -------
    p_n : float
        The n-gram precision. In the case that the candidate has length 0,
        `p_n` is 0.
    """
    if len(candidate) == 0:
        return 0
    # can use count to figure out how many times ngrams from candidate occur in ngrams of sequence
    ngrams_reference = grouper(reference, n)
    ngrams_candidate = grouper(candidate, n)
    common = 0
    if len(ngrams_candidate) !=0:
        for elt in ngrams_candidate:
            for tgt in ngrams_reference:
                if elt == tgt:
                    common += 1
                    break
            # common += ngrams_reference.count(elt)
        total_ngrams = len(ngrams_candidate)
        
        return common/total_ngrams
    else:
        return 0

    # assert False, "Fill me"


def brevity_penalty(reference: Sequence[str], candidate: Sequence[str]) -> float:
    """
    Calculate the brevity penalty between a reference and candidate

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)

    Returns
    -------
    BP : float
        The brevity penalty. In the case that the candidate transcription is
        of 0 length, `BP` is 0.
    """
    if len(candidate) == 0:
        return 0
    if len(reference) != 0:
        BPi = len(candidate)/len(reference)
        if BPi < 1:
            return 0
        if BPi >=1:
            return exp(1-BPi)
    else:
        return 0

    # assert False, "Fill me"


def BLEU_score(reference: Sequence[str], candidate: Sequence[str], n) -> float:
    """
    Calculate the BLEU score.  Please scale the BLEU score by 100.0

    Parameters
    ----------
    reference : sequence
        The reference transcription. A sequence of words or token ids.
    candidate : sequence
        The candidate transcription. A sequence of words or token ids
        (whichever is used by `reference`)
    n : int
        The maximum order of n-gram precision to use in the calculations,
        inclusive. For example, ``n = 2`` implies both unigram and bigram
        precision will be accounted for, but not trigram.

    Returns
    -------
    bleu : float
        The BLEU score
    """
    n_gram_product = 1
    for i in range(n):
        n_gram_product *= n_gram_precision(reference, candidate, i)
    bleu = brevity_penalty(reference, candidate) * (n_gram_product)**(1/n)
    return bleu * 100

    # assert False, "Fill me"
