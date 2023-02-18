import os
import random
import re
import sys

# added
from collections import Counter
import copy

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    prob = {}
    n_pages = len(corpus.keys())
    links = corpus[page]
    n_links = len(links)

    for p in corpus.keys():
        if p in links:
            prob[p] = damping_factor/n_links + (1-damping_factor)/n_pages
        else:
            prob[p] = (1-damping_factor)/n_pages
    
    return prob


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pages = list(corpus.keys())
    n_pages = len(pages)
    p = pages[random.randint(0,n_pages-1)]
    prob = transition_model(corpus, p, damping_factor)
    outcomes = [p]

    for _ in range(n-1):
        p = random.choices(list(prob.keys()), weights=list(prob.values()), k=1)[0]
        prob = transition_model(corpus, p, damping_factor)
        outcomes.append(p)
    out_prob = {k: v/n for k, v in Counter(outcomes).items()}

    return out_prob


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    e = 0.001
    pages = list(corpus.keys())
    n_pages = len(pages)
    prob = {p:1/n_pages for p in pages} # initialized prob
    next_prob = {}
    link_to = {p:[] for p in pages} # invert of corpus; values will collect pages that link to the key

    for p in pages:
        for v in corpus[p]:
            link_to[v].append(p)

    while True:
        for p in pages:
            links = link_to[p]
            next_prob[p] = (1-damping_factor)/n_pages + damping_factor*sum([prob[i]/len(corpus[i]) for i in links])
        delta = [abs(n-p) for n, p in zip(next_prob.values(), prob.values())]
        prob = copy.deepcopy(next_prob)
        if sum([1 if d>e else 0 for d in delta]) == 0:
            break

    return prob


if __name__ == "__main__":
    main()
