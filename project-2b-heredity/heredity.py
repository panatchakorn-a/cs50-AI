import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Conditional Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Conditional Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Conditional Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]

def get_num_gene(name, one_gene, two_genes):
    """
    Return the number of copies of mutated gene from person's name
    """
    if name in one_gene:
        return 1
    elif name in two_genes:
        return 2
    else:
        return 0

def get_from(num_gene, is_inherit):
    """
    Return the probability of (not) inheriting gene from certain number of mutated genes
    """
    if num_gene == 0:
        if is_inherit:
            return PROBS["mutation"]
        else:
            return 1-PROBS["mutation"]
    
    elif num_gene == 1:
        return 0.5

    elif num_gene == 2:
        if is_inherit:
            return 1-PROBS["mutation"]
        else:
            return PROBS["mutation"]

def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    prob = 1
    for name in people.keys():
        p = 1

        # gene
        num_gene = get_num_gene(name, one_gene, two_genes)
        if people[name]["mother"] is None:
            p *= PROBS["gene"][num_gene]
        else:
            num_gene_mother = get_num_gene(people[name]["mother"], one_gene, two_genes)
            num_gene_father = get_num_gene(people[name]["father"], one_gene, two_genes)
            if num_gene == 0:
                p *= get_from(num_gene_mother, False)*get_from(num_gene_father, False)
            elif num_gene == 1:
                p *= get_from(num_gene_mother, True)*get_from(num_gene_father, False) + get_from(num_gene_mother, False)*get_from(num_gene_father, True)
            else:
                p *= get_from(num_gene_mother, True)*get_from(num_gene_father, True)

        # conditional prob of trait given num_gene
        if name in have_trait:
            p *= PROBS["trait"][num_gene][True]
        else:
            p *= PROBS["trait"][num_gene][False]
        
        prob *= p
    
    return prob


def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    for name in probabilities.keys():
        # gene
        if name not in one_gene and name not in two_genes:
            probabilities[name]["gene"][0] += p
        elif name in one_gene:
            probabilities[name]["gene"][1] += p
        else:
            probabilities[name]["gene"][2] += p

        # trait
        if name in have_trait:
            probabilities[name]["trait"][True] += p
        else:
            probabilities[name]["trait"][False] += p

def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    for name in probabilities.keys():
        total = sum(probabilities[name]["gene"].values())
        for k, _ in probabilities[name]["gene"].items():
            probabilities[name]["gene"][k] = probabilities[name]["gene"][k] / total
        
        total = sum(probabilities[name]["trait"].values())
        for k, _ in probabilities[name]["trait"].items():
            probabilities[name]["trait"][k] = probabilities[name]["trait"][k] / total


if __name__ == "__main__":
    main()
