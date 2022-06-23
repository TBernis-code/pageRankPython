'''
Page Rank

Implémentation de l'algorithme de PageRank en python basé sur la théorie de la
connexion entre les sites web. L'algorithme est basé sur 
la distribution de probabilité d'un surfeur aléatoire.

'''

# Importing Libraries

import os
import random
import re
import sys
import matplotlib.pyplot as plt
import pandas as pan


# Global Constantes

AMORTISSEMENT = 0.85
SAMPLES = 100000

# Programme principal
def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py site")

    site = crawl(sys.argv[1])
    ranks = pagerank(site, AMORTISSEMENT, SAMPLES)
    print(f"Résultats du pageRank (n = {SAMPLES})")

    for page in sorted(ranks):
    	print(f"  {page}: {ranks[page]:.4f}")

    pan.DataFrame([ranks]).plot(kind='bar')
    plt.show()

# Fonction de crawl
def crawl(dir):

    """

    Parse le répertoire de pages HTML et check si il y a des les liens vers d'autres pages.
    Retourne un dictionnaire où chaque clé est une page, et les valeurs sont
    une liste de toutes les autres pages du site qui sont liées à la page.

    """
    
    pages = dict()

    # Extrait tout les liens de la page
    for filename in os.listdir(dir):
        
        if not filename.endswith(".html"):
            continue

        with open(os.path.join(dir, filename)) as f:
            
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Inclus seulement les liens vers des pages du site
    
    for filename in pages:
        
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(site, page, amortissement):

    """
    Retourne une probabilité de vers quelle page visité ensuite,
    en fonction de la page courante.

    """
    
    distribution = dict()

    
    if site[page]:

        for link in site:
            distribution[link] = (1-amortissement) / len(site)

            if link in site[page]:
                distribution[link] += amortissement / len(site[page])
    
    else:
        
        # Si la page n'a pas de liens sortants, on choisit aléatoirement parmi toutes les pages
        for link in site:
            distribution[link] = 1 / len(site)

    return distribution


def pagerank(site, amortissement, n):

    """
    
    Renvoie les valeurs du PageRank pour chaque page en échantillonnant `n = 10000` pages, 
    en commençant par une page au hasard.
    Retourne un dictionnaire où les clés sont les noms des pages, 
    et les valeurs sont leur PageRank estimée (une valeur entre 0 et 1). 
    La somme de tout les pageRank doit être égale a 1.

    
    """
    
    pagerank = dict()
    sample = None
    random.seed()

    for page in site:
        pagerank[page] = 0

    for step in range(n):
        if sample is None:
            
            # Premier sample parmi une page au hasard
            
            sample = random.choices(list(site.keys()), k=1)[0]
        
        else:
            
            # Next sample generated from the previous one based on its transition model
            #Prochain sample généré à partir du précédent en se basant sur son transition model
            
            model = transition_model(site, sample, amortissement)
            population, weights = zip(*model.items())
            sample = random.choices(population, weights=weights, k=1)[0]

        pagerank[sample] += 1

    
    #  On normalise les résultats
    
    for page in site:
        pagerank[page] /= n

    return pagerank


if __name__ == "__main__":
    main()