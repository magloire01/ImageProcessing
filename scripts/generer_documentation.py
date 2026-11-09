"""Génère le document de conception de WaterVision, en français.

Le PDF explique le rôle de chaque partie du code et le fonctionnement de
l'interface. Il ne remplace pas les commentaires mathématiques des modules :
il les relie, dans l'ordre où les données traversent le logiciel.

Lancer depuis la racine du projet :

    python scripts/generer_documentation.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "WaterVision_Documentation.pdf"

PAGE_W = 8.27
PAGE_H = 11.69
LEFT = 0.08
RIGHT = 0.92
TOP = 0.93
BOTTOM = 0.06


class Manual:
    """Mise en page simple : le texte descend, une nouvelle page s'ouvre s'il manque de place."""

    def __init__(self, destination: Path):
        self.destination = destination
        self._pdf = PdfPages(destination)
        self.figure: Figure | None = None
        self.axis = None
        self.y = TOP
        self.page_number = 0

    def close(self) -> None:
        if self.figure is not None:
            self._pdf.savefig(self.figure)
            self.figure = None
        info = self._pdf.infodict()
        info["Title"] = "WaterVision — document de conception"
        info["Author"] = "WaterVision"
        info["Subject"] = "Explication du code et de l'interface"
        self._pdf.close()

    def _new_page(self, continued: bool = False) -> None:
        if self.figure is not None:
            self._pdf.savefig(self.figure)
        self.page_number += 1
        self.figure = Figure(figsize=(PAGE_W, PAGE_H), dpi=120, facecolor="white")
        self.axis = self.figure.add_axes((0, 0, 1, 1))
        self.axis.set_xlim(0, 1)
        self.axis.set_ylim(0, 1)
        self.axis.axis("off")
        self.y = TOP
        self.axis.plot([LEFT, RIGHT], [0.955, 0.955], color="#0B3A6A", linewidth=1.2, transform=self.axis.transAxes)
        self.axis.text(LEFT, 0.968, "WaterVision  ·  document de conception", color="#0B3A6A", fontsize=8, fontfamily="DejaVu Sans")
        if continued:
            self.axis.text(RIGHT, 0.968, "suite", color="#5C6B7A", fontsize=8, ha="right", fontfamily="DejaVu Sans")
        self.axis.text(0.5, 0.028, str(self.page_number), ha="center", va="center", color="#5C6B7A", fontsize=9, fontfamily="DejaVu Sans")

    def _need(self, height: float) -> None:
        if self.figure is None or self.y - height < BOTTOM:
            self._new_page(continued=self.page_number > 0)

    def _write(self, text: str, size: float, color: str, weight: str, family: str, step: float, indent: float = 0.0) -> None:
        self._need(step + 0.004)
        self.axis.text(
            LEFT + indent,
            self.y,
            text,
            fontsize=size,
            color=color,
            fontweight=weight,
            fontfamily=family,
            va="top",
            ha="left",
        )
        self.y -= step

    def h1(self, text: str) -> None:
        self._need(0.07)
        self.y -= 0.012
        self._write(text, 16, "#0B3A6A", "bold", "DejaVu Sans", 0.040)

    def h2(self, text: str) -> None:
        self._need(0.055)
        self.y -= 0.008
        self._write(text, 13, "#14508A", "bold", "DejaVu Sans", 0.032)

    def paragraph(self, text: str) -> None:
        for line in textwrap.wrap(text, width=86, break_long_words=False):
            self._write(line, 10.5, "#1C2833", "normal", "DejaVu Sans", 0.020)

    def bullet(self, text: str) -> None:
        lines = textwrap.wrap(text, width=82, break_long_words=False)
        if not lines:
            return
        self._write("•  " + lines[0], 10.5, "#1C2833", "normal", "DejaVu Sans", 0.020, indent=0.01)
        for line in lines[1:]:
            self._write(line, 10.5, "#1C2833", "normal", "DejaVu Sans", 0.020, indent=0.038)

    def formula(self, text: str) -> None:
        self._need(0.034)
        self.y -= 0.004
        self.axis.add_patch(
            FancyBboxPatch(
                (LEFT, self.y - 0.026),
                RIGHT - LEFT,
                0.030,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor="#F4F7FB",
                edgecolor="#D5E2F0",
                linewidth=0.6,
                transform=self.axis.transAxes,
                mutation_aspect=0.6,
            )
        )
        self.axis.text(0.5, self.y - 0.011, text, ha="center", va="center", fontsize=10.5, color="#0B3A6A", fontfamily="DejaVu Sans")
        self.y -= 0.038

    def code(self, text: str) -> None:
        lines = text.strip("\n").splitlines()
        height = 0.018 * len(lines) + 0.016
        self._need(height + 0.01)
        self.y -= 0.004
        self.axis.add_patch(
            FancyBboxPatch(
                (LEFT, self.y - height + 0.006),
                RIGHT - LEFT,
                height,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                facecolor="#F7F5F2",
                edgecolor="#E4DDD4",
                linewidth=0.6,
                transform=self.axis.transAxes,
            )
        )
        cursor = self.y - 0.006
        for line in lines:
            self.axis.text(LEFT + 0.012, cursor, line, fontsize=8.5, color="#243040", fontfamily="DejaVu Sans Mono", va="top")
            cursor -= 0.018
        self.y -= height + 0.008

    def gap(self, amount: float = 0.012) -> None:
        self.y -= amount


def cover(doc: Manual) -> None:
    doc._new_page()
    doc.axis.add_patch(
        FancyBboxPatch(
            (0.08, 0.58),
            0.84,
            0.28,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor="#0B1220",
            edgecolor="#0B1220",
            transform=doc.axis.transAxes,
        )
    )
    doc.axis.text(0.12, 0.78, "WATERVISION", color="#22D3EE", fontsize=28, fontweight="bold", fontfamily="DejaVu Sans", va="center")
    doc.axis.text(0.12, 0.70, "Détection et estimation des surfaces d'eau", color="white", fontsize=13, fontfamily="DejaVu Sans", va="center")
    doc.axis.text(0.12, 0.64, "Document de conception", color="#8BA3C7", fontsize=12, fontfamily="DejaVu Sans", va="center")
    doc.paragraph("")
    doc.y = 0.52
    doc.h2("À quoi sert ce document")
    doc.paragraph(
        "Ce document décrit comment le logiciel WaterVision est construit. Il suit le chemin d'une image, depuis le fichier ouvert par l'utilisateur jusqu'au masque, au pourcentage d'eau et au rapport exporté. Chaque étape indique le fichier Python qui la réalise et l'idée mathématique qui la justifie."
    )
    doc.gap()
    doc.paragraph(
        "Le logiciel est un projet de traitement numérique des images. Les opérations sont écrites avec NumPy et SciPy. L'interface est une fenêtre de bureau CustomTkinter. Pillow lit et écrit les images. Matplotlib trace les graphiques et les rapports. OpenCV n'est pas utilisé, et aucun modèle entraîné ne prend de décision."
    )
    doc.gap()
    doc.h2("Comment lire la suite")
    doc.bullet("Les sections 1 à 4 présentent l'objectif, les dossiers, l'état partagé et le chemin des données.")
    doc.bullet("Les sections 5 à 13 détaillent les algorithmes, dans l'ordre du calcul.")
    doc.bullet("Les sections 14 à 18 décrivent l'interface, les fils d'exécution, la démonstration et la manière de lancer le logiciel.")
    doc.gap(0.02)
    doc.paragraph("Pour lancer l'application : python main.py. Pour relire ce document après une modification du générateur : python scripts/generer_documentation.py.")


def body(doc: Manual) -> None:
    doc._new_page()
    doc.h1("1. Objectif")
    doc.paragraph(
        "WaterVision détecte des étendues d'eau naturelles dans une photographie : lac, rivière, étang ou retenue. Il produit un masque binaire, compte les pixels d'eau, et n'annonce une surface en mètres carrés que si l'utilisateur fournit la taille réelle d'un pixel."
    )
    doc.paragraph(
        "L'eau naturelle n'est pas un bleu unique. Elle peut être verte, brune, grise, sombre ou réfléchissante. Le logiciel sépare donc la teinte de la luminosité, et il refuse de conclure à la présence d'eau à partir de la seule texture. Une plage de sable lisse a un spectre aussi pauvre en hautes fréquences qu'une eau calme."
    )

    doc.h1("2. Organisation du code")
    doc.paragraph("Le programme démarre par main.py, qui appelle watervision.app.main. Le paquet est séparé en trois responsabilités.")
    doc.bullet("watervision/processing contient les calculs. Aucun widget n'y est importé. Ces fonctions peuvent être testées sans ouvrir de fenêtre.")
    doc.bullet("watervision/state.py garde les images et les réglages d'une session. C'est le seul endroit où les grands tableaux NumPy sont conservés.")
    doc.bullet("watervision/ui et watervision/app.py construisent la fenêtre. Les pages lisent l'état et demandent à l'application de lancer un calcul.")
    doc.gap()
    doc.h2("Fichiers de calcul")
    doc.bullet("color.py : luminance BT.601 et conversion RGB vers HSV, écrites en NumPy.")
    doc.bullet("filters.py : médiane, gaussien, étirement, gamma, égalisation de la luminance.")
    doc.bullet("fourier.py : transformée 2D, filtres idéal et Butterworth, énergie haute fréquence locale.")
    doc.bullet("segmentation.py : masque HSV, masque de texture, score hybride.")
    doc.bullet("morphology.py : disque, érosion, dilatation, ouverture, fermeture, trous, petites régions.")
    doc.bullet("metrics.py : pourcentage, surface calibrée, IoU, précision, rappel, F1, exactitude.")
    doc.bullet("io_utils.py : lecture Pillow, masque de référence, image résultat fond noir et contour.")
    doc.bullet("demo.py : scène synthétique et son masque de vérité.")
    doc.bullet("report.py : tableau CSV et rapport PDF d'une analyse.")
    doc.gap()
    doc.h2("Fichiers d'interface")
    doc.bullet("app.py : fenêtre principale, navigation, fils d'exécution, dialogues de fichiers.")
    doc.bullet("ui/shell.py : barre latérale et en-tête, avec la barre de progression.")
    doc.bullet("ui/widgets.py : aperçus, curseurs, interrupteurs, cartes de chiffres, tableau.")
    doc.bullet("ui/charts.py : histogramme et diagrammes en barres sur fond sombre.")
    doc.bullet("ui/pages : un fichier par module, du tableau de bord jusqu'aux réglages.")
    doc.bullet("theme.py : couleurs bleu nuit et cyan, et les polices Segoe UI.")

    doc.h1("3. L'état d'une session")
    doc.paragraph(
        "La classe AppState, dans state.py, est le cahier de la session. Elle conserve l'image originale, l'image de travail après prétraitement, le spectre complexe centré, la carte d'énergie locale, les masques bruts, les masques nettoyés, les métriques, le masque de référence et le dictionnaire de paramètres."
    )
    doc.paragraph(
        "Un entier revision augmente chaque fois que l'image de travail change. Les calculs lancés en arrière-plan capturent cette valeur au départ. À leur retour, ils abandonnent le résultat si la révision a changé : l'utilisateur a chargé une autre photo pendant le calcul, et l'ancien masque ne doit pas recouvrir la nouvelle image."
    )
    doc.paragraph(
        "set_original et set_working appellent _invalidate_products. Le spectre, les masques et les scores sont effacés, parce qu'ils décriraient une image qui n'est plus affichée. Le masque de référence n'est effacé que lors d'un nouveau chargement, pas lors d'un simple prétraitement : la vérité terrain reste valable si les dimensions n'ont pas changé."
    )
    doc.paragraph(
        "Le dictionnaire params regroupe tous les curseurs. DEFAULT_PARAMS donne l'identité du prétraitement : médiane de taille 1, gaussien nul, contraste « none ». L'image n'est donc pas modifiée tant que l'utilisateur n'appuie pas sur Apply. Les poids hybrides par défaut sont 0,50 pour la couleur, 0,35 pour la texture et 0,15 pour l'homogénéité. Le seuil de décision est 0,56. La somme texture plus homogénéité vaut 0,50, donc elle reste sous le seuil : une zone lisse sans la bonne couleur n'est pas classée comme eau."
    )

    doc.h1("4. Chaîne de traitement")
    doc.paragraph(
        "Le schéma suivant est l'ordre réel des données. Le prétraitement est facultatif. La transformée de Fourier de l'image entière sert à visualiser et à filtrer. La carte locale sert à la segmentation. Les trois masques passent ensuite par la même morphologie, afin que la comparaison porte sur la décision et non sur le bruit résiduel."
    )
    draw_pipeline(doc)

    doc.h1("5. Acquisition de l'image")
    doc.paragraph(
        "io_utils.load_rgb ouvre le fichier avec Pillow. Les formats acceptés sont JPG, JPEG, PNG, TIFF et BMP. ImageOps.exif_transpose redresse les photographies de téléphone selon l'orientation EXIF, sinon le lac serait analysé couché. Un fichier TIFF de plusieurs pages ne fournit que la première, et le logiciel l'écrit dans le statut."
    )
    doc.paragraph(
        "Une image avec transparence est composée sur du noir : un pixel transparent n'est pas une observation d'eau. Les autres modes sont convertis en RGB 8 bits. Pillow ramène aussi les TIFF 16 bits sur 0–255. Le tableau obtenu est un ndarray uint8 de forme (hauteur, largeur, 3), rendu contigu pour les calculs suivants. Une image plus petite que 8×8 pixels est refusée."
    )
    doc.paragraph(
        "Dans l'interface, AcquisitionPage affiche ce tableau dans un canevas Tk. Le zoom est un seul facteur appliqué aux deux axes : à 100 %, l'image tient dans la zone ; au-delà, des barres de défilement apparaissent et le glisser déplace la vue. Le bouton Fit ramène le facteur à 1. Le panneau de droite indique la largeur, la hauteur, les trois canaux 8 bits et la taille du fichier."
    )

    doc.h1("6. Couleur : luminance et HSV")
    doc.h2("Luminance")
    doc.paragraph(
        "La transformée de Fourier d'un manuel est définie sur un champ scalaire. color.rgb_to_gray réduit donc les trois canaux avant tout spectre. Les poids sont ceux de la recommandation UIT-R BT.601, classiques dans les cours de traitement d'images. Ils sont positifs et leur somme vaut 1, donc une image déjà dans [0, 1] y reste."
    )
    doc.formula("Y = 0,299 R + 0,587 G + 0,114 B")
    doc.paragraph(
        "Le vert domine parce que l'œil y est plus sensible. Un gris arithmétique (R+G+B)/3 sous-estimerait cette sensibilité. La fonction as_float01 divise un uint8 par 255 et laisse un tableau déjà dans [0, 1] tel quel, pour qu'une formule ne change pas de sens entre les deux échelles."
    )
    doc.h2("HSV")
    doc.paragraph(
        "rgb_to_hsv applique la définition du cône hexagonal. Sur des canaux R, G, B ramenés à [0, 1], on calcule le maximum Cmax, le minimum Cmin et l'écart delta. La valeur V est Cmax. La saturation S est nulle si Cmax est nul, sinon delta/Cmax. La teinte H, en degrés, dépend du canal qui porte le maximum."
    )
    doc.formula("H = 60 × (((G − B) / delta) mod 6)    si le maximum est R")
    doc.formula("H = 60 × (((B − R) / delta) + 2)      si le maximum est G")
    doc.formula("H = 60 × (((R − G) / delta) + 4)      si le maximum est B")
    doc.paragraph(
        "Si delta vaut 0, le pixel est gris et la teinte est indéterminée. Le code range 0. Ce choix est sans effet, car la saturation est alors nulle et les tests de teinte ne doivent pas décider seuls. Le modulo ramène le secteur rouge, qui peut être négatif près du magenta, dans l'intervalle [0, 360)."
    )
    doc.paragraph(
        "HSV est utilisé parce qu'une même eau, plus profonde ou plus éclairée, garde à peu près la même teinte alors que ses valeurs RGB changent toutes les trois. Un seuil sur le canal bleu aurait rejeté l'eau verte, l'eau turbide et l'eau sombre."
    )

    doc.h1("7. Prétraitement")
    doc.paragraph(
        "filters.preprocess part toujours de l'image originale, pas de l'image déjà filtrée. Les curseurs décrivent donc un réglage absolu. L'aperçu est calculé dans un fil, et l'image de travail n'est remplacée que lorsque l'utilisateur appuie sur Apply to workspace. Les modules Fourier et Segmentation lisent cette image engagée."
    )
    doc.paragraph("L'ordre est fixe, et il est justifié par le type de défaut que chaque filtre sait enlever.")
    doc.bullet("La médiane passe en premier. Dans une fenêtre, elle remplace le pixel par la valeur centrale de la liste triée. Une impulsion isolée ne peut pas tirer cette valeur, contrairement à une moyenne. Le bruit poivre et sel est donc retiré avant qu'un filtre linéaire ne l'étale en halo. La taille du noyau est impaire. La valeur 1 désactive le filtre. SciPy calcule la médiane avec size=(k, k, 1) : les trois couleurs ne sont pas mélangées. Le bord est réfléchi, pour ne pas inventer un cadre sombre qui ressemblerait à une rive.")
    doc.bullet("Le gaussien vient ensuite. Son noyau continu est G(x, y) = 1/(2 π σ²) exp(−(x²+y²)/(2 σ²)). SciPy l'échantillonne, le normalise, et l'applique avec un écart-type nul sur l'axe des couleurs. Une image constante est inchangée. Une sinusoïde de fréquence f est multipliée par exp(−2 π² σ² f²) : plus f est grand, plus l'atténuation est forte. σ est en pixels. En dessous de 0,05, l'étape est sautée.")
    doc.bullet("Le contraste est en dernier, pour ne pas amplifier le bruit que les deux filtres viennent de réduire. Quatre modes existent.")
    doc.paragraph(
        "Le mode stretch étire chaque canal entre deux percentiles, par défaut 1 et 99. Quelques pixels saturés ne fixent donc pas l'échelle. C'est une application affine : elle ne crée pas de structure, elle consacre la dynamique d'affichage aux intensités réellement présentes."
    )
    doc.formula("C' = clip( (C − p_bas) / (p_haut − p_bas) , 0 , 1 )")
    doc.paragraph(
        "Le mode gamma applique C' = C^γ sur [0, 1]. γ = 1 ne change rien. γ < 1 relève les ombres, car la dérivée est grande près de zéro. Les extrémités 0 et 1 restent fixes, et le même γ est utilisé sur les trois canaux pour préserver la teinte."
    )
    doc.paragraph(
        "Le mode equalize calcule l'histogramme de la luminance sur 256 niveaux, puis sa fonction de répartition. La luminance égalisée est cdf(Y). Appliquer cette courbe séparément à R, G et B déplacerait la teinte. Le code multiplie plutôt le triplet RGB par le rapport Y'/Y, avec un rapport de 1 là où Y est nul. La direction chromatique reste, seule la clarté bouge."
    )

    doc.h1("8. Transformée de Fourier de l'image")
    doc.paragraph(
        "fourier.fft2_shifted calcule la transformée discrète bidimensionnelle de la luminance, puis fftshift place la fréquence nulle au centre. Pour une image de M lignes et N colonnes, NumPy évalue"
    )
    doc.formula("F(u, v) = Σ_x Σ_y  f(x, y) exp( −j 2π (u x / M + v y / N) )")
    doc.paragraph(
        "La transformée inverse, ifft2, contient le facteur 1/(M N). Après fftshift, l'indice (M//2, N//2) est le terme continu F(0, 0), égal à la somme des pixels. La distance au centre est la fréquence spatiale : le milieu décrit les variations lentes, les coins décrivent la plus fine oscillation que la grille de pixels peut porter."
    )
    doc.paragraph(
        "Le spectre est stocké en complex64. Le garder en mémoire permet de changer le filtre sans recalculer la transformée. reconstruct annule le décalage, applique l'inverse, et ne conserve que la partie réelle. La partie imaginaire est un résidu numérique, car un filtre réel et une image réelle produisent un spectre à symétrie hermitienne."
    )
    doc.h2("Visualisation")
    doc.paragraph(
        "L'affichage utilise log(1 + |F|). La valeur absolue est la racine de la somme des carrés des parties réelle et imaginaire. Le terme continu est énorme, puisqu'il additionne tous les pixels. Le logarithme comprime cette dynamique, et le 1 intérieur évite le logarithme de zéro. Un étirement entre les percentiles 1 et 99,8 sert uniquement à l'œil : le filtre, lui, multiplie les coefficients complexes, jamais l'image logarithmique."
    )
    doc.h2("Filtres")
    doc.paragraph(
        "La distance de chaque case au centre est D(u, v) = sqrt( (u−uc)² + (v−vc)² ). Le curseur de coupure est une fraction de la distance jusqu'au coin du spectre. Une fraction de 1 couvre donc toutes les fréquences, quelle que soit la taille de l'image."
    )
    doc.paragraph("Le passe-bas de Butterworth d'ordre n vaut 1 à l'origine, donc il conserve la moyenne. Il vaut 1/2 lorsque D = D0, pour tout ordre. Un ordre élevé rapproche la transition d'un disque idéal.")
    doc.formula("H_pb(u, v) = 1 / ( 1 + (D(u, v) / D0) ^ (2 n) )")
    doc.paragraph(
        "Le passe-haut classique n'est pas exactement 1 − H_pb, mais il a le même rayon à mi-puissance. Au centre, D0/D n'est pas défini : le code écrit 0 explicitement, ce qui retire la moyenne."
    )
    doc.formula("H_ph(u, v) = 1 / ( 1 + (D0 / D) ^ (2 n) )    pour D > 0")
    doc.paragraph(
        "Le filtre idéal est un disque dur : 1 à l'intérieur du rayon, 0 dehors. Sa transformée inverse est un jinc, et la convolution avec ce noyau produit des ondulations de Gibbs le long des contours francs. Il est proposé parce que c'est la définition vue en premier dans un cours. Le Butterworth est le réglage par défaut."
    )
    doc.paragraph(
        "Multiplier F par H réel ne change pas la phase, donc le filtre atténue des fréquences sans décaler les contours. Un passe-bas reconstruit une luminance lissée, affichée sur l'échelle d'origine. Un passe-haut reconstruit un résidu de moyenne nulle, avec des valeurs négatives. Pour le montrer, le résidu est étiré sur [0, 1], et le cartouche de l'image le dit : cet étirement n'est pas l'image filtrée elle-même."
    )

    doc.h1("9. Énergie haute fréquence locale")
    doc.paragraph(
        "La question n'est plus « comment filtrer toute la photo », mais « dans ce petit voisinage, quelle part de l'énergie est rapide ». local_high_frequency_ratio découpe la luminance en carrés. Le pas est choisi pour qu'il y ait au plus max_frequency_windows fenêtres le long de chaque axe, par défaut 56. Une grande photo ne demande donc pas une transformée par pixel."
    )
    doc.paragraph(
        "Chaque carré est centré, c'est-à-dire privé de sa moyenne, puis multiplié par une fenêtre de Hann séparable. La fenêtre w(n) = 0,5 − 0,5 cos(2π n / (N−1)) vaut zéro aux bords. Sans elle, la transformée traiterait le carré comme une période d'un signal périodique, et le saut aux bords fuirait dans les hautes fréquences. C'est la fuite spectrale. La même fenêtre est appliquée partout, donc les rapports restent comparables."
    )
    doc.paragraph(
        "On calcule ensuite le spectre du carré et sa densité |P|². Le théorème de Parseval dit que la somme de |P|² est proportionnelle à l'énergie du carré fenêtré. Le rapport de deux sommes est donc un rapport d'énergie. Un disque de rayon cutoff_ratio × (côté / 2) sépare les basses fréquences du reste."
    )
    doc.formula("E_haut / E_totale = Σ_{D > D0} |P|²   /   Σ |P|²")
    doc.paragraph(
        "Un carré parfaitement plat a une énergie totale presque nulle après retrait de la moyenne. Son rapport est défini comme 0. Un feuillage ou un bruit a un rapport élevé. La carte grossière est ensuite agrandie par interpolation bilinéaire jusqu'à la taille de l'image. Cet agrandissement ne crée pas de nouvelles mesures : il permet seulement aux règles pixel à pixel de lire une valeur."
    )
    doc.paragraph(
        "Cette carte décrit une texture. Elle ne détecte pas l'eau. Le sable, un toit, une route et une ombre plate produisent aussi un rapport faible. L'interface l'écrit sur la page Fourier et sur l'onglet Frequency de la segmentation."
    )

    doc.h1("10. Trois segmentations")
    doc.paragraph(
        "segmentation.segment calcule les trois masques sur la même image de travail, puis leur applique la même morphologie. Les masques bruts et les masques nettoyés sont tous conservés."
    )
    doc.h2("Méthode 1 — seuils HSV")
    doc.paragraph(
        "hsv_hard_mask est un test booléen. Un pixel est retenu s'il est dans la boîte primaire, ou dans la seconde boîte si elle est activée, ou dans la règle d'eau sombre si elle est activée. La boîte primaire par défaut couvre les teintes de 90° à 255°, donc du vert jusqu'au bleu, avec une saturation d'au moins 0,12 et une valeur entre 0,08 et 0,92. Le plafond de valeur écarte les nuages et la mousse très clairs."
    )
    doc.paragraph(
        "La teinte est un angle. hue_in_range sait traverser 0° : si le minimum est supérieur au maximum, l'arc passe par le rouge, et un pixel est dedans s'il est au-dessus du minimum ou en dessous du maximum. La seconde bande, activée par défaut, est un ocre étroit, de 8° à 42°, avec une valeur maximale de 0,55. Elle vise une eau chargée de sédiments, pas un sable clair dont la valeur dépasse ce plafond."
    )
    doc.paragraph(
        "La règle d'eau sombre accepte un pixel seulement s'il est à la fois peu lumineux et peu saturé. La teinte d'un gris est instable. Un végétal sombre mais saturé échoue au test de saturation. Un trottoir gris clair échoue au test de valeur."
    )
    doc.h2("Méthode 2 — candidat fréquentiel")
    doc.paragraph(
        "Le masque vaut vrai là où le rapport d'énergie haute fréquence est strictement inférieur au seuil de texture, 0,42 par défaut. Le nom du masque dans l'interface est « Fourier candidate ». C'est volontaire : le résultat montre les régions lisses. Il ne constitue pas, à lui seul, une détection d'eau."
    )
    doc.h2("Méthode 3 — score hybride")
    doc.paragraph(
        "C'est la méthode prévue pour la surface annoncée. Chaque pixel reçoit un score entre 0 et 1, combinaison convexe de trois termes. Les poids affichés sont renormalisés pour sommer à 1. S'ils sont tous nuls, le code retombe sur la couleur seule, afin que le score reste défini."
    )
    doc.formula("S = w_c C + w_t T + w_h H     avec w_c + w_t + w_h = 1")
    doc.paragraph(
        "C est une appartenance douce aux mêmes bandes HSV. Hors d'un intervalle, la teinte décroît comme une gaussienne dont l'écart est hue_softness, 16° par défaut. La saturation et la valeur décroissent de la même façon, avec un écart de 0,06. Les trois portes se multiplient : C = C_H × C_S × C_V. Un sable clair peut frôler une teinte brune ; si sa valeur est loin de l'intervalle, le produit tombe et le terme de texture ne suffit plus à franchir le seuil. La règle sombre est ajoutée par un maximum, pas par ce produit, parce qu'un gris n'a pas de teinte exploitable."
    )
    doc.paragraph(
        "T est la version souple du seuil de la méthode 2. C'est une logistique centrée sur ce seuil. En dessous, le pixel est lisse et T approche 1. Au-dessus, T approche 0. La pente texture_steepness vaut 14 par défaut. L'argument de l'exponentielle est borné pour éviter un dépassement numérique."
    )
    doc.formula("T(E) = 1 / ( 1 + exp( k (E − seuil) ) )")
    doc.paragraph(
        "H mesure l'homogénéité spatiale, qui n'est pas le même énoncé que le rapport de Fourier. La variance locale dans une boîte est E[X²] − (E[X])², les deux espérances étant des filtres uniformes SciPy. L'écart-type est comparé à une référence, 0,08 sur l'échelle [0, 1]."
    )
    doc.formula("H = clip( 1 − σ / σ_ref , 0 , 1 )")
    doc.paragraph(
        "Un dégradé lent au bord d'un lac a de la variance mais presque pas de haute fréquence. Des rides ou du feuillage ont les deux. Le terme H est facultatif. Le pixel est eau lorsque S est au moins égal au seuil de décision. Avec les poids par défaut, la couleur doit participer : 0,35 + 0,15 = 0,50, et 0,50 est inférieur à 0,56."
    )

    doc.h1("11. Morphologie")
    doc.paragraph(
        "morphology.py construit le disque avec NumPy : un pixel à l'offset (y, x) appartient à l'élément structurant lorsque y² + x² ≤ r². Le rayon 0 donne un seul pixel, et toutes les opérations deviennent l'identité. SciPy réalise l'érosion et la dilatation binaires. L'ouverture et la fermeture sont composées dans ce fichier, pour que l'ordre reste visible."
    )
    doc.formula("Érosion : A ⊖ B = { z | B_z est inclus dans A }")
    doc.formula("Dilatation : A ⊕ B = { z | A rencontre B_z }")
    doc.formula("Ouverture : (A ⊖ B) ⊕ B      Fermeture : (A ⊕ B) ⊖ B")
    doc.paragraph(
        "L'extérieur de l'image est traité comme du fond. Les objets rétrécissent, et un objet qui touche le bord n'est pas artificiellement conservé. L'ouverture passe avant la fermeture. Elle efface les faux positifs plus fins que le disque avant que la fermeture puisse les souder à un lac. La fermeture bouche ensuite les cassures dues aux reflets. Le rayon par défaut est 2."
    )
    doc.paragraph(
        "Le remplissage des trous utilise binary_fill_holes. Le complément est inondé depuis le bord. Tout pixel de fond que l'inondation n'atteint pas est entouré d'eau et rejoint le masque. Une baie qui touche le bord de la photo n'est pas un trou : une embouchure ou un lac coupé par le cadre reste ouvert."
    )
    doc.paragraph(
        "Les composantes sont étiquetées en connexité 8, avec un élément 3×3 de uns. Un fil d'eau en diagonale forme donc une seule région, ce qui correspond à une rivière qui traverse la grille. Les composantes dont le nombre de pixels est strictement inférieur à min_region_pixels, 80 par défaut, sont supprimées en dernier. Ce seuil est un nettoyage en pixels. Ce n'est pas une surface au sol."
    )

    doc.h1("12. Surface, calibration et scores")
    doc.h2("Compte des pixels")
    doc.paragraph(
        "Tous les pixels de l'image de travail sont valides. Le logiciel n'invente pas de masque de données manquantes. Un cadre noir ou un cartouche fait partie du dénominateur."
    )
    doc.formula("pourcentage = (pixels d'eau / pixels de l'image) × 100")
    doc.paragraph(
        "La surface physique est un autre énoncé. Elle exige une distance au sol : la largeur et la hauteur, en mètres, représentées par un pixel."
    )
    doc.formula("surface_m² = pixels d'eau × largeur_pixel_m × hauteur_pixel_m")
    doc.paragraph(
        "Si la calibration est éteinte, ou si une des deux longueurs n'est pas finie et strictement positive, la surface en mètres carrés reste indéfinie. L'interface affiche alors une phrase explicite : le résultat est un pourcentage de pixels, pas une aire au sol. Lorsque la calibration est active, la note rappelle que l'échelle est supposée constante et qu'aucune pente de terrain n'est corrigée. L'hectare affiché est simplement le nombre de mètres carrés divisé par 10 000."
    )
    doc.paragraph(
        "Le nombre de régions et la taille de la plus grande viennent de l'étiquetage 8-connexe du masque nettoyé. La méthode active, hybride par défaut, alimente les cartes du tableau de bord. Le diagramme en barres compare les trois pourcentages."
    )
    doc.h2("Vérité terrain")
    doc.paragraph(
        "Un masque de référence est une image dont les pixels au-dessus de 127, ou au-dessus de 0,5 s'il est flottant, sont l'eau. S'il n'a pas la taille de l'image de travail, resize_nearest le rééchantillonne au plus proche voisin. Une interpolation linéaire inventerait des gris sans signification pour une étiquette. L'interface et le rapport disent que ce redimensionnement a eu lieu."
    )
    doc.paragraph("Pour une prédiction P et une référence G, les quatre effectifs sont les cardinaux des intersections.")
    doc.formula("TP = |P ∩ G|    FP = |P ∩ complément de G|")
    doc.formula("FN = |complément de P ∩ G|    TN = |complément de P ∩ complément de G|")
    doc.formula("précision = TP / (TP + FP)     rappel = TP / (TP + FN)")
    doc.formula("F1 = 2 PR / (P + R)     IoU = TP / (TP + FP + FN)")
    doc.formula("exactitude = (TP + TN) / N")
    doc.paragraph(
        "Le F1 d'une seule classe binaire est le coefficient de Dice. Si un dénominateur est nul parce que les deux masques sont vides, le score vaut 1 : il n'y avait pas d'eau et aucune eau n'a été prédite. L'exactitude est affichée parce qu'elle est demandée, avec un avertissement : quand l'eau occupe peu de la photo, les vrais négatifs la maintiennent haute même si le rivage est mauvais. L'IoU et le F1 décrivent la région d'eau."
    )
    doc.paragraph(
        "L'image d'accord colorie ces quatre ensembles : cyan pour les vrais positifs, ambre pour les faux positifs, violet pour les faux négatifs, et le fond sombre pour les vrais négatifs."
    )

    doc.h1("13. Image résultat : fond noir et contour")
    doc.paragraph(
        "overlay_mask, dans io_utils.py, construit l'image montrée à droite du tableau de bord et enregistrée comme image annotée. Tout pixel hors du masque devient noir. Les pixels d'eau gardent leur couleur d'origine, donc la surface reste visible. Le contour est le gradient morphologique, une bande de deux pixels située à l'intérieur de l'eau."
    )
    doc.formula("contour = A  privé de  (A érodé deux fois)")
    doc.paragraph(
        "Cette bande est peinte en blanc, sur l'eau. Le paramètre alpha de la fonction n'est plus utilisé : le résultat est une séparation franche, pas un voile translucide posé sur toute la photographie."
    )

    doc.h1("14. Interface")
    doc.h2("Fenêtre")
    doc.paragraph(
        "WaterVisionApp, dans app.py, est une fenêtre CustomTkinter. Le thème est forcé en sombre avant la création des widgets. La grille a deux colonnes : la barre latérale, large d'environ 230 pixels, et la zone principale. L'en-tête porte le nom du module, une phrase d'explication, le nom du fichier et le statut. Une barre de progression déterminée reste au repos ; elle passe en mode indéterminé seulement pendant un calcul."
    )
    doc.paragraph(
        "Les neuf pages sont créées une fois et empilées dans la même cellule. navigate affiche la page demandée et retire les autres avec grid_remove. Elle resynchronise les curseurs depuis params, puis rafraîchit les aperçus. Revenir sur Fourier ne relance la transformée que si la révision de l'image a changé."
    )
    doc.h2("Pages")
    doc.bullet("Dashboard. Cinq cartes : pourcentage, pixels d'eau, nombre de régions, plus grande région, durée de la segmentation. À gauche, l'image originale. À droite, l'eau sur fond noir. Le bandeau sous les cartes est ambre tant qu'aucune taille de pixel n'est fournie, et vert lorsqu'une surface en mètres carrés existe.")
    doc.bullet("Image Acquisition. Chargement, scène de démonstration, zoom, ajustement à la vue, remise à zéro de tout l'espace de travail.")
    doc.bullet("Preprocessing. Deux aperçus et l'histogramme de la luminance de l'aperçu. Les curseurs mettent à jour un calcul différé de 280 ms. Le texte d'avertissement rappelle que l'aperçu n'est pas encore l'image de travail.")
    doc.bullet("Fourier Analysis. Quatre images : photo de travail, spectre logarithmique, image filtrée, carte d'énergie locale. Les curseurs de filtre relancent seulement le produit H×F et la transformée inverse, grâce au spectre mis en cache.")
    doc.bullet("Segmentation. Grille 2×2 : image, masque HSV, candidat fréquentiel, masque hybride. Les réglages sont dans quatre onglets. Le sélecteur HSV / Fourier / Hybrid choisit le masque utilisé pour la surface, le tableau de bord et l'export par défaut. Run segmentation lance les trois méthodes. Reset parameters restaure les réglages de segmentation et retire les masques devenus incohérents.")
    doc.bullet("Area Estimation. Les cartes résument la méthode active. Le tableau et le diagramme comparent les trois. Apply pixel size valide deux nombres strictement positifs, ou éteint la calibration.")
    doc.bullet("Evaluation. Charge un masque de référence, montre l'accord coloré, le tableau des scores et le diagramme groupé IoU, précision, rappel et F1.")
    doc.bullet("Export. Masque PNG 0/255, image annotée, spectre, CSV et PDF d'analyse. Le CSV et le PDF acceptent une image sans segmentation : les sections absentes sont marquées comme non calculées, elles ne sont pas inventées.")
    doc.bullet("Settings. Nombre maximal de fenêtres fréquentielles, palette du spectre, texte de rappel sur ce que chaque mesure affirme, et restauration de tous les paramètres.")
    doc.h2("Contrôles")
    doc.paragraph(
        "SliderRow affiche le nom, la valeur formatée et le curseur. Pendant une remise à jour programmée, un drapeau silencieux empêche le curseur de relancer le calcul qui est en train de l'initialiser. SwitchRow fait de même pour les interrupteurs. Les info-bulles sont de petites fenêtres Tk sans bordure : CustomTkinter n'en fournit pas. ImageCard redimensionne l'aperçu avec Pillow en conservant le rapport largeur/hauteur, puis le donne à un CTkImage de cette taille exacte. Le widget n'étire donc pas la photo."
    )
    doc.paragraph(
        "Un défaut de Tkinter apparaît si l'on détruit le CTkImage pendant que l'étiquette l'affiche encore : le message « image pyimageN doesn't exist ». ImageCard conserve l'objet précédent jusqu'à ce que l'étiquette ait reçu la nouvelle image. Une pastille de 1×1 pixel, jamais détruite, remplace l'aperçu lorsqu'il n'y a rien à montrer."
    )

    doc.h1("15. Calculs et interface : qui appelle qui")
    doc.paragraph(
        "Une page ne lance pas elle-même un fil. Elle appelle une méthode de WaterVisionApp : open_image, load_demo, compute_spectrum, compute_local_energy, run_segmentation, set_calibration, ou un export. run_task refuse un second gros calcul tant que le drapeau busy est vrai, et une boîte de dialogue le dit. Le travail s'exécute dans un fil démon. Le résultat revient sur le fil de Tk par after(0, ...). Seul ce fil touche aux widgets."
    )
    doc.paragraph(
        "Le filtre de Fourier est une exception légère. Changer la coupure ne prend pas le drapeau busy. Un compteur de génération jette les résultats périmés si l'utilisateur a encore bougé le curseur. Le spectre complexe, lui, n'est pas recalculé."
    )
    doc.paragraph(
        "open_image ouvre le sélecteur avec la fenêtre WaterVision comme parent, et les motifs de fichiers sont séparés par des points-virgules, forme attendue par le dialogue Windows. Après un chargement réussi, l'application affiche tout de suite le tableau de bord, puis lance la segmentation de cette photo. Le nom de fichier dans l'en-tête devient celui du fichier choisi. La scène de démonstration suit le même retour au tableau de bord, pour montrer le lac sur fond noir plutôt que de laisser l'utilisateur sur une autre page."
    )
    doc.paragraph(
        "Les graphiques sont des objets matplotlib.figure.Figure, pas des figures pyplot. Le canevas TkAgg est dessiné sur le fil principal. Le PDF d'analyse, lui, est construit hors écran dans le fil de l'export, avec le backend PDF. Les deux ne partagent pas l'état global de pyplot."
    )

    doc.h1("16. Scène de démonstration")
    doc.paragraph(
        "demo.make_demo_scene fabrique une image 480×640 et son masque de vérité, sans fichier extérieur. La graine aléatoire est fixe. Le sol est un jaune-vert texturé, volontairement hors de la bande de teinte primaire. Un banc de sable est lisse et beige : la méthode fréquentielle l'accepte, la teinte et le score hybride le refusent. Le lac et la rivière sont bleu-vert, avec un très faible bruit. Un petit étang gris, peu saturé, ne passe pas le seuil de saturation primaire ; c'est la règle d'eau sombre qui peut l'admettre."
    )
    doc.paragraph(
        "La vérité terrain est l'union géométrique du lac, de la rivière et de l'étang. Le sable en est exclu. Le test test_methods_separate_smooth_sand_from_water vérifie que l'hybride couvre le lac, que Fourier couvre le sable, et que l'hybride ne traite pas ce sable comme de l'eau. Il vérifie aussi que l'IoU hybride dépasse l'IoU fréquentiel sur cette scène."
    )

    doc.h1("17. Exports d'une analyse")
    doc.paragraph(
        "write_csv écrit trois blocs dans un fichier UTF-8 avec indicateur d'ordre des octets, pour qu'Excel sous Windows reconnaisse l'encodage : le tableau des surfaces, les scores s'ils existent, puis tous les paramètres. Une surface sans calibration est marquée « pixel-based estimate » et la colonne en mètres carrés reste vide."
    )
    doc.paragraph(
        "write_pdf assemble un rapport d'analyse, distinct du présent document. Les pages contiennent le rappel de la formule du pourcentage, l'image de travail, l'eau sur fond noir, les trois masques, le spectre s'il a été calculé, les scores s'il existe une référence, puis la liste des paramètres et le rôle de chaque méthode. Une section absente est une phrase, pas une figure inventée."
    )

    doc.h1("18. Lancer, vérifier, limites")
    doc.h2("Installation et lancement")
    doc.code(
        "python -m pip install -r requirements.txt\n"
        "python main.py\n"
        "python -m unittest discover -s tests -t ."
    )
    doc.paragraph(
        "Les dépendances sont NumPy, SciPy, Pillow, Matplotlib et CustomTkinter. Python 3.11 ou plus récent est requis. Les tests du dossier tests ne créent pas de fenêtre : ils contrôlent les teintes primaires, l'inversion de la transformée de Fourier, l'ouverture morphologique, le refus d'une calibration nulle, et le comportement de la scène synthétique."
    )
    doc.h2("Ce que le logiciel ne prétend pas faire")
    doc.bullet("Il ne géoréférence pas une image. Sans largeur et hauteur de pixel, il n'y a pas de mètres carrés.")
    doc.bullet("Il ne corrige pas le relief. Un pixel en pente ne représente pas la même aire au sol qu'un pixel plat.")
    doc.bullet("Il ne distingue pas, par la fréquence seule, l'eau d'une autre surface lisse.")
    doc.bullet("Une eau très ridée peut perdre le terme de texture. L'utilisateur peut alors baisser le seuil de décision ou augmenter le poids de la couleur. Une végétation vraiment verte et saturée peut, à l'inverse, entrer dans l'intervalle de teinte : le poids de texture sert à la défavoriser lorsqu'elle est rugueuse.")
    doc.bullet("Le prétraitement et la segmentation décrivent la photo telle qu'elle est cadrée. Un cadre noir compte dans le total des pixels.")
    doc.gap()
    doc.paragraph(
        "Les formules détaillées, avec les cas limites numériques, restent dans les docstrings des modules de watervision/processing. Ce document est le plan qui permet de les lire dans l'ordre."
    )


def draw_pipeline(doc: Manual) -> None:
    """Un schéma horizontal couperait les libellés. Le flux est vertical, une boîte par étape."""
    labels = [
        "Fichier  JPG, PNG, TIFF, BMP",
        "Pillow  →  tableau RGB uint8",
        "Prétraitement facultatif  (filters.py)",
        "Luminance et HSV  (color.py)",
        "Spectre centré et filtres  (fourier.py)",
        "Énergie locale des carrés",
        "HSV   ·   fréquence   ·   hybride",
        "Morphologie commune",
        "Pourcentage, scores, export",
    ]
    height = 0.034 * len(labels) + 0.02
    doc._need(height + 0.02)
    top = doc.y
    for index, label in enumerate(labels):
        y = top - index * 0.034
        doc.axis.add_patch(
            FancyBboxPatch(
                (0.18, y - 0.026),
                0.64,
                0.024,
                boxstyle="round,pad=0.003,rounding_size=0.004",
                facecolor="#0B1220" if index in (6, 8) else "#F4F7FB",
                edgecolor="#1E3A5F",
                linewidth=0.7,
                transform=doc.axis.transAxes,
            )
        )
        doc.axis.text(
            0.50,
            y - 0.014,
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color="white" if index in (6, 8) else "#0B3A6A",
            fontfamily="DejaVu Sans",
        )
        if index < len(labels) - 1:
            doc.axis.annotate(
                "",
                xy=(0.50, y - 0.030),
                xytext=(0.50, y - 0.026),
                arrowprops={"arrowstyle": "-", "color": "#8BA3C7", "lw": 0.8},
                xycoords=doc.axis.transAxes,
                textcoords=doc.axis.transAxes,
            )
    doc.y = top - len(labels) * 0.034 - 0.01


def main() -> None:
    doc = Manual(OUTPUT)
    cover(doc)
    body(doc)
    doc.close()
    print(OUTPUT)


if __name__ == "__main__":
    main()
