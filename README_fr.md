[English](./README.md) | Français

# Babl
Babl est une application conçue pour servir d'outil local de recommandation de livres.

Mon but initial était de classer les livres qui pouvaient me plaire dans les [opérations Masse Critique de Babelio](https://www.babelio.com/massecritique.php). Mais... il se peut que je me sois un peu emporté, et cet outil est devenu bien plus utile que je ne le croyais au départ, c'est pour cela que je le rend public !

![app screenshot](./static/images/screenshot.png)

🐛 [Bugs & problèmes](https://github.com/louisdalo/Babl/issues)  

## Fonctionnalités

- Une base de données de 50 000 livres prêts à l'emploi
- Un outil d'annotation permettant d'indiquer quels livres vous aimez (ou pas)
- Un outil de recommendation de livres présents dans [Open Library](https://openlibrary.org/)
- Une récupération semi-automatique des lives des [opérations Masse Critique de Babelio](https://www.babelio.com/massecritique.php)
- Un outil de recommendation pour les livres des [opérations Masse Critique de Babelio](https://www.babelio.com/massecritique.php)
- Un modèle de *Machine learning* qui tourne sur votre ordinateur, et facilement compréhensible
- Un mode sombre et lumineux (utilise évidemment le mode sombre par défaut)

## Comment démarrer
### Pré-requis

Pour utiliser ce projet, vous devez tout d'abord installer les logiciels suivants sur votre ordinateur:

- `git`: voir [ce tutoriel (en anglais)](https://github.com/git-guides/install-git)
- `uv`: voir [cet autre tutoriel (en anglais)](https://docs.astral.sh/uv/getting-started/installation/)


### Installation

Ouvrez un terminal sur votre ordinateur (tapez "powershell" dans le menu Démarrer pour windows). Dans le terminal, copiez et executez les commandes suivantes pour copier ce projet sur votre ordinateur:

```bash
git clone https://github.com/louisdalo/Babl.git
cd Babl
```
Ensuite, dans le même terminal, copiez et executez cette commande:

```bash
uv run streamlit run app.py
```

Votre navigateur devrait s'ouvrir directement sur la page de l'application, vous êtes prêt !

### Tips

Toutes vos données sont dans le sous-dossier `my_infos`, et ne quittent jamais votre ordinateur.

Si vous voulez editer/ajouter/retirer des informations, vous pouvez le faire dans ce sous-dossier. *Normalement* n'importe quel fichier/dossier à l'intérieur peut-être supprimé, ce qui remettra à zéro vos données concernant cet élément lors de votre prochaine session.

## Utilisation avancée

Si 50 000 livres est trop peu pour vous, alors rendez vous à [openlibrary.org/developers/dumps](https://openlibrary.org/developers/dumps). Téléchargez les fichiers nommés "all types" et "ratings" dumps (attention : ils font plusieurs gigaoctets). Ensuite, vous pouvez décider quels livres utiliser pour votre jeu de données avec le script `prepare_data.py`.

Executez cette commande dans votre terminal pour plus de détails:
```bash
uv python prepare.py -h
```

---

## FEUILLE DE ROUTE
Je liste ici les problèmes connus sur lesquels je travaille actuellement, ainsi que les fonctionnalités que j'aimerais ajouter.
- [ ] Créer une interface pour éditer les annotations.
- [ ] Traduction en français
    - [X] README.md
    - [ ] Interface utilisateur
    - [ ] métadonnées des livres
- [ ] Améliorer le regroupement des séries (peu fiable à cause des tags de la communauté):
    - [ ] Detecter manuellement les livres de "la guerre des clans".

## Contact

N'hésitez pas à me contacter en cas de problèmes ou pour faire part de vos suggestions.

- 🐛 **Bugs & problèmes :** Utilisez [GitHub Issues](https://github.com/louisdalo/Babl/issues)

Pour tout autre demande, n'hésitez pas à me contacter via [Bluesky](https://bsky.app/profile/louis-dalo.bsky.social)!

---

## Disclaimer

Une grande partie de ce code a été écrite à l'aide de [claude.ai](https://claude.ai), le LLM d'[Anthropic](https://www.anthropic.com/) (la version gratuite par contre, parce que j'ai pas d'argent a jeter par les fenêtres). J'ai systématiquement passé en revue tout le code généré par [claude.ai](https://claude.ai), je l'ai compris et l'ai modifié lorsque cela s'avérait nécessaire. Je pense que c'est une bonne pratique de travailler de cette manière, car **les LLM ne sont que des sources d'informations auxquelles on ne peut pas se fier aveuglément.**

***Je ne peux que recommander à tout le monde de faire de même.***