# Projet : tarification du parking et mode de transport

Projet de groupe (politique publique). Consignes complètes dans `Project_guidelines.pdf`. Rendu le 8 décembre, 23h59.

## Contexte
- Employeur public suisse, 2 sites : Lausanne (VD, 800 employés) et Sion (VS, 200 employés).
- 2024 : parking à 2 CHF/jour sur les deux sites. 2025 : Lausanne passe à 4 CHF, Sion reste à 2 CHF.
- Panel : 1000 employés observés en 2024 et 2025 (`parking_data.csv`, 2000 lignes).
- 46 semaines travaillées par an.
- Questions : combien arrêtent de conduire quand le prix double ? que se passe-t-il à 6 ou 8 CHF ? qui paie ?

## Livrables
1. Mémo politique PDF, 3 pages max, 2 tableaux ou figures max (destiné au responsable mobilité et à la commission du personnel).
2. Rapport technique PDF, 8 pages max, sans page de garde. Doit contenir les Tables A1, B1 à B3, C1 à C3, D1, E1 et la Figure D1.
3. Un seul script reproductible `analysis.py` qui recrée tous les résultats depuis `parking_data.csv` + `transit_access.csv`.
4. `transit_access.csv` : indice d'accès aux transports publics par commune, avec sources et notes de codage.

## Parties
- **A** : indice d'accès TP par commune (temps TP / temps voiture pour arrivée vers 08h00, nombre de départs 06h00 à 09h00, première arrivée possible). Score continu ou catégories ordonnées. Table A1 + paragraphe de validation.
- **B** (marge extensive, uses_car) : B1 parts modales par site et année + flux voiture vers non-voiture et inverse ; variation en points de % et en élasticité ; B2 hétérogénéité (accès TP, tercile de revenu, nombre d'enfants) avec effectifs ; B3 régression en 4 colonnes : (a) Lausanne seule, (b) + Sion, (c) + contrôles dont accès TP, (d) + interactions log(fee) x accès TP, revenu, enfants. Modèle de probabilité linéaire. Un seul paragraphe court sur la comparaison avec Sion.
- **C** (marge intensive, automobilistes seulement, log(car_days_per_week)) : C1 élasticité par groupe ; C2 mêmes 4 spécifications ; discussion de la sélection des automobilistes 2025 ; C3 décomposition elasticité totale = extensive + intensive avec définitions et échantillons cohérents.
- **D** : scénarios à 5, 6, 8, 10 CHF (+ base 4 CHF) : part voiture, jours par automobiliste, jours voiture par employé, recette annuelle. Table D1, Figure D1, sensibilité, limites de l'extrapolation.
- **E** : dépense annuelle de parking en % du revenu brut observé, par groupe : (a) tous les employés, (b) ceux qui conduisent encore en 2025, (c) au tarif recommandé. Table E1. Ne pas inférer de bien-être ni de coûts TP non observés.
- **F** : recommandation de tarif 2026 + mesures complémentaires, alternative écartée, plan de suivi avec 2 ou 3 indicateurs.

## Problèmes de données repérés (à traiter et documenter)
- `home_commune` : 62 libellés pour environ 39 communes (majuscules, tirets : « PRILLY » contre « Prilly »). Harmoniser. Après harmonisation, aucun employé ne change de commune entre les années.
- `gross_income_chf` : 110 valeurs manquantes. Imputer depuis l'autre année du même employé si possible.
- `car_days_per_week` : E1001 en 2025 vaut 7,4 (max théorique 5).
- Le revenu est au taux d'activité observé (`fte`) : utiliser revenu / fte pour les terciles, revenu observé pour le taux d'effort.

## Conventions
- Python (pandas, statsmodels, matplotlib). Erreurs standard groupées par employé.
- Tableaux mis en forme soi-même, jamais de sortie brute de statsmodels.
- Indiquer variable dépendante, unités, catégories de référence, méthode des erreurs standard, N et restrictions d'échantillon.
- Arrondis identiques dans tout le rapport.
- Ne pas utiliser de tirets longs dans les textes rédigés.
- L'usage de l'IA doit être déclaré brièvement dans le rapport.
