# Comparaison humaine entre le système de RAG servant de baseline et le système de RAG final

## Contexte

Deux personnes ont été sollicitées afin de comparer les réponses générées par le système de RAG servant de baseline et le système de RAG final.
L’objectif était d’obtenir une évaluation humaine, considérée comme plus fiable que les métriques basées sur des LLM-as-Judges.

Pour chaque question de l’ensemble du jeu de données (jeux de développement et de test), les évaluateurs disposaient des éléments suivants :
- la question
- la réponse de référence (*ground truth*)
- les références (*ground truth*)
- les réponses générées par chacun des deux systèmes.

Les documents de référence eux-mêmes n’ont pas été consultés. Seuls les numéros de LEX ou d’article ont parfois été utilisées pour s'assurer que le LLM citait la bonne source.
Les réponses des systèmes baseline et final étaient présentées dans un ordre aléatoire afin de pas biaiser l'évaluation.

Pour chaque question, les évaluateurs devaient sélectionner la meilleure réponse et fournir une brève justification de leur choix.
L’évaluation reposait en partie sur l’appréciation subjective de chaque personne. Celle-ci pouvait notamment dépendre :
- des connaissances personnelles en plus ou en moins par rapport à la réponse de référence
- de la longueur jugée appropriée de la réponse
- du poids accordé à une réponse plus complète mais comportant quelques erreurs, par rapport à une réponse correcte mais incomplète.

## Analyse

Sur les 27 points au total, 22 points ont été accordés au système final et 5 points au système baseline.

Les points accordés au système baseline peuvent se justifier de la façon suivante :
- 3 points proviennent du fait que le système final n'a pas réussi à construire un contexte pertinent à envoyer au LLM
- 2 points (4 fois 0.5) proviennent d'un jugement subjectif sur la verbosité de la réponse et l'attitude du LLM lorsqu'il ne possède pas les informations nécessaires dans son contexte.

Les réponses générées par le système final sont ainsi sélectionnées dans 81% des cas.

## Evaluation

### Jeu de développement (13 questions)

1)
Question : Quels sont les membres qui composent la Direction de l'EPFL ?

Réponse : La Direction de l’EPFL se compose d’un(e) Président(e) et de plusieurs Vice-Président(e)s.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.1_o_organisation_EPFL_fr.pdf
-	Document : LEX 1.1.1
-	Paragraphe : Article 3

Réponse système baseline :

```markdown
Le contexte fourni ne mentionne pas explicitement les membres qui composent la **Direction de l'EPFL**. Les informations disponibles concernent principalement les organes internes des facultés (comme les commissions d'enseignement, les services généraux, etc.) et des commissions spécifiques (comme la Commission de restauration), mais pas la composition de la Direction de l'EPFL elle-même.

Ainsi, **aucune réponse précise ne peut être donnée** à partir du contexte fourni.
```

Réponse système final :

```markdown
La Direction de l'EPFL se compose des membres suivants (selon l'article 3 de l'ordonnance) :

1. La Présidente ou le Président ;
2. La Vice-présidente ou le Vice-président académique (VPA / Provost) ;
3. La Vice-présidente ou le Vice-président pour le soutien aux initiatives stratégiques (VPS) ;
4. La Vice-présidente ou le Vice-président pour l’innovation et l’impact (VPI) ;
5. La Vice-présidente ou le Vice-président pour le développement humain (VPH) ;
6. La Vice-présidente ou le Vice-président pour les finances (VPF) ;
7. La Vice-présidente ou le Vice-président pour les opérations (VPO).

La Direction est majoritairement composée de professeures et professeurs de l'EPFL (article 3, alinéa 3).
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est claire, détaillée et concise alors que l'autre version ne permet pas de répondre à la question.

2)
Question : Les emprunts dans un travail écrit doivent-ils être reconnaissable via une note en bas de page ?

Réponse : Non, uniquement reconnaissables.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.3.3_dir_plagiat_etudiant_fr.pdf
-	Document : LEX 1.3.3
-	Paragraphe : Article 4

Réponse système baseline :

```markdown
Oui, les emprunts dans un travail écrit doivent être reconnaissables comme tels, notamment par une référence à la source. Selon le contexte fourni (Chunk 1, art. 4 et art. 5), tout emprunt doit être clairement indiqué dans le travail, que ce soit sous forme de citation ou de reprise de substance, afin de distinguer l'apport personnel de l'étudiant. Une note en bas de page ou un renvoi vers les coordonnées de la source (référence) est une manière courante de respecter cette obligation.
```

Réponse système final :

```markdown
Oui, les emprunts dans un travail écrit doivent être reconnaissables comme tels, notamment par une **référence à la source** (LEX 1.3.3, art. 4 al. 2 et LEX 3.3.2, art. 14 al. 2). Bien que le contexte ne mentionne pas explicitement les notes en bas de page, les références peuvent prendre cette forme (par exemple, via des citations ou des renvois bibliographiques). L'important est que l'emprunt soit clairement identifié pour éviter le plagiat (LEX 1.3.3, art. 8 et LEX 3.3.2, art. 18).
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est moins catégorique que l'autre version, mais elles se ressemblent beaucoup.

3)
Question : Est-ce que le COSEC est rattaché à la VPF à l'EPFL ?

Réponse : Il n'est pas rattaché à la VPF.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.5.1_dir_sante_securite_travail_fr.pdf
-	Document : LEX 1.5.1
-	Paragraphe : Article 14

Réponse système baseline :

```markdown
Le contexte fourni ne mentionne pas explicitement un rattachement du COSEC (Correspondant de Sécurité) à la VPF (Vice-Présidence Finances) de l'EPFL. Les informations disponibles concernent principalement les responsabilités, les critères de sélection et les tâches du COSEC, ainsi que son rôle en tant que personne de contact du DSE (Département de Sécurité et Environnement) au sein de l'unité. Aucune mention n'est faite d'un lien hiérarchique ou fonctionnel avec la VPF.

**Réponse :** Non, le contexte fourni ne permet pas de conclure que le COSEC est rattaché à la VPF à l'EPFL.
```

Réponse système final :

```markdown
Non, le COSEC (Correspondant de Sécurité) n'est pas rattaché à la Vice-Présidence Formation (VPF) à l'EPFL. Selon le contexte fourni, le COSEC est rattaché au **Département de Sécurité, Sûreté et Exploitation (DSE)** de la **Vice-Présidence pour les Opérations (VPO)**.

Le DSE, sous la responsabilité de la VPO, supervise l'organisation de la sécurité et de la santé au travail (SSST) à l'EPFL, et le COSEC agit comme son contact au sein des unités (Article 13 et 14 de la directive). La VPF (Vice-Présidence Formation) n'est pas mentionnée dans les responsabilités liées au COSEC.

La VPO est responsable de la **conduite opérationnelle de la sécurité** (Article 3 et 4), tandis que la **Vice-Présidence Académique (VPA)** se concentre sur la promotion de la sécurité dans la formation et la recherche (Article 10). La VPF n'apparaît pas dans le contexte fourni.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée donne plus d'informations et donne des détails après avoir donné la réponse recherchée.

4)
Question : Est-ce qu'une décision de la Commission des prix de la recherche ou d’un jury spécifique doit être notifiées formellement au candidat par le biais d'une lettre recommandée ?

Réponse : Pas besoin de recommandé.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2021/10/LEX-3.3.4.pdf
-	Document : LEX 3.3.4
-	Paragraphe : Article 5

Réponse système baseline :

```markdown
Non, selon le contexte fourni, les décisions de la Commission des prix de la recherche ou d’un jury spécifique doivent être notifiées formellement au candidat par le biais d'une **lettre ou d’un email de décision**, mais il n'est pas mentionné que la notification doit se faire par lettre recommandée.
```

Réponse système final :

```markdown
Non, la décision de la Commission des prix de la recherche ou d’un jury spécifique doit être notifiée formellement au candidat par le biais d'une **lettre ou d’un email de décision**, comme le stipule l'article 5, alinéa 8 du règlement. Il n'est pas mentionné que cette notification doit obligatoirement être envoyée par lettre recommandée.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est plus fluide et plus complète.

5)
Question : A combien de semaines de vacances par an ai-je le droit à partir de 60 ans en tant qu'employé à l'EPFL ?

Réponse : 6 semaines

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.1.4_r_gestion_temps_travail_fr.pdf
-	Document : LEX 4.1.4
-	Paragraphe : Article 17

Réponse système baseline :

```markdown
D'après le contexte fourni, à partir de 60 ans, vous avez droit à **5 semaines de vacances par an** (soit 25 jours). Le droit à 6 semaines (30 jours) s'applique uniquement aux moins de 20 ans et aux 50 ans et plus, mais il n'est pas mentionné de prolongation au-delà de 50 ans.

*(Source : Chunk 1, Article 17.2 et Chunk 3, Art. 51.2)*
```

Réponse système final :

```markdown
1. **5 semaines de vacances** par année civile pour les collaborateurs de moins de 50 ans.
2. **6 semaines de vacances** à partir de l'année où le collaborateur atteint **50 ans** (art. 51 al. 2 de l'OPers-EPF).
3. **6 semaines de vacances** pour les jeunes jusqu'à l'année où ils atteignent **20 ans** (art. 51 al. 3 de l'OPers-EPF).

**Conclusion :**
À partir de **50 ans**, vous avez droit à **6 semaines de vacances par an** (soit **30 jours ouvrés**). Ce droit reste valable au-delà de 60 ans, sauf disposition contraire non mentionnée dans le contexte fourni.

*(Source : Art. 51 OPers-EPF et Art. 17 LPers, cités dans le contexte.)*
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est complète mais très verbeuse et pourrait porter à confusion.

6)
Question : Est-ce qu'un doctorant peut racheter son ordinateur ?

Réponse : Un doctorant peut racheter son ordinateur uniquement s'il est âgé de 4 ans ou plus, s'il n'a pas été réparé par le CRI pendant les six derniers mois, si le responsable d’unité détentrice de l’équipement est d'accord de le céder et si le responsable d'unité dispose des crédits nécessaires pour remplacer l'ordinateur.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/6.4.1.1_dir_recyclage_materiel_it_fr.pdf
-	Document : LEX 6.4.1
-	Paragraphe : Article 5

Réponse système baseline :

```markdown
Le contexte fourni ne contient aucune information concernant la possibilité pour un doctorant de racheter son ordinateur. Par conséquent, il n'est pas possible de répondre à cette question en se basant uniquement sur les éléments fournis.
```

Réponse système final :

```markdown
Le contexte fourni ne mentionne pas explicitement la possibilité pour un doctorant de racheter son ordinateur. Aucune disposition dans les documents fournis (Directive en matière de santé et sécurité du travail, Ordonnance sur les mesures disciplinaires, ou Directive concernant la formation doctorale) ne traite de cette question.

Par conséquent, sur la base du contexte donné, il n'est pas possible de répondre à cette question. Il faudrait consulter d'autres sources ou directives spécifiques à l'EPFL concernant la gestion du matériel informatique pour les doctorants.
```

Evaluation :
- scores:
  - système baseline : 0.5
  - système final : 0.5
- justification : La réponse du système final donne une piste de recherche sans être forcément meilleure que l'autre version. A l'inverse, il est dit clairement dans la réponse du système baseline qu'il n'est pas possible de répondre à cette question.

7)
Question : En tant qu’étudiant, puis-je créer un site web pour mon association avec un nom de domaine externe (ex : monlabo.ch) au lieu d’un site epfl.ch ? Est-ce que j'ai des contraintes à respecter concernant le choix du nom de domaine ?

Réponse : Tout dépend de l'association. Si l'association respecte les critères cités dans l'article 7 de la LEX 6.2.1, le choix du nom de domaine doit être validé par l’unité MEDIACOM.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/6.2.1_dir_gestion_noms_domaines_autres_epfl_fr.pdf
-	Document : LEX 6.2.1
-	Paragraphe : Article 7

Réponse système baseline :

```markdown
Oui, en tant qu'étudiant, vous pouvez créer un site web pour votre association avec un nom de domaine externe (ex: monlabo.ch) plutôt qu'un sous-domaine epfl.ch, mais vous devez respecter certaines contraintes.

Selon le **Chunk 1** (Directive pour la gestion de noms de domaines), les associations étudiantes reconnues par l'EPFL peuvent demander un nom de sous-domaine correspondant à leur acronyme (ex: association.epfl.ch). Cependant, si vous choisissez un nom de domaine externe, les règles suivantes s'appliquent :

1. **Responsabilité** : Le responsable de l’unité concernée (probablement votre association) est responsable du nom de domaine (Article 7, Chunk 1).
2. **Validation par MEDIACOM** : Si votre projet ou association répond aux critères cumulatifs suivants, le choix du nom de domaine doit être validé par l’unité MEDIACOM :
   - Le projet implique des participants actifs en dehors de l'EPFL.
   - La gestion du site web est sous la responsabilité d’une unité de l’École.
   - L’adresse du site (URL) est largement diffusée en dehors de l'EPFL (Chunk 1 et Chunk 4, Article 7).
3. **Interdiction de mots isolés** : Le nom de domaine ne doit pas contenir de mots isolés provenant de dictionnaires standard (anglais ou français), sauf exception approuvée par la Direction de l'EPFL (Chunk 1, point 6).
4. **Enregistrement et frais** : Vous êtes responsable de l'enregistrement du nom de domaine auprès de l'autorité compétente et du paiement des frais. Le nom doit être enregistré au nom de l’association, et une adresse e-mail de groupe (et non personnelle) doit être utilisée (Chunk 4, Article 9).

Si votre association n'est pas reconnue par l'EPFL, les mêmes règles s'appliquent (Chunk 4, Article 7, paragraphe 2).

En résumé, vous pouvez utiliser un nom de domaine externe, mais il doit être validé par MEDIACOM si votre site est largement diffusé en dehors de l'EPFL, et il ne doit pas contenir de mots isolés courants.
```

Réponse système final :

```markdown
Oui, en tant qu'étudiant, vous pouvez créer un site web pour votre association avec un nom de domaine externe (ex: monlabo.ch) au lieu d'un sous-domaine epfl.ch, sous certaines conditions.

**Contraintes à respecter :**
1. **Validation par MEDIACOM** : Le choix du nom de domaine doit être validé par l’unité MEDIACOM si votre association répond aux critères suivants (Article 7) :
   - Le projet implique des participants actifs en dehors de l'EPFL.
   - La gestion du site web est sous la responsabilité d’une unité de l’Ecole.
   - L’adresse du site (URL) est largement diffusée en dehors de l'EPFL.

2. **Responsabilité de l'enregistrement** : Vous êtes responsable de l'enregistrement du nom de domaine auprès de l'autorité compétente et du paiement des frais afférents (Article 9).

3. **Interdiction des serveurs DNS internes** : Il n'est pas permis d'installer un serveur DNS accessible de l’extérieur sur une machine raccordée au réseau de l'EPFL (Article 9).

4. **Utilisation du logo EPFL** : Si votre association est reconnue par l'EPFL, vous pouvez utiliser le logo "Association reconnue par l'EPFL" sur votre site, mais cela ne dispense pas des règles ci-dessus.

**Cas particulier des associations étudiantes reconnues** :
Si votre association est reconnue par l'EPFL et figure dans l’annuaire des unités de l’EPFL, vous pouvez demander un nom de sous-domaine correspondant à votre acronyme (ex: monlabo.epfl.ch) (Article 6 al. 5). Cependant, vous avez aussi la possibilité d'utiliser un nom de domaine externe, sous réserve des conditions mentionnées ci-dessus.

En résumé, un nom de domaine externe est possible, mais il doit être validé par MEDIACOM si les critères sont remplis, et vous devez respecter les règles d'enregistrement et d'infrastructure.
```

Evaluation :
- scores:
  - système baseline : 0.5
  - système final : 0.5
- justification : Les deux réponses sont détaillées et contiennent des informations correctes. La sélection dépend de la verbosité de la réponse générée.

8)
Question : Quelles sont les conditions de réussite pour un bloc au niveau master à l'EPFL ?

Réponse : Un bloc est réussi à partir du moment où toutes les matières du bloc ont été présentées et si la moyenne du bloc est de 4,00 ou plus. Le calcul de la moyenne tient compte des coefficients. A noter qu'il ne faut pas avoir de N/A car cela empêche de calculer la moyenne d'un bloc.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2015/476/fr
-	Document : LEX 2.6.0.1
-	Paragraphe : Article 6

Réponse système baseline :

```markdown
D'après le contexte fourni, les conditions de réussite pour un bloc au niveau master à l'EPFL ne sont pas explicitement mentionnées dans les extraits fournis. Les informations disponibles concernent principalement les procédures d'achat, les règles d'examen, les langues d'enseignement et les procédures de consultation, mais pas les critères de réussite spécifiques pour un bloc de master.

Si vous avez besoin de ces informations, il serait utile de consulter directement l'ordonnance ou la directive pertinente sur les conditions de réussite des blocs au niveau master à l'EPFL.
```

Réponse système final :

```markdown
D'après le contexte fourni, les conditions de réussite pour un bloc au niveau master à l'EPFL sont les suivantes :

1. **Cycle master** :
   - Le cycle master est considéré comme réussi lorsque l'étudiant a acquis **60 ou 90 crédits ECTS** (selon la filière), conformément à l'ordonnance sur la formation à l'EPFL (Art. 10, al. 3).
   - La durée maximale pour réussir le cycle master de 60 crédits ECTS est d'**une année**, mais elle ne peut excéder **deux ans**. Pour un cycle de 90 crédits ECTS, la durée est d'**une année et demie**, mais ne peut excéder **trois ans** (Art. 10, al. 2).

2. **Projet de master** :
   - Le projet de master, qui représente **30 crédits ECTS**, doit être entamé **au plus tard un an après la réussite du cycle master** (Art. 11, al. 1bis).
   - Il doit être réussi **au plus tard trois ans après l’entrée au cycle master de 60 crédits ECTS** et **quatre ans après l’entrée au cycle master de 90 crédits ECTS** (Art. 11, al. 2).
   - La réussite du cycle master est une condition préalable pour entamer le projet de master (Art. 11, al. 3).

Ces conditions sont définies dans l'**Ordonnance sur la formation menant au bachelor et au master de l'EPFL** (LEX 2.11.0.1 2), en particulier aux articles 10 et 11.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est plus détaillée et semble plus acceptable que l'autre version.

9)
Question : Je viens d'obtenir mon bachelor à l'EPFL. Quel est mon titre officiel ?

Réponse : Vous pouvez trouver le titre et la désignation professionnelle de votre formation dans l'annexe I de l'ordonnance sur la formation menant au bachelor et au master : https://www.fedlex.admin.ch/eli/cc/2004/653/fr.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2004/653/fr
-	Document : LEX 2.11.0.1
-	Paragraphe : Annexe 1

Réponse système baseline :

```markdown
D'après le contexte fourni, il n'y a pas d'information spécifique sur le titre officiel attribué après l'obtention d'un bachelor à l'EPFL. Le contexte mentionne des directives sur les titres académiques (comme maître d'enseignement et de recherche ou professeur titulaire), les langues d'enseignement, les procédures de contestation, et une convention d'objectifs, mais pas le titre exact pour un bachelor.

Cependant, selon les pratiques courantes des universités suisses, le titre officiel après un bachelor à l'EPFL est généralement **"Bachelor of Science (BSc)"** dans le domaine d'études concerné (par exemple, BSc en ingénierie, BSc en sciences, etc.). Si vous avez besoin d'une confirmation exacte, il est recommandé de consulter directement le règlement des études de l'EPFL ou le service académique compétent.
```

Réponse système final :

```markdown
D'après le contexte fourni, le titre officiel décerné par l'EPFL après l'obtention du bachelor est **Bachelor of Science (BSc)**. Ce titre est mentionné dans l'annexe I de l'ordonnance sur la formation à l'EPFL, où il est précisé que le bachelor est accompagné de la désignation "Bachelor of Science" (BSc) dans le domaine d'études concerné. Par exemple, pour un bachelor en architecture, le titre complet serait "Bachelor of Science BSc Architecture".
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée permet de répondre à la question.

10)
Question : J'envisage de commencer mes études à l'EPFL. Combien d'années faut-il compter pour faire sa formation à l'EPFL ?

Réponse : Pour obtenir un diplôme de master de l'EPFL le cursus se déroule en 5 années minimum, qui se compose de 3 années pour le Bachelor suivi de 2 années pour le Master. Les étudiants en architecture doivent en outre effectuer un stage de 12 mois entre le Bachelor et le Master. Selon le certificat d'études secondaire obtenu certains candidats doivent suivre au préalable une année préparatoire avant de commencer le Bachelor. Les conditions d'admission détaillées peuvent être consultées sur la page suivante : https://www.epfl.ch/education/admission/fr/admission/conditions-dadmission-inscription-bachelor/.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2004/653/fr
-	Document : LEX 2.11.0.1
-	Paragraphe : informations réparties sur plusieurs articles

Réponse système baseline :

```markdown
Le contexte fourni ne mentionne pas explicitement la durée des formations à l'EPFL. Cependant, il évoque des critères d'admission pour le bachelor et le master, ainsi que des règles liées à la prévoyance ou aux titres académiques. Pour obtenir une réponse précise sur la durée des études, il serait nécessaire de consulter les informations spécifiques aux programmes de bachelor ou de master proposés par l'EPFL, qui ne sont pas incluses dans les extraits fournis.

(Note : En général, un bachelor à l'EPFL dure 3 ans, et un master 1,5 à 2 ans, mais cette information n'est pas tirée du contexte donné.)
```

Réponse système final :

```markdown
D'après le contexte fourni, la durée de la formation à l'EPFL dépend du niveau d'études visé :

1. **Bachelor** :
   - Cycle propédeutique : 2 semestres (1 an maximum).
   - Cycle bachelor : 2 années d'études (soit 4 semestres).
   - **Total** : **3 ans** (minimum) à **4 ans** (si prolongation nécessaire).

2. **Master** :
   - Cycle master : 1 an (60 crédits ECTS) ou 1,5 an (90 crédits ECTS).
   - Projet de master : 1 semestre (30 crédits ECTS).
   - **Total** : **2 ans** (pour un master à 60 crédits) ou **2,5 ans** (pour un master à 90 crédits).

**Durée totale estimée** (bachelor + master) :
- **5 à 6 ans** selon le parcours choisi.

*Remarque* : Des prolongations peuvent être accordées pour des motifs valables (maladie, maternité, etc.), comme le prévoit l'article 12 de l'ordonnance.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est moins confuse et meilleure malgré les calculs incorrects pour les totaux.

 11)
Question : Existe-t-il un code d'honneur pour les étudiants en bachelor et où puis-je le trouver ?

Réponse : Oui il en existe un. Chaque semestre lorsque vous validez le formulaire FRAC vous avez le lien du code d'honneur. En outre vous pouvez également le trouver sur la page suivante : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/2.3.1_ch_code_honneur_fr.pdf.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/2.3.1_ch_code_honneur_fr.pdf
-	Document : LEX 2.3.1
-	Paragraphe : pas une structure standardisée mais courte liste à puces

Réponse système baseline :

```markdown
Oui, il existe un **code d'honneur** pour les étudiants en bachelor à l'EPFL. Il est mentionné dans le contexte fourni (Chunk 2 et Chunk 5) et fait référence à la **charte éthique de l'EPFL** (disponible en ligne : [https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/Charte-ethique-de-lEFPL.pdf](https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/Charte-ethique-de-lEFPL.pdf)).

Vous pouvez également consulter le **Code d'honneur** directement via le lien suivant : [formation.epfl.ch](http://formation.epfl.ch) (mentionné dans le Chunk 5).

Ce code souligne les responsabilités éthiques des étudiants et leur engagement à respecter les principes de l'EPFL.
```

Réponse système final :

```markdown
Oui, il existe un **Code d'honneur** pour les étudiants en bachelor à l'EPFL. Il est mentionné dans le contexte fourni sous la référence **[LEX 2.3.1 1]**.

**Où le trouver ?**
Le Code d'honneur est disponible sur le site de l'EPFL, en référence à la **Charte éthique de l'EPFL** :
🔗 [https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/Charte-ethique-de-lEFPL.pdf](https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/Charte-ethique-de-lEFPL.pdf)

**Résumé du Code d'honneur** :
Il souligne les **responsabilités éthiques** des futurs ingénieurs, scientifiques et architectes de l'EPFL. En l'adoptant, chaque étudiant s'engage à :
- Contribuer au **rayonnement de l'École**,
- Renforcer la **confiance mutuelle** sur le campus,
- Valoriser les **titres EPFL**.

Pour plus de détails, consultez la **Charte éthique** ou la rubrique **déontologie et éthique** de l'EPFL.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée mentionne la LEX de référence et est complète.

12)
Question : Puis-je être immatriculé dans deux universités en même temps ?

Réponse : Non, il n'est pas autorisé à un étudiant d'être immatriculé dans deux universités en parallèle. Si vous souhaitez suivre une formation à l'EPFL vous devez vous exmatriculer de votre université précédente.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2025/782/fr
-	Document : LEX 2.1.0.1
-	Paragraphe : Article 4

Réponse système baseline :

```markdown
Le contexte fourni ne mentionne pas explicitement la possibilité d'être immatriculé dans deux universités en même temps. Cependant, il évoque la mobilité horizontale (étudier dans une autre université tout en restant rattaché à l'EPFL) et la mobilité verticale (admission en master à l'EPFL pour des étudiants externes). Ces dispositions suggèrent que des collaborations institutionnelles existent, mais ne précisent pas si une double immatriculation est autorisée.

Pour une réponse définitive, il faudrait consulter les règles spécifiques de chaque section ou le responsable de la mobilité de l'EPFL, comme indiqué dans le [Chunk 5].
```

Réponse système final :

```markdown
Non, selon l'article 4, alinéa 3, let. a, de l'ordonnance de la Direction de l'EPFL sur l'admission à l’École polytechnique fédérale de Lausanne (Ordonnance sur l’admission à l’EPFL), une personne qui est déjà immatriculée dans une autre haute école n'est pas admissible à l'EPFL, sauf circonstances particulières dictées par la formation suivie. Cela signifie qu'il n'est généralement pas possible d'être immatriculé simultanément dans deux universités.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est correctement sourcée et moins confuse que l'autre version.

13)
Question : Je suis en première année de bachelor et j'ai obtenu la moyenne de 3,49 au semestre. Comment m'inscrire en BA2 ?

Réponse : Les étudiants qui obtiennent une moyenne inférieure à 3,50 au bloc 1 à la fin du premier semestre de Bachelor ne sont pas autorisés à suivre le BA2. Ils doivent faire le semestre de Mise à niveau durant le semestre de printemps. Il est obligatoire de la réussir pour être autorisé à redoubler l'année, sinon vous serez en situation d'exclusion de l'EPFL.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2015/476/fr
-	Document : LEX 2.6.0.1
-	Paragraphe : Article 22

Réponse système baseline :

```markdown
Le contexte fourni ne contient pas d'informations spécifiques sur les conditions d'inscription en deuxième année de bachelor (BA2) à l'EPFL, notamment en ce qui concerne les critères de moyenne ou de procédure d'inscription.

Pour obtenir une réponse précise, il serait nécessaire de consulter les directives académiques de l'EPFL ou de contacter le service des affaires étudiantes.
```

Réponse système final :

```markdown
Pour vous inscrire en deuxième année de bachelor (BA2) à l'EPFL, vous devez avoir réussi le cycle propédeutique (BA1), qui s'étend sur deux semestres et permet d'acquérir 60 crédits ECTS (art. 7 de l'ordonnance sur la formation à l'EPFL).

Votre moyenne de 3,49 au premier semestre ne suffit pas à garantir automatiquement votre passage en BA2. Voici les étapes à suivre :

1. **Vérifiez les conditions de réussite** :
   - Le cycle propédeutique est considéré comme réussi si vous avez acquis les 60 crédits ECTS requis (art. 7, al. 4 de l'ordonnance sur la formation à l'EPFL).
   - Si vous n'avez pas encore validé tous les crédits nécessaires, vous devrez les compléter avant de passer en BA2.

2. **Consultez votre relevé de notes** :
   - Vérifiez si vous avez obtenu les crédits ECTS nécessaires pour valider le cycle propédeutique. Si des matières sont en échec, vous devrez les repasser.

3. **Procédure d'inscription** :
   - Une fois les 60 crédits ECTS acquis, vous serez automatiquement inscrit en BA2 pour le semestre suivant, sous réserve de respecter les délais administratifs (art. 8 de l'ordonnance sur la formation à l'EPFL).
   - Si vous avez des doutes, contactez le service des admissions ou votre secrétariat étudiant pour confirmer votre éligibilité.

4. **Délais** :
   - Assurez-vous de respecter les délais d'inscription pour le semestre suivant, publiés sur le site de l'EPFL.

Si vous n'avez pas encore validé tous les crédits, vous devrez les obtenir avant de pouvoir passer en BA2.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est précise, mais bien trop verbeuse.

### Jeu de test

1)
Question : L'AE est l'organe faîtier de participation au sens de quelle ordonnance et quel article ?

Réponse : Ordonnance EPFZ EPFL art. 18

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.1_o_organisation_EPFL_fr.pdf
-	Document : LEX 1.1.1
-	Paragraphe : Article 23

Réponse système baseline :

```markdown
L'AE (Assemblée d'école) est l'organe faîtier de participation au sens de l'**ordonnance de l'EPFZ et de l'EPFL** et de l'**article 18** de cette ordonnance.

*(Source : Chunk 2, Article 23, paragraphe 1)*
```

Réponse système final :

```markdown
L'AE (Assemblée d'École) est l'organe faîtier de participation au sens de l'**article 18 de l'ordonnance sur l'EPFZ et l'EPFL** (RS 414.110.37).

*(Source : Article 23, alinéa 1 de l'ordonnance sur l'organisation de l'EPFL [LEX 1.1.1 1], contexte fourni.)*
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée ressemble beaucoup à l'autre version mais n'est pas redondante.

2)
Question : C'est quoi OHS ?

Réponse : service de Sécurité et Santé au travail

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.5.1_dir_sante_securite_travail_fr.pdf
-	Document : LEX 1.5.1
-	Paragraphe : Article 7

Réponse système baseline :

```markdown
D'après le contexte fourni, **OHS** signifie **Occupational Health and Safety** (Santé et Sécurité au Travail). Il s'agit d'un service de l'EPFL chargé de superviser les questions de santé et de sécurité au travail, en collaboration avec les Ressources Humaines (RH) et d'autres entités. Ses missions incluent notamment :
- La gestion du cabinet de médecine du travail.
- La coordination avec les correspondants de sécurité (COSECs).
- La formation obligatoire en sécurité pour les nouveaux employés.
- Le rôle de point de contact avec les autorités officielles en matière de sécurité et santé au travail.

(Source : Chunks 2, 3 et 5 du contexte.)
```

Réponse système final :

```markdown
D'après le contexte fourni, **OHS** signifie **Occupational Health and Safety Service** (Service de sécurité et santé au travail). Il s'agit du centre de compétence de l'EPFL pour la mise en œuvre des mesures internes en matière de sécurité et santé au travail. Ses missions incluent :

- L'évaluation des risques et l'analyse des accidents.
- La formation obligatoire en sécurité (FOBS, COSEC).
- La gestion de la médecine du travail et des équipements de protection.
- La collaboration avec les unités pour assurer le respect des directives.

(Source : Articles 7 et 14 de la directive DOHSS/DSSST.)
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est plus concise et mentionne le terme exact.

3)
Question : De combien de membres la commission des prix de la recherche de l'EPFL est-elle composée ?

Réponse : C'est entre 10 et 15 membres.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2021/10/LEX-3.3.4.pdf
-	Document : LEX 3.3.4
-	Paragraphe : Article 2

Réponse système baseline :

```markdown
La Commission des prix de la recherche de l'EPFL est composée de **dix (10) à quinze (15) membres** (selon le contexte fourni dans [Chunk 2], Article 2. Membres).
```

Réponse système final :

```markdown
La Commission des prix de la recherche de l'EPFL est composée de **dix (10) à quinze (15) membres**, selon l'article 2.1 du règlement.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est quasiment identique à l'autre version, la mention de la référence est cependant mieux présentée.

4)
Question : Est-ce que je peux utiliser mon ordinateur du travail pour des tâches privées en tant qu'employée à l'EPFL ?

Réponse : Oui, sous certaines conditions. L'utilisation privée ne doit pas entraîner de grands coûs (économique et énergétique) et ne doit pas nuire aux obligations professionnelles des collaborateurs vis-à-vis de l'EPFL.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2020/01/LEX-6.1.4_FR.pdf
-	Document : LEX 6.1.4
-	Paragraphe : Article 5

Réponse système baseline :

```markdown
D'après le contexte fourni, l'employeur (l'EPFL) permet généralement aux employés d'utiliser Internet et le courrier électronique à titre privé, **dans certaines limites** (Chunk 2). Cependant, cela dépend du champ d'activité et des obligations contractuelles. Si l'utilisation privée n'entrave pas l'accomplissement des tâches professionnelles, elle est autorisée.

En revanche, le contexte ne mentionne pas explicitement l'utilisation de l'ordinateur professionnel pour des tâches privées autres que la navigation Internet et les e-mails. Il est donc préférable de se référer aux règles internes de l'EPFL ou de consulter les ressources humaines pour une réponse précise.

*(Réponse basée uniquement sur les informations fournies dans le contexte.)*
```

Réponse système final :

```markdown
Non, selon le contexte fourni, l'utilisation de l'ordinateur professionnel de l'EPFL pour des tâches privées n'est pas autorisée. Le règlement applicable au télétravail (LEX 4.1.8) et la directive sur l'utilisation du matériel informatique privé à des fins professionnelles (LEX 6.1.3) précisent que l'EPFL fournit tout le matériel nécessaire au travail de ses collaborateurs et que l'utilisation de matériel informatique privé à des fins professionnelles est tolérée sous certaines conditions strictes, mais l'inverse (utilisation du matériel professionnel à des fins privées) n'est pas mentionné comme autorisé.

De plus, la directive LEX 6.1.3 stipule que les collaborateurs doivent respecter le secret de fonction, la protection des données et les conditions d'utilisation des logiciels, ce qui implique une utilisation exclusive à des fins professionnelles. Toute utilisation privée pourrait engager la responsabilité de l'employée et violer les règles de sécurité et de confidentialité de l'EPFL.
```

Evaluation :
- scores:
  - système baseline : 1
  - système final : 0
- justification : La réponse sélectionnée parait plus juste et il n'y a pas de prise de risque.

5)
Question : Quelles sont les conditions pour entrer à l'EPFL si je viens du Sénégal ?

Réponse : Un diplôme d'études secondaires supérieures est nécessaire et il faut passer par un examen d'admission.

Référence (réponse rédigée sur la base d’informations disponibles sur le site de Swissuniversities) :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2025/782/fr
-	Document : LEX 2.1.0.1
-	Paragraphe : Article 6

Réponse système baseline :

```markdown
D'après le contexte fourni, les conditions pour entrer à l'EPFL si vous venez du Sénégal sont régies par l'**Ordonnance de la Direction de l’EPFL sur l’admission à l’École polytechnique fédérale de Lausanne** (Chunk 2). Cette ordonnance énonce les conditions à remplir pour être admis, que ce soit en **1ère année** ou en **master**, et elle règle également les modalités de l'examen d'admission pour les candidats qui y sont soumis avant d'entrer en 1ère année.

Cependant, le contexte ne précise pas les critères spécifiques pour les candidats venant du Sénégal (comme les équivalences de diplômes, les exigences linguistiques, etc.). Pour des détails précis, il faudrait consulter directement l'ordonnance ou le site officiel de l'EPFL.

*(Note : Le contexte ne contient pas d'informations spécifiques sur les conditions d'admission pour les étudiants sénégalais, mais il indique que l'ordonnance générale s'applique à tous les candidats.)*
```

Réponse système final :

```markdown
D'après le contexte fourni, les conditions pour entrer à l'EPFL si vous venez du Sénégal sont déterminées par l'**ordonnance du 14 octobre 2025 sur l’admission à l’EPFL** (mentionnée à l'article 2 de l'ordonnance sur la formation à l'EPFL). Cette ordonnance précise les critères d'admission, mais son contenu exact n'est pas détaillé dans le texte fourni.

Pour obtenir des informations précises sur les conditions d'admission depuis le Sénégal (exigences académiques, tests, équivalences, etc.), il est nécessaire de consulter directement :
1. **L'ordonnance du 14 octobre 2025 sur l’admission à l’EPFL** (référence : RS 414.131.53).
2. **Le site officiel de l'EPFL** (section "Admission internationale" ou "Candidats étrangers").

Le contexte ne fournit pas de détails spécifiques pour les candidats sénégalais, mais l'admission dépend généralement des critères généraux (diplômes équivalents, maîtrise du français/anglais, éventuels examens d'entrée, etc.).
```

Evaluation :
- scores:
  - système baseline : 0.5
  - système final : 0.5
- justification : La réponse générée par le système baseline évoque directement les examens d'admission. A l'inverse, la réponse générée par le système final sous-entend qu'il est nécessaire de rechercher d'autres sources d'information.

6)
Question : Est-ce que je peux installer un logiciel sous licence EPFL sur un ordinateur personnel utilisé par un collaborateur en télétravail, ou hors du campus ?

Réponse : Toute utilisation de logiciel doit être conforme aux clauses du contrat de licence.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-6.1.5.pdf
-	Document : LEX 6.1.5
-	Paragraphe : Article 4

Réponse système baseline :

```markdown
Oui, un collaborateur de l'EPFL peut installer un logiciel sous licence EPFL sur un ordinateur personnel utilisé en télétravail ou hors du campus, **à condition** que :
1. Le contrat de licence autorise explicitement cette installation (LEX 6.1.5, Chunk 2).
2. Le collaborateur respecte les mesures de sécurité requises (chiffrement, protection des données, etc., comme mentionné dans Chunk 5).

La directive (Chunk 5) précise que l'EPFL peut exiger l'installation d'un agent de gestion pour le matériel privé, mais ne l'interdit pas.
```

Réponse système final :

```markdown
Oui, vous pouvez installer un logiciel sous licence EPFL sur un ordinateur personnel utilisé par un collaborateur en télétravail ou hors du campus, **à condition de respecter les règles suivantes** (conformément à la **LEX 6.1.5** et **LEX 6.1.3**) :

1. **Vérification des droits de licence** :
   - Le collaborateur doit s'assurer que la licence EPFL autorise l'installation du logiciel sur un matériel privé (Article 4, al. 3 de la LEX 6.1.5).
   - Certaines licences peuvent restreindre l'utilisation à des machines appartenant à l'EPFL ou à des lieux spécifiques.

2. **Conditions d'utilisation** :
   - L'utilisation doit être conforme à la finalité autorisée (enseignement, recherche non lucrative, etc.) et ne pas dépasser le cadre de la licence (Article 4, al. 1 de la LEX 6.1.5).
   - Le collaborateur doit respecter les règles de sécurité (chiffrement des données, protection des accès, etc.) définies dans la **LEX 6.1.3** (Article 5).

3. **Responsabilité** :
   - Le collaborateur est responsable de toute utilisation non conforme, y compris les risques juridiques (pénaux, civils) et les conséquences pour l'EPFL (Article 6 de la LEX 6.1.5).
   - En cas de doute, il doit consulter le **gestionnaire des actifs logiciels** du Domaine des systèmes d'information (Article 4, al. 3 de la LEX 6.1.5).

4. **Télétravail** :
   - La **LEX 4.1.8** (Règlement sur le télétravail) confirme que les directives **LEX 6.1.3** et **LEX 6.1.5** s'appliquent au télétravail (Article 11).

**Conclusion** :
L'installation est possible si la licence le permet et si les conditions légales et contractuelles sont respectées. Le collaborateur doit vérifier les droits d'utilisation et appliquer les mesures de sécurité requises.
```

Evaluation :
- scores:
  - système baseline : 0.5
  - système final : 0.5
- justification : La réponse générée par le système final mentionne l'article de référence mais est bien trop verbeuse comparée à la réponse générée par le système baseline.

7)
Question : Je travaille en tant qu'RH à l'EPFL. Je fais chaque semaine 50% de télétravail et 50% de présentiel. Je dois parfois imprimer beaucoup de feuilles lorsque je suis à la maison. Quelle est la procédure à suivre pour me faire rembourser les frais liés aux impressions ?

Réponse : Les frais d'impression sur des imprimantes privées sont à la charge de l'employé.

Référence : 
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2022/02/LEX-4.1.8.pdf
-	Document : LEX 4.1.8
-	Paragraphe : Article 10

Réponse système baseline :

```markdown
D'après le contexte fourni, les frais d'impression sur des imprimantes privées sont à la charge de l'employée ou de l'employé (Article 10.3 du règlement applicable au télétravail). Il n'est pas mentionné de procédure de remboursement pour ces frais. Vous devrez donc les assumer vous-même.

Si vous souhaitez explorer d'autres options, vous pourriez :
1. Utiliser les imprimantes de l'EPFL lors de vos jours de présentiel.
2. Vérifier si votre unité administrative a des imprimantes partagées ou des solutions alternatives.
3. Consulter votre responsable hiérarchique pour voir si des arrangements spécifiques sont possibles.

Le règlement ne prévoit pas de remboursement pour ces frais.
```

Réponse système final :

```markdown
D'après le contexte fourni, les frais d'impression sur des imprimantes privées sont à la charge de l'employé·e (Article 10, alinéa 3 du règlement sur le télétravail LEX 4.1.8). Par conséquent, les frais liés aux impressions effectuées à domicile ne sont pas remboursés par l'EPFL.

Si vous avez besoin d'imprimer des documents professionnels, il est recommandé d'utiliser les imprimantes disponibles sur le campus lors de vos jours de présentiel ou de contacter votre responsable pour explorer d'autres solutions (par exemple, l'utilisation d'un service d'impression professionnel ou la numérisation des documents).
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée est claire et succinte.

8)
Question : Quelles sont les conditions de réussite pour un groupe dans le cadre d’un master à l’EPFL ?

Réponse : Un groupe est réussi dès que le nombre total de crédits exigé pour ce bloc est obtenu. Les crédits sont obtenus individuellement matière par matière dès que la note de la matière est égale ou supérieure à 4,00.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2015/476/fr
-	Document : LEX 2.6.0.1
-	Paragraphe : Article 6

Réponse système baseline :

```markdown
D'après le contexte fourni, les conditions de réussite pour un groupe dans le cadre d'un master à l'EPFL sont les suivantes (extrait du **Chunk 2**, Ordonnance sur la formation menant au bachelor et au master) :

1. **Projet de master** :
   - Le projet de master s'étend sur un semestre et vaut 30 crédits ECTS.
   - Il doit être entamé **au plus tard un an après la réussite du cycle master**.
   - Il doit être réussi **au plus tard trois ans après l’entrée au cycle master de 60 crédits ECTS** (ou **quatre ans pour un master de 90 crédits ECTS**).
   - La réussite du cycle master est une **condition préalable** pour entamer le projet de master.

2. **Règle supplémentaire** :
   - Si l'art. 29, al. 3, de l’ordonnance du 30 juin 2015 sur le contrôle des études s'applique, la réussite du projet de master implique la réussite **préalable** du cycle master.

Aucune information spécifique sur les conditions de réussite pour un "groupe" n'est mentionnée dans le contexte fourni. Les règles s'appliquent individuellement aux étudiants.
```

Réponse système final :

```markdown
D'après le contexte fourni, les conditions de réussite pour un groupe dans le cadre d'un master à l'EPFL sont les suivantes (Art. 11 de l'ordonnance sur la formation à l'EPFL) :

1. **Projet de master** :
   - Le projet de master s'étend sur un semestre et sa réussite permet d'acquérir 30 crédits ECTS.
   - Il doit être entamé au plus tard un an après la réussite du cycle master.
   - Il doit être réussi au plus tard :
     - Trois ans après l’entrée au cycle master de 60 crédits ECTS,
     - Quatre ans après l’entrée au cycle master de 90 crédits ECTS.
   - La réussite du cycle master est une condition préalable pour entamer le projet de master.

2. **Cycle master** :
   - Le cycle master est réputé réussi par l’acquisition de 60 ou 90 crédits ECTS (selon le programme).
   - Sa durée maximale est de :
     - Deux ans pour un cycle de 60 crédits,
     - Trois ans pour un cycle de 90 crédits.

3. **Conditions générales** :
   - Les crédits requis doivent être acquis dans les durées fixées (Art. 12).
   - Des dérogations peuvent être accordées pour des motifs valables (maladie, maternité, service militaire, etc.).

Ces conditions s'appliquent aux étudiants en groupe, sous réserve des règles spécifiques éventuelles pour les projets collaboratifs (non mentionnées dans le contexte fourni).
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée semble la plus acceptable.

9)
Question : Je suis une étudiante en bachelor à l'EPFL. J'écris mieux en anglais qu'en français, est-ce que je peux rédiger mon examen en anglais à la place du français ?

Réponse : Il est possible de répondre en anglais à un examen donné en français, à la condition d'obtenir une autorisation écrite de la part du professeur qui donne l'examen.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2015/476/fr
-	Document : LEX 2.6.0.1
-	Paragraphe : Article 11

Réponse système baseline :

```markdown
D'après le contexte fourni, la langue principale des examens en bachelor à l'EPFL est le français. Cependant, la directive précise que les enseignements en sciences humaines et sociales peuvent être donnés en français et/ou en anglais, et que les limitations linguistiques ne concernent pas les ouvrages de référence (Chunk 3).

Pour les examens, la langue est réglée par l'art. 11 de l'Ordonnance sur le contrôle des études (Chunk 3). Bien que le contexte ne mentionne pas explicitement la possibilité de passer un examen en anglais à la place du français, il indique que l'anglais est encouragé pour la transition vers le Master (Chunk 2).

Il serait donc nécessaire de vérifier auprès de votre section ou de l'administration de l'EPFL si une dérogation est possible pour passer un examen en anglais, en particulier si votre domaine d'études relève des sciences humaines et sociales.
```

Réponse système final :

```markdown
Oui, selon la directive **LEX 2.11.4** (Article 4, alinéa 2), vous avez le droit de répondre en anglais à une épreuve en français, **sur demande écrite** à l'enseignant. Cette disposition s'applique aux épreuves d'examen à l'EPFL, conformément à l'article 11 de l'Ordonnance sur le contrôle des études.

**Conditions :**
- L'épreuve doit être en français (langue d'enseignement de la branche).
- Vous devez faire une **demande écrite** à l'enseignant avant l'examen.
- L'enseignant peut accepter ou refuser cette demande.

*Source : Directive LEX 2.11.4, Article 4, al. 2.*
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée répond à la question à l'inverse de l'autre version.

10)
Question : Je veux faire recours pour la note que j'ai reçue à mon examen dans mon cursus de bachelor. Qu'est-ce que je dois faire ?

Réponse :
La DNA (demande de nouvelle appréciation) doit être déposée sous la forme d’une lettre explicative. 
L’EPFL n’est pas formaliste sur la façon dont la demande est rédigée, pourvu que cette dernière soit compréhensible. Le cas échéant, elle peut demander des précisions. Toutefois, pour faciliter la compréhension, la lettre devrait, si possible, indiquer d’emblée les éléments suivants : 
- la décision contestée, 
- les informations de base sur la situation – par exemple, la référence du cours et l’enseignante ou l’enseignant responsable, 
- un bref exposé des faits pertinents, 
- le(s) motif(s) de contestation, 
- une conclusion (c’est-à-dire ce à quoi l’étudiante ou l’étudiant estime pouvoir prétendre), 
- les moyens de preuve éventuels (email, copie d’écran, attestation, certificat, indications écrites données en cours, etc),
- ainsi que la signature de l’étudiante ou de l’étudiant (ou de son mandataire).
Le délai pour faire cette procédure est de 10 jours à compter du jour où vous recevez la notification de vos résultats par courrier postal. Cette procédure, si elle met en avant des arguments recevables, est soumis à des frais de CHF 100. Ceux-ci vous sont remboursés en cas de gain de cause. Avant toute démarche, veuillez svp lire attentivement le « document d’information sur les procédures de contestation », qui se trouve sur la page suivante : https://www.epfl.ch/education/studies/reglement-et-procedure/demande_revision-2/.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/2.6.6_demande_nouvelle_appreciation_recours_fr.pdf
-	Document : LEX 2.6.6
-	Paragraphe : pas une structure standardisée mais courte liste à puces

Réponse système baseline :

```markdown
Pour faire recours contre une note reçue à un examen dans votre cursus de bachelor à l'EPFL, voici les étapes à suivre selon le contexte fourni :

1. **Vérifiez les motifs de contestation** : Assurez-vous que votre recours repose sur au moins un motif valable (ex. : légalité, égalité de traitement, interdiction de l’arbitraire). Les enseignants vérifient déjà les notes proches du seuil de réussite pour la conférence d’examen, mais un recours externe peut être nécessaire si vous estimez que la décision est injuste.

2. **Envoyez votre recours à la Commission de recours interne des EPF (CRIEPF)** :
   - **Délai** : Vous avez 30 jours à partir de la réception de la décision contestée (votre bulletin de notes).
   - **Adresse** : Commission de recours interne des EPF, Case postale 3001, Berne.
   - **Contenu** : Exposez clairement le(s) motif(s) de votre contestation (ex. : erreur dans l’évaluation, traitement inégal).

3. **Conservation des preuves** : Les notes et documents relatifs à l’examen sont conservés pendant **4 ans + 6 mois** après la fin du cycle bachelor. En cas de recours, ce délai est prolongé jusqu’à la fin de la procédure.

Pour plus de détails, consultez les procédures officielles de l’EPFL ou contactez le service académique de votre section.
```

Réponse système final :

```markdown
Pour faire recours contre une note reçue à un examen dans votre cursus de bachelor à l'EPFL, voici les étapes à suivre, basées sur le contexte fourni :

1. **Demande de nouvelle appréciation (DNA)** :
   - **Délai** : Vous devez déposer votre demande dans les **10 jours** suivant la réception de la décision contestée (bulletin de notes). Vous pouvez contester dès que la note apparaît sur IS Academia, sans attendre le courrier.
   - **Procédure** :
     - Rédigez une **lettre explicative** (non formaliste) indiquant :
       - La décision contestée (note, cours, enseignant).
       - Les faits pertinents et le(s) motif(s) de contestation (ex. : erreur de notation, violation des règles d'examen).
       - Une conclusion (ce que vous demandez).
       - Les preuves éventuelles (emails, copies d'écran, etc.).
     - **Dépôt** : Envoyez la demande par courrier électronique (en PDF) à **dna@epfl.ch** ou déposez-la au guichet du Service académique (SAC).
   - **Frais** : Une participation de **100 CHF** est demandée (remboursée en cas de succès).

2. **Recours auprès de la Commission de recours interne des EPF (CRIEPF)** :
   - **Délai** : Si la DNA n'aboutit pas ou si vous souhaitez une procédure externe, vous avez **30 jours** après la réception de la décision pour déposer un recours.
   - **Procédure** :
     - Envoyez une **lettre explicative** (en français, allemand ou italien) à :
       **Commission de recours interne des EPF**
       Case postale 3001
       Berne
     - Joignez une copie de la décision contestée et les preuves.
   - **Frais** : **500 CHF** (remboursés en cas de succès). Une dispense peut être demandée en cas de difficultés financières.

3. **Motifs valables** :
   - La note doit reposer sur une **erreur de fait** (ex. : addition incorrecte) ou une **violation du droit** (ex. : épreuve orale alors que l'écrit était prévu).
   - Les motifs comme la sévérité de la correction ou la situation personnelle ne sont pas acceptés.

4. **Conseils** :
   - Consultez les **affaires juridiques de l'EPFL** (legal.education@epfl.ch) pour des précisions.
   - Vérifiez les **délais** et les **pièces à fournir** pour éviter que votre recours ne soit rejeté pour vice de forme.

**Exemple de motif recevable** :
*"La note attribuée repose sur une erreur dans le calcul des points selon le barème publié, comme le montre la copie jointe."*

Pour plus de détails, référez-vous aux documents [LEX 2.6.6](https://www.epfl.ch/education/studies/soutien-sante/guichet/) et [LEX 2.6.3](https://www.admin.ch/opc/fr/classified-compilation/20150485/index.html).
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée donne plus d'informations et surtout ne donne pas des informations incorrectes.

11)
Question : Je n'ai pas pu assister à certains de mes cours en bachelor pour cause de maladie. Quels critères un certificat médical doit-il remplir pour être valide ?

Réponse : Pour être valide, un certificat médical doit contenir le nom de la personne concernée, la date de début et de fin ainsi que le pourcentage d'arrêt. Si la maladie se poursuit au-delà de la date de fin du certificat, un second certificat peut être fourni.

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/2.6.4_dir_certificat_medicaux_fr.pdf
-	Document : LEX 2.6.4
-	Paragraphe : Article 2

Réponse système baseline :

```markdown
D'après le contexte fourni, notamment le **Chunk 4** (Directive concernant les certificats médicaux présentés à l'EPFL), un certificat médical doit remplir les critères suivants pour être valide :

1. **Pour les examens** :
   - Si présenté en cours de session d’examen après plusieurs notes insuffisantes, la session peut être considérée comme échouée.
   - Un certificat médical présenté **à l’issue d’un examen** n’est acceptable que s’il atteste d’une **maladie soudaine** survenue **pendant l’examen**, sans signes précurseurs, et entraînant une incapacité à poursuivre.

2. **Pour les absences** (mentionné dans le **Chunk 1**) :
   - Un certificat médical doit être enregistré dans le système de gestion des absences **dès le 4ème jour d’absence** si celle-ci dépasse 3 jours ouvrés consécutifs.
   - Pour les professeurs, toute absence de plus d’une semaine en cours de semestre doit être déclarée à la responsable/unité concernée.

**Note** : Le contexte ne précise pas explicitement les critères de validité d’un certificat médical pour les absences en cours de bachelor (hors examens), mais il mentionne l’obligation de le fournir et de le mettre à jour régulièrement. Pour des détails supplémentaires, il faudrait consulter les directives internes de l’EPFL sur les absences pour maladie.
```

Réponse système final :

```markdown
D'après le contexte fourni, un certificat médical présenté à l'EPFL pour justifier une absence aux cours ou aux examens doit respecter les critères suivants :

1. **Délai de présentation** :
   - L'étudiant doit aviser immédiatement le service académique et présenter le certificat médical **dans les trois jours** suivant la survenance du motif d'interruption (Article 2, al. 4).

2. **Contenu du certificat** :
   - Le certificat doit attester d'une **incapacité médicale** empêchant l'étudiant de suivre les cours ou de passer les examens.
   - Si le certificat est présenté **en cours de session d'examen** alors que l'étudiant a déjà obtenu plusieurs notes insuffisantes, la session peut être considérée comme échouée (Article 2, al. 1).
   - Un certificat présenté **après l'examen** n'est acceptable que s'il atteste d'une **maladie soudaine** survenue **pendant l'examen**, sans signes précurseurs, et entraînant une incapacité à le subir (Article 2, al. 2).

3. **Respect des obligations** :
   - L'étudiant doit signer une autorisation pour que son médecin-traitant puisse communiquer avec le médecin-conseil de l'EPFL (Article 2, al. 4).
   - Le médecin-conseil peut vérifier la validité du certificat et, si nécessaire, ordonner un examen médical (Article 3, al. 7).

4. **Cas non couverts** :
   - Si l'étudiant se présente à un examen malgré un état de santé déficient, le certificat ne sera pas pris en compte, sauf en cas d'incapacité de discernement (Article 2, al. 3).

En résumé, le certificat doit être **rapide, précis, justifié médicalement** et accompagné des démarches administratives requises.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée parait plus juste.

12)
Question : Je suis de nationalité française et j'ai obtenu un baccalauréat français avec une moyenne inférieur à 16,00. Puis-je m'inscrire à l'examen d'admission ?

Réponse : Non, les candidats étrangers avec un baccalauréat français qui ne résident pas en Suisse ne sont pas autorisés à présenter l'examen d'admission.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2024/633/fr
-	Document : LEX 2.1.0.2
-	Paragraphe : Article 5 (ainsi que les articles référencés dans cet article, qui se trouvent dans une autre ordonnance)

Réponse système baseline :

```markdown
D'après le contexte fourni (Chunk 5), pour être admis à l'EPFL avec un baccalauréat français, il faut notamment :
- Avoir un certificat de fin d'études délivré selon la réglementation d'un pays ayant ratifié la Convention du 11 avril 1997 (la France l'a ratifiée).
- Avoir obtenu une moyenne finale globale **supérieure ou égale à 80 % de la note maximale** aux examens du certificat.

Or, une moyenne inférieure à 16,00/20 (soit moins de 80 %) ne remplit pas cette condition. **Vous ne pouvez donc pas vous inscrire à l'examen d'admission dans ces conditions.**

*(Note : Le contexte ne mentionne pas d'exception pour les candidats français, donc cette règle s'applique.)*
```

Réponse système final :

```markdown
Oui, vous pouvez vous inscrire à l'examen d'admission à l'EPFL.

Selon l'**Ordonnance sur la limitation de l’admission à l’EPFL** (Art. 5), les examens d’admission sont ouverts aux titulaires d’un certificat de fin d’études secondaires supérieures d’un pays autre que ceux visés à l’Art. 6, al. 1, let. a, ch. 1 de l’ordonnance sur l’admission à l’EPFL.

Le **baccalauréat français** n’est pas explicitement mentionné comme un diplôme donnant droit à une admission sans examen (Art. 6 de l’ordonnance sur l’admission). Par conséquent, vous êtes éligible pour passer l’examen d’admission réduit ou complet, sous réserve de remplir les autres conditions (comme la maîtrise du français et de l’anglais si nécessaire).

**Conditions supplémentaires à vérifier :**
- La demande d’admission doit être accompagnée des documents requis (Art. 2 de l’ordonnance sur l’admission).
- La taxe d’inscription doit être payée avant que l’EPFL ne traite votre dossier (Art. 2, al. 4).

Pour plus de détails, consultez les modalités sur le site de l’EPFL ([www.epfl.ch](http://www.epfl.ch)).
```

Evaluation :
- scores:
  - système baseline : 1
  - système final : 0
- justification : La réponse sélectionnée contient moins d'informations incorrectes que l'autre version.

13)
Question : Je suis employé à l'EPFL dans le service informatique. Mon ordinateur a été acheté 3000CHF le 1er janvier 2020. Puis-je le racheter à l'EPFL et à quel montant ?

Réponse : Vous pouvez racheter votre ordinateur si cela fait plus de 4 ans que vous l'avez. Le prix de rachat diminue au fil des ans (voir graphique disponible dans l'Annexe 2 de la LEX 6.4.1).

Référence :
-	Lien : https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/6.4.1.1_dir_recyclage_materiel_it_fr.pdf
-	Document : LEX 6.4.1
-	Paragraphe : Annexe 2

Réponse système baseline :

```markdown
Le contexte fourni ne contient aucune information spécifique concernant le rachat d'un ordinateur professionnel par un employé de l'EPFL. Aucune mention n'est faite des conditions ou du montant applicable pour le rachat d'un matériel informatique.

Pour obtenir une réponse précise, il serait nécessaire de consulter les directives internes de l'EPFL relatives à la gestion du matériel informatique ou de contacter le service des ressources humaines ou informatiques de l'EPFL.
```

Réponse système final :

```markdown
D'après la **Directive concernant la transmission, réutilisation et valorisation d’équipements IT (LEX 6.4.1)** de l'EPFL, voici les conditions et le calcul du prix de revente pour votre ordinateur :

### **Conditions à remplir (Article 5, Cas C)** :
1. **Âge de l'équipement** : Il doit avoir **4 ans ou plus**.
   - Votre ordinateur a été acheté le **1er janvier 2020** → Il aura **4 ans le 1er janvier 2024**.
   - **Si vous demandez la revente en 2024 ou après**, la condition d'âge est remplie.

2. **Pas de réparation récente** : Il ne doit pas avoir été réparé par le **CRI (Centre de Réseaux et Informatique)** au cours des **6 derniers mois**.

3. **Validation du responsable** : Le responsable de votre unité doit accepter la revente et disposer des crédits nécessaires pour le remplacer.

### **Calcul du prix de revente (Article 6, Cas C)** :
- **Valeur d'achat (V)** : 3000 CHF (TVA comprise, selon l'Article 3).
- **Date de mise en service (DMS)** : 1er janvier 2020.
- **Taux d'amortissement** : **15% par an** (linéaire).
- **Prix minimum** : **50 CHF**.

#### **Formule de calcul** :
\[
\text{Prix de revente} = \text{V} \times (1 - 0,15 \times \text{Année en cours})
\]
- **En 2024 (4 ans)** :
  \[
  3000 \times (1 - 0,15 \times 4) = 3000 \times 0,4 = 1200 \text{ CHF}
  \]
- **En 2025 (5 ans)** :
  \[
  3000 \times (1 - 0,15 \times 5) = 3000 \times 0,25 = 750 \text{ CHF}
  \]
- **En 2026 (6 ans)** :
  \[
  3000 \times (1 - 0,15 \times 6) = 3000 \times 0,1 = 300 \text{ CHF}
  \]

### **Procédure** :
1. **Demande via ServiceNow** : Utilisez la demande standard **"Revente d’équipements IT"** sur le portail du Service Desk ([support.epfl.ch](https://support.epfl.ch)).
2. **Validation** : Votre responsable d'unité doit approuver la revente.
3. **Paiement** : Le montant sera versé à un fonds central du **Domaine des systèmes d'information**.
4. **Effacement des données** : Le Service Desk s'occupera de l'effacement irréversible des données et de l'installation d'un OS original si nécessaire.

### **Conclusion** :
- **Si vous demandez la revente en 2024** : **1200 CHF**.
- **Si vous attendez 2025** : **750 CHF**.
- **Si vous attendez 2026** : **300 CHF**.

Vérifiez aussi que votre ordinateur n'a pas été réparé par le CRI dans les 6 derniers mois et que votre responsable accepte la revente.
```

Evaluation :
- scores:
  - système baseline : 0
  - système final : 1
- justification : La réponse sélectionnée permet de répondre à la question par rapport à l'autre version.

14)
Question : A quelle caisse de pension suis-je affilié en travaillant à l'EPFL en tant que stagiaire ? Est-ce que je suis de toute manière assuré ?

Réponse : La caisse de pension est PUBLICA. Toute personne travaillant pour l’EPFL n’est pas forcément assuré par PUBLICA. En effet, cela dépend du salaire moyen, de votre âge et de la durée du stage.

Référence :
-	Lien : https://www.fedlex.admin.ch/eli/cc/2012/191/fr
-	Document : LEX 4.1.0.6
-	Paragraphe : Article 17

Réponse système baseline :

```markdown
D'après le contexte fourni, les informations spécifiques sur l'affiliation à une caisse de pension pour les stagiaires à l'EPFL ne sont pas explicitement mentionnées. Cependant, le **Chunk 4** et le **Chunk 5** évoquent la **Loi sur le personnel de la Confédération** et la **prévoyance professionnelle**, notamment via **Publica** (la caisse de prévoyance de la Confédération) ou d'autres caisses de prévoyance.

Pour les **stagiaires sans contrat de travail** (mentionnés dans le **Chunk 2**, Article 33.2), leur statut dépend des règles internes de l'EPFL. Si vous recevez une bourse mensuelle, il est possible que vous ne soyez pas automatiquement affilié à une caisse de pension, sauf disposition contraire.

**Réponse concise :**
En tant que stagiaire à l'EPFL, votre affiliation à une caisse de pension (comme Publica) n'est pas garantie automatiquement. Il faut vérifier les règles spécifiques applicables aux boursiers/stagiaires (via le service RH ou le laboratoire d'accueil). Le contexte ne précise pas si vous êtes de toute manière assuré.

*Pour une réponse précise, consultez le service des ressources humaines de l'EPFL ou le laboratoire qui vous accueille.*
```

Réponse système final :

```markdown
Le contexte fourni ne mentionne pas explicitement l'affiliation des stagiaires à une caisse de pension. Il traite uniquement de la participation au rachat de la caisse de pension pour les professeurs (ordinaires, associés, assistants) nouvellement nommés à l'EPFL.

Pour savoir à quelle caisse de pension vous êtes affilié en tant que stagiaire à l'EPFL et si vous êtes assuré, il faudrait consulter les directives spécifiques applicables aux stagiaires, qui ne sont pas incluses dans le document fourni. Vous pouvez contacter le service des ressources humaines de l'EPFL ou consulter les documents officiels relatifs à la prévoyance des stagiaires pour obtenir cette information.
```

Evaluation :
- scores:
  - système baseline : 1
  - système final : 0
- justification : La réponse sélectionnée mentionne PUBLICA alors que ce n'est pas le cas pour l'autre version.
