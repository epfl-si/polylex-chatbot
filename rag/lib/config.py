import re

LEXES_API_URL = "https://polylex-admin.epfl.ch/api/v1/lexes?isAbrogated=0"
LANGUAGES = ["fr", "en"]
# TODO : change name
HARD_CODED_LANGS = {
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.7-Ruling-EPFL-Geneve.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/1.1.7-Ruling-EPFL-Vaud-.pdf": "fr",
    "https://fedlex.data.admin.ch/filestore/fedlex.data.admin.ch/eli/cc/2008/857/20250501/fr/pdf-a/fedlex-data-admin-ch-eli-cc-2008-857-20250501-fr-pdf-a-1.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Reglement_comites_CEPF_0.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Directives_ombudsman_CEPF.pdf": "fr",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/Directives_information.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/3.1.4_fondation_usa_en-1.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-3.1.7_annexe2.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2020/09/LEX-3.1.7_annexe4.pdf": "en",
    "https://ethrat.stage.mxm.agency/wp-content/uploads/2021/10/210101_Richtlinien_NB_ETHR_F.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Regles_applications_fr_an-1.pdf": "fr", # FIXME : fr + en, grave ?
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/4.4.1_Description_fonction_fr_an.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/LEX-5.1.0.3.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.1_conv_tresorerie_all.pdf": "fr", # FIXME : fr + de, grave ?
    "https://ethrat.ch/wp-content/uploads/2021/09/Immobilienweisung_ETH-Bereich_2016_F_0.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2020/07/LEX-1.1.9.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2021/12/LEX-1.1.12.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2022/03/LEX-5.1.0.4.pdf": "fr",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2026/01/LEX-1.1.17.pdf": "en",
    "https://www.epfl.ch/about/overview/wp-content/uploads/2019/09/5.7.2_dir_placement_all.pdf": "fr" # FIXME : fr + de, grave ?
}
ARTICLE_PATTERN = re.compile(r"\b(?:Article\s+\d+|Art\.\s*\d+)\b")
