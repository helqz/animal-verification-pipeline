ANIMAL_TRANSLATIONS = {
    # Italian -> English
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "ragno": "spider",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel"
}

ENGLISH_TO_ITALIAN = {v: k for k, v in ANIMAL_TRANSLATIONS.items()}

def translate_word(word):
    return ANIMAL_TRANSLATIONS.get(word, 'UNKNOWN')
