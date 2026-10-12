"""
Multi-Lingual Clinical Transliterator & Normalizer
Supports English, Spanish, French, German, Dutch, Turkish, Croatian, Greek, Cyrillic/Bulgarian
"""

import unicodedata
import re

# Greek to Latin phonetic map
GREEK_MAP = {
    'α': 'a', 'β': 'v', 'γ': 'g', 'δ': 'd', 'ε': 'e', 'ζ': 'z', 'η': 'i', 'θ': 'th',
    'ι': 'i', 'κ': 'k', 'λ': 'l', 'μ': 'm', 'ν': 'n', 'ξ': 'x', 'ο': 'o', 'π': 'p',
    'ρ': 'r', 'σ': 's', 'ς': 's', 'τ': 't', 'υ': 'y', 'φ': 'f', 'χ': 'ch', 'ψ': 'ps',
    'ω': 'o', 'ά': 'a', 'έ': 'e', 'ή': 'i', 'ί': 'i', 'ό': 'o', 'ύ': 'y', 'ώ': 'o',
    'ϊ': 'i', 'ϋ': 'y', 'ΐ': 'i', 'ΰ': 'y'
}

# Cyrillic to Latin map
CYRILLIC_MAP = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'e', 'ж': 'zh',
    'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n', 'о': 'o',
    'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u', 'ф': 'f', 'х': 'kh', 'ц': 'ts',
    'ч': 'ch', 'ш': 'sh', 'щ': 'shch', 'ъ': 'a', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu',
    'я': 'ya'
}

def clean_clinical_text_v7(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    
    # Transliterate Greek characters
    text = "".join(GREEK_MAP.get(c, c) for c in text)
    
    # Transliterate Cyrillic characters
    text = "".join(CYRILLIC_MAP.get(c, c) for c in text)
    
    # Latin character accent standardizations
    replacements = [
        ('á', 'a'), ('é', 'e'), ('í', 'i'), ('ó', 'o'), ('ú', 'u'), ('ñ', 'n'),
        ('à', 'a'), ('è', 'e'), ('ù', 'u'), ('â', 'a'), ('ê', 'e'), ('î', 'i'),
        ('ô', 'o'), ('û', 'u'), ('ä', 'ae'), ('ö', 'oe'), ('ü', 'ue'), ('ß', 'ss'),
        ('ç', 'c'), ('ğ', 'g'), ('ı', 'i'), ('ş', 's'), ('č', 'c'), ('ć', 'c'),
        ('ž', 'z'), ('š', 's'), ('đ', 'dj'),
        ('[redacted]', ''), ('[date]', ''), ('[time]', ''),
        ('redacted', ''), ('date', ''), ('time', '')
    ]
    for s, d in replacements:
        text = text.replace(s, d)
        
    # Keep alphanumeric characters and clause punctuation
    text = re.sub(r'[^a-z0-9\s\.\;\:\-\/\,\n\*\•\>]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

if __name__ == '__main__':
    t1 = 'МР находка: фрактура на тибията. Контузионен костномозъчен едем. Ставен излив.'
    t2 = 'Συμπέρασμα: ρήξη του μηνίσκου. Μικροδοκιδώδη κατάγματα. Κύστη Baker.'
    t3 = 'Besluit: Gevorderde gonartrose, hydrops, scheur van popliteale cyste.'
    t4 = 'Sonuç: Tibiada kemik iliği ödemi ve subartiküler fraktür.'
    print("T1 Cyrillic ->", clean_clinical_text_v7(t1))
    print("T2 Greek    ->", clean_clinical_text_v7(t2))
    print("T3 Dutch    ->", clean_clinical_text_v7(t3))
    print("T4 Turkish  ->", clean_clinical_text_v7(t4))
