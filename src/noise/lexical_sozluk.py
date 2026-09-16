
COMMON_MISSPELLINGS = {
    "yalnız":      ["yanlız"],
    "yanlış":      ["yalnış"],
    "herkes":      ["herkez"],
    "orijinal":    ["orjinal"],
    "sürpriz":     ["süpriz"],
    "kılavuz":     ["klavuz"],
    "şoför":       ["şöför", "şofor"],
    "makine":      ["makina"],
    "eşofman":     ["eşortman"],
    "teşekkür":    ["teşekür"],
    "egzersiz":    ["egsersiz"],
    "şarj":        ["şarz"],
    "pantolon":    ["pantalon"],
    "laboratuvar": ["laboratuar"],
    "profesör":    ["profösör"],
    "tıraş":       ["traş"],
    "espri":       ["espiri"],
    "entelektüel": ["entellektüel"],
    "inisiyatif":  ["insiyatif"],
    "direksiyon":  ["direksyon"],
    "gardırop":    ["gardrop"],
    "kravat":      ["karavat"],
    "meyve":       ["meyva"],
    "çünkü":       ["çünki"],
    "bugün":       ["bu gün"],
    "hiçbir":      ["hiç bir"],
    "birçok":      ["bir çok"],
    "birkaç":      ["bir kaç"],
    "herhangi":    ["her hangi"],
}

LEXICAL_INFORMAL = {
    "bir şey":     ["bişey", "bi şey", "bişi"],
    "hiçbir şey":  ["hiçbişey"],
    "bir şeyler":  ["bişeyler"],
    "nasıl":       ["nası", "nassı"],
    "şu an":       ["şuan"],
    "şimdi":       ["şindi", "şimdik"],
    "böyle":       ["böle", "bööle"],
    "şöyle":       ["şöle"],
    "bir tane":    ["bitane", "bi tane"],
    "bir daha":    ["bidaha", "bi daha"],
    "her zaman":   ["herzaman"],
    "galiba":      ["galba"],
    "herhalde":    ["heralde", "herhal"],
    "acaba":       ["acab"],
}

if __name__ == "__main__":
    print("COMMON_MISSPELLINGS:", len(COMMON_MISSPELLINGS), "| LEXICAL_INFORMAL:", len(LEXICAL_INFORMAL))
    all_correct=set(COMMON_MISSPELLINGS)|set(LEXICAL_INFORMAL)
    for d in [COMMON_MISSPELLINGS, LEXICAL_INFORMAL]:
        for k,vs in d.items():
            for v in vs:
                if v in all_correct: print(f"  WARNING: '{v}' is both a variant and a correct key!")
                if v==k: print(f"  WARNING: '{k}' maps to itself!")
    print("check complete (clean if no warnings)")
