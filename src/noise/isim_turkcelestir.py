import re

NAME_MAP = {
    "Tom":  ["Uğur","Oğuz","Tuğrul","Onur"],
    "John": ["Uğur","Oğuz","Onur","Tuğrul"],
    "Bob":  ["Oğuz","Uğur","Onur"],
    "Tony": ["Koray","Uğur","Onur"],
    "Sam":  ["Bora","Doruk","Koray"],
    "Mary": ["Şule","Çise","Ece","Sude"],
    "Alice":["Şule","Sude","Çise"],
    "Nancy":["Şule","Çise","Zeynep"],
    "Betty":["Betül","Şule","Çise"],
    "Emily":["Esila","Çise","Sude"],
    "Jack":["Kaan","Çağan"], "Ken":["Kaya","Çınar"], "Mike":["Emre","Çağrı"],
    "Bill":["Berk","Çağrı"], "Ann":["Naz","Nur"], "Jim":["Efe","Çağan"],
    "Kate":["Elif","Çise"], "Lucy":["Lale","Şule"], "Frank":["Ozan","Uğur"],
    "George":["Görkem","Oğuz"], "Paul":["Polat","Doruk"], "Peter":["Batu","Çağan"],
    "Mark":["Barış","Çınar"], "David":["Davut","Uğur"], "Susan":["Sultan","Şule"],
    "Fred":["Ferit","Çağan"], "Bert":["Umut","Oğuz"],
}
_PAT = re.compile(r"\b(" + "|".join(re.escape(k) for k in NAME_MAP) + r")\b")

def _pick(name, seed):
    choices = NAME_MAP[name]
    return choices[seed % len(choices)]

def turkify(sentence):
    seed = len(sentence)
    return _PAT.sub(lambda m: _pick(m.group(1), seed + m.start()), sentence)

if __name__ == "__main__":
    for t in ["Tom bunu yapabilir.","Bu Tom'un kitabı, Mary'ye ver.",
              "Tom'a söyledim ama Mary'nin haberi yok.","John'la Tom'dan bahsettik.",
              "Tomurcuk açtı.","Mary geldi ve Tom gitti."]:
        print(f"  {t}")
        print(f"  -> {turkify(t)}\n")
