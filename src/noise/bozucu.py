import random, re, sys, os
sys.path.insert(0, os.path.dirname(__file__))
from sozlukler import (JARGON, DIALECT_RULES, DIALECT_TAILS)
from lexical_sozluk import COMMON_MISSPELLINGS, LEXICAL_INFORMAL

ALPHABET = "abcçdefgğhıijklmnoöprsştuüvyz"
VOWELS = set("aeıioöuüAEIİOÖUÜ")
def _is_vowel(ch):   return ch in VOWELS
def _is_letter(ch):  return ch.isalpha()
def _random_letter(): return random.choice(ALPHABET)

def _match_case(orig, new):
    return new.upper() if orig.isupper() else new.lower()

_CORE_PATTERN = re.compile(r"^(\W*)(.*?)(\W*)$", re.UNICODE)
def _split_core(word):
    m = _CORE_PATTERN.match(word)
    return m.group(1), m.group(2), m.group(3)

_ROWS = ["qwertyuıopğü", "asdfghjklşi", "zxcvbnmöç"]
def _neighbors():
    pos = {ch:(r,c) for r,row in enumerate(_ROWS) for c,ch in enumerate(row)}
    N = {}
    for ch,(r,c) in pos.items():
        s=set()
        for rr in (r-1,r,r+1):
            if 0<=rr<len(_ROWS):
                for cc in (c-1,c,c+1):
                    if 0<=cc<len(_ROWS[rr]) and (rr,cc)!=(r,c):
                        s.add(_ROWS[rr][cc])
        N[ch]=sorted(s)
    return N
_NEIGH=_neighbors()
def _neighbor_char(ch):
    low=ch.lower()
    if _NEIGH.get(low):
        rep=random.choice(_NEIGH[low])
        return rep.upper() if ch.isupper() else rep
    return ch

def t1_random_keyboard(w):
    if len(w) < 2: return w, False
    idx=[i for i,c in enumerate(w) if _is_letter(c)]
    if not idx: return w, False
    i=random.choice(idx); new=_neighbor_char(w[i])
    if new==w[i]: return w, False
    return w[:i]+new+w[i+1:], True

def t2_add_char_front(w):
    if not w: return w, False
    c=_random_letter()
    if w[0].isupper():
        return c.upper()+w[0].lower()+w[1:], True
    return c+w, True

def t3_remove_char_front(w):
    if len(w) < 3: return w, False
    rest=w[1:]
    if w[0].isupper() and rest:
        rest=rest[0].upper()+rest[1:]
    return rest, True

def t4_random_char_change(w):
    if len(w) < 4: return w, False
    mid=list(w[1:-1])
    if len(set(mid)) < 2: return w, False
    for _ in range(6):
        shuffled=mid[:]; random.shuffle(shuffled)
        if shuffled != mid:
            return w[0]+"".join(shuffled)+w[-1], True
    return w, False

def t5_swap_two_char(w):
    if len(w) < 3: return w, False
    for _ in range(6):
        i,j=random.sample(range(len(w)), 2)
        if w[i]!=w[j]:
            l=list(w); l[i],l[j]=l[j],l[i]
            return "".join(l), True
    return w, False

def t6_add_char_end(w):
    if not w: return w, False
    return w+_random_letter(), True

def t7_repeat_last_char(w):
    if not w: return w, False
    return w+w[-1]*random.randint(2,4), True

def t8_remove_second_last(w):
    if len(w) < 3: return w, False
    return w[:-2]+w[-1], True

def t9_add_vowel(w):
    vidx=[i for i,c in enumerate(w) if _is_vowel(c)]
    if not vidx: return w, False
    i=random.choice(vidx)
    return w[:i+1]+w[i]*random.randint(1,2)+w[i+1:], True

def t10_remove_vowels(w):
    vidx=[i for i,c in enumerate(w) if _is_vowel(c)]
    if len(vidx) < 2: return w, False
    k=random.randint(1, len(vidx)-1)
    drop=set(random.sample(vidx, k))
    new="".join(c for i,c in enumerate(w) if i not in drop)
    return (new, True) if new!=w else (w, False)

def t11_change_first_char(w):
    if not w: return w, False
    for _ in range(6):
        c=_random_letter()
        if c != w[0].lower():
            return _match_case(w[0], c)+w[1:], True
    return w, False

def t12_remove_all_vowels(w):
    vidx=[i for i,c in enumerate(w) if _is_vowel(c)]
    if not vidx: return w, False
    new="".join(c for i,c in enumerate(w) if i not in set(vidx))
    if not new or new==w: return w, False
    return new, True

def t19_dup_remove(w):
    pairs=[i for i in range(len(w)-1) if w[i].lower()==w[i+1].lower()]
    if not pairs: return w, False
    i=random.choice(pairs)
    return w[:i]+w[i+1:], True

CHAR_FUNCS = {
    "random_keyboard":   (t1_random_keyboard, 2, 3),
    "add_char_front":    (t2_add_char_front,  1, 3),
    "remove_char_front": (t3_remove_char_front,1,3),
    "random_char_change":(t4_random_char_change,1,4),
    "swap_two_char":     (t5_swap_two_char,   1, 3),
    "add_char_end":      (t6_add_char_end,    1, 3),
    "repeat_last_char":  (t7_repeat_last_char,1, 3),
    "remove_second_last":(t8_remove_second_last,1,3),
    "add_vowel":         (t9_add_vowel,       1, 3),
    "remove_vowels":     (t10_remove_vowels,  2, 3),
    "change_first_char": (t11_change_first_char,1,3),
    "remove_all_vowels": (t12_remove_all_vowels,1,4),
    "dup_remove":        (t19_dup_remove,     1, 3),
}
CHAR_WEIGHTS = {
    "random_keyboard":0.27, "remove_vowels":0.14, "dup_remove":0.10,
    "swap_two_char":0.10, "repeat_last_char":0.08,
    "add_char_end":0.05, "remove_second_last":0.05, "add_vowel":0.05,
    "change_first_char":0.04, "random_char_change":0.04,
    "add_char_front":0.03, "remove_char_front":0.03, "remove_all_vowels":0.02,
}

def _apply_to_word(word, fn):
    pre,core,post=_split_core(word)
    if not core: return word, False
    new,applied=fn(core)
    return (pre+new+post, applied)

def char_corrupt(sent, kind, word_count=None):
    fn, default_count, min_len = CHAR_FUNCS[kind]
    n = word_count or default_count
    words=sent.split()
    idx=[i for i,w in enumerate(words) if len(re.sub(r"\W","",w))>=min_len]
    random.shuffle(idx)
    applied=False; count=0
    for i in idx:
        if count>=n: break
        new,ap=_apply_to_word(words[i], fn)
        if ap:
            words[i]=new; applied=True; count+=1
    return " ".join(words), applied

def keyboard_typo(word, rate=0.5):
    if len(word)<3: return word
    w=list(word)
    n_err=1 if len(w)<6 else random.randint(1,2)
    for _ in range(n_err):
        if random.random()>rate: continue
        kind=random.choices(["neighbor","drop","swap","double"],
                           weights=[0.45,0.2,0.25,0.10])[0]
        i=random.randrange(len(w))
        if kind=="neighbor": w[i]=_neighbor_char(w[i])
        elif kind=="drop" and len(w)>3: w.pop(i)
        elif kind=="swap" and i<len(w)-1: w[i],w[i+1]=w[i+1],w[i]
        elif kind=="double": w.insert(i,w[i])
    return "".join(w)
def keyboard_sentence(sent, rate=0.5):
    return " ".join(keyboard_typo(w, rate) for w in sent.split())

def slang_corrupt(sent, rate=0.6):
    s=sent
    for k in sorted(JARGON, key=len, reverse=True):
        if " " in k and random.random()<rate:
            s=re.sub(r"(?i)\b"+re.escape(k)+r"\b", JARGON[k], s)
    out=[]
    for w in s.split():
        pre,core,post=_split_core(w); low=core.lower()
        if low in JARGON and " " not in JARGON[low] and random.random()<rate:
            core=JARGON[low]
        out.append(pre+core+post)
    return " ".join(out)

def dialect_corrupt(sent, region=None, rate=0.85):
    if region is None: region=random.choice(list(DIALECT_RULES))
    s=sent
    for pattern,repl in DIALECT_RULES[region]:
        if random.random()<rate:
            s=re.sub(pattern, repl, s, flags=re.IGNORECASE)
    if random.random()<0.35:
        tail=random.choice(DIALECT_TAILS[region])
        if tail:
            s=re.sub(r"([.!?]*)$", "", s).rstrip()+" "+tail
    return s

_LEXICAL_ALL = {**COMMON_MISSPELLINGS, **LEXICAL_INFORMAL}
_LEXICAL_SORTED = sorted(_LEXICAL_ALL.items(), key=lambda kv: -len(kv[0]))
def _lexical_case(matched, new):
    return (new[:1].upper() + new[1:]) if matched[:1].isupper() else new
def lexical_corrupt(sent, rate=0.8):
    s=sent
    for correct, variants in _LEXICAL_SORTED:
        pat=r"\b"+re.escape(correct)+r"\b"
        if random.random()<rate and re.search(pat, s, re.IGNORECASE):
            bad=random.choice(variants)
            s=re.sub(pat, lambda m:_lexical_case(m.group(0), bad), s, count=1, flags=re.IGNORECASE)
    return s

_DEASCII = str.maketrans("çğıöşüÇĞİÖŞÜ", "cgiosuCGIOSU")
def deascii_corrupt(sent):
    new=sent.translate(_DEASCII)
    return new, (new!=sent)

_QUESTION_PARTICLES = {
    "mi","mı","mu","mü",
    "mısın","misin","musun","müsün","mıyım","miyim","muyum","müyüm",
    "mıyız","miyiz","muyuz","müyüz","mısınız","misiniz","musunuz","müsünüz",
    "mıdır","midir","mudur","müdür","mıydı","miydi","muydu","müydü",
    "mıydım","miydim","mıymış","miymiş",
}
_CONJUNCTION_PARTICLES = {"de","da","ki"}

def _merge_particle(sent, particle_set, rate):
    toks=sent.split(); out=[]; applied=False
    for w in toks:
        core=w.lower().strip(".,!?;:\"'")
        if out and core in particle_set and random.random()<rate:
            out[-1]=out[-1]+w; applied=True; continue
        out.append(w)
    return " ".join(out), applied

def conjunction_corrupt(sent, rate=0.85):
    return _merge_particle(sent, _CONJUNCTION_PARTICLES, rate)

def question_particle_corrupt(sent, rate=0.85):
    return _merge_particle(sent, _QUESTION_PARTICLES, rate)

def particle_corrupt(sent, rate=0.7):
    s,_=_merge_particle(sent, _CONJUNCTION_PARTICLES | _QUESTION_PARTICLES, rate)
    return s

def apostrophe_corrupt(sent):
    new=sent.replace("'", "").replace("’", "")
    return new, (new!=sent)

def spacing_corrupt(sent, rate=0.9):
    words=sent.split()
    if len(words)<2: return sent, False
    if random.random()<0.7:
        i=random.randrange(len(words)-1)
        words[i]=words[i]+words[i+1]; del words[i+1]
        return " ".join(words), True
    long_idx=[i for i,w in enumerate(words) if len(re.sub(r"\W","",w))>=6]
    if not long_idx:
        i=random.randrange(len(words)-1)
        words[i]=words[i]+words[i+1]; del words[i+1]
        return " ".join(words), True
    i=random.choice(long_idx); w=words[i]; k=random.randint(2, len(w)-2)
    words[i]=w[:k]+" "+w[k:]
    return " ".join(words), True

_ALL_DIALECT_PATTERNS = [d for rules in DIALECT_RULES.values() for d,_ in rules]
def _fits_dialect(s):    return any(re.search(d, s, re.IGNORECASE) for d in _ALL_DIALECT_PATTERNS)
def _fits_slang(s):
    low=" "+s.lower()+" "
    return any((" "+k+" ") in low or (" "+k) in low for k in JARGON)
def _fits_lexical(s):
    low=" "+s.lower()+" "
    return any((" "+k+" ") in low for k in _LEXICAL_ALL)
def _fits_conjunction(s):
    toks=[w.lower().strip(".,!?;:\"'") for w in s.split()]
    return any(w in _CONJUNCTION_PARTICLES for w in toks)
def _fits_question_particle(s):
    toks=[w.lower().strip(".,!?;:\"'") for w in s.split()]
    return any(w in _QUESTION_PARTICLES for w in toks)
def _fits_apostrophe(s):  return ("'" in s) or ("’" in s)
def _fits_deascii(s):     return any(c in "çğıöşüÇĞİÖŞÜ" for c in s)
def _fits_spacing(s):     return len(s.split())>=2
def _fits_duplicate(s):
    return any(w[i].lower()==w[i+1].lower() for w in s.split() for i in range(len(w)-1))
def _fits_char_general(s):
    return any(len(re.sub(r"\W","",w))>=3 for w in s.split())

def fits(sent, kind):
    if kind=="dialect":          return _fits_dialect(sent)
    if kind=="slang":            return _fits_slang(sent)
    if kind=="lexical":          return _fits_lexical(sent)
    if kind=="conjunction":      return _fits_conjunction(sent)
    if kind=="question_particle":return _fits_question_particle(sent)
    if kind=="apostrophe":       return _fits_apostrophe(sent)
    if kind=="deascii":          return _fits_deascii(sent)
    if kind=="spacing":          return _fits_spacing(sent)
    if kind=="dup_remove":       return _fits_duplicate(sent)
    if kind in CHAR_FUNCS: return _fits_char_general(sent)
    return True

def corrupt_type(sent, kind, region=None):
    if kind in CHAR_FUNCS:
        s,ap=char_corrupt(sent, kind)
        return (s, kind, "") if ap else (sent, None, "")
    if kind=="conjunction":
        s,ap=conjunction_corrupt(sent);      return (s,"conjunction","") if ap else (sent,None,"")
    if kind=="question_particle":
        s,ap=question_particle_corrupt(sent);return (s,"question_particle","") if ap else (sent,None,"")
    if kind=="apostrophe":
        s,ap=apostrophe_corrupt(sent); return (s,"apostrophe","") if ap else (sent,None,"")
    if kind=="spacing":
        s,ap=spacing_corrupt(sent);    return (s,"spacing","")  if ap else (sent,None,"")
    if kind=="deascii":
        s,ap=deascii_corrupt(sent);    return (s,"deascii","") if ap else (sent,None,"")
    if kind=="slang":
        s=slang_corrupt(sent,0.7);     return (s,"slang","")  if s!=sent else (sent,None,"")
    if kind=="lexical":
        s=lexical_corrupt(sent,0.85);  return (s,"lexical","") if s!=sent else (sent,None,"")
    if kind=="dialect":
        reg=region or random.choice(list(DIALECT_RULES))
        fits_dialect=_fits_dialect(sent)
        s=dialect_corrupt(sent,reg,0.85)
        return (s,"dialect",reg) if (fits_dialect and s!=sent) else (sent,None,"")
    raise ValueError(f"unknown type: {kind}")

def corrupt(sent, mode="mixed", region=None, seed=None):
    if seed is not None: random.seed(seed)
    if mode=="keyboard": return keyboard_sentence(sent, rate=0.6)
    if mode=="slang":
        s=slang_corrupt(sent, rate=0.7); return keyboard_sentence(s, rate=0.15)
    if mode=="dialect":
        s=dialect_corrupt(sent, region=region, rate=0.85)
        if random.random()<0.5: s=slang_corrupt(s, rate=0.3)
        return keyboard_sentence(s, rate=0.2)
    if mode=="particle":
        s=particle_corrupt(sent, rate=0.85)
        if random.random()<0.3: s=slang_corrupt(s, rate=0.25)
        return keyboard_sentence(s, rate=0.15)
    if mode=="lexical":
        s=lexical_corrupt(sent, rate=0.85)
        if random.random()<0.3: s=dialect_corrupt(s, region=region, rate=0.3)
        return keyboard_sentence(s, rate=0.15)
    s=sent; r=random.random()
    if r<0.45:
        s=dialect_corrupt(s, region=region, rate=0.7)
        if random.random()<0.4: s=slang_corrupt(s, rate=0.3)
    elif r<0.75:
        s=slang_corrupt(s, rate=0.55)
        if random.random()<0.3: s=dialect_corrupt(s, region=region, rate=0.4)
    return keyboard_sentence(s, rate=0.25)


if __name__ == "__main__":
    random.seed(7)
    print("="*66); print("  CHARACTER FAMILY — word level (first/last letter rules)"); print("="*66)
    words=["merhaba","seviyorum","istanbul","kelime","erzurum","selam","burak","salla"]
    for slug,(fn,_,_) in CHAR_FUNCS.items():
        examples=[]
        for w in words:
            new,ap=fn(w)
            if ap and len(examples)<3: examples.append(f"{w}->{new}")
        print(f"  {slug:20s}: {', '.join(examples) if examples else '(no suitable word)'}")

    print("\n"+"="*66); print("  ALL TYPES — on full sentences (source | applied?)"); print("="*66)
    tests = {
        "random_keyboard":  "Merhaba nasılsın bugün",
        "add_char_front":   "Merhaba dünya güzel",
        "remove_char_front":"Bugün hava çok güzel",
        "random_char_change":"Kelime anlamını bilmiyorum",
        "swap_two_char":    "Erzurum çok soğuk bir şehir",
        "add_char_end":     "Seni seviyorum çok fazla",
        "repeat_last_char": "Merhaba arkadaşım nasılsın",
        "remove_second_last":"Selam sana da olsun",
        "add_vowel":        "Burak bugün okula gitti",
        "remove_vowels":    "İstanbul çok kalabalık bir yer",
        "change_first_char":"İstanbul Türkiye'nin incisi",
        "remove_all_vowels":"İstanbul büyük şehir",
        "dup_remove":       "Salla ellerini yukarı kaldır",
        "dialect":          "Ne yapıyorsun, yarın geleceğim sana",
        "slang":            "Selam nasılsın, teşekkür ederim",
        "lexical":          "Bu bir sürpriz oldu, yalnız kaldım",
        "deascii":          "Işığı aç, çünkü çok karanlık",
        "apostrophe":       "Ankara'da hava Bursa'dan soğuk",
        "conjunction":      "Sen de gel bizimle, ben de geliyorum",
        "question_particle":"Yarın gelecek misin, hazır mısın",
        "spacing":          "Okula gittim ve arkadaşımı gördüm",
    }
    for kind,c in tests.items():
        s,nt,reg=corrupt_type(c,kind)
        status="OK" if nt else "SKIPPED (not applicable)"
        rstr=f" [{reg}]" if reg else ""
        print(f"  [{kind:18s}] {status}{rstr}")
        print(f"       clean: {c}")
        print(f"       noisy: {s}")
