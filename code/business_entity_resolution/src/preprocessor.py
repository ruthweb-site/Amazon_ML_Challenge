import re
import unicodedata

# Common legal and address abbreviations
LEGAL_NORM = {
    r"\bcorp\b": "corporation",
    r"\binc\b": "incorporated",
    r"\bltd\b": "limited",
    r"\bpvt\b": "private",
    r"\bco\b": "company",
    r"\bllc\b": "llc",
    r"\bllp\b": "llp",
    r"\bsarl\b": "sarl",
}

ADDRESS_NORM = {
    r"\brd\b": "road",
    r"\bst\b": "street",
    r"\bave\b": "avenue",
    r"\bdr\b": "drive",
    r"\bln\b": "lane",
    r"\bblvd\b": "boulevard",
    r"\bapt\b": "apartment",
    r"\bste\b": "suite",
    r"\bfl\b": "floor",
    r"\bpk\b": "park",
    r"\bsec\b": "sector",
    r"\bopp\b": "opposite",
    r"\bnear\b": "near",
}

def clean_text(text: str) -> str:
    """Normalize text: unicode normalization, lowercasing, punctuation cleanup."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Unicode NFKD normalization
    text = unicodedata.normalize("NFKD", text)
    text = text.lower().strip()
    
    # Replace & with and
    text = re.sub(r"\s*&\s*", " and ", text)
    
    # Remove special punctuation but keep alphanumeric and spaces
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def normalize_business_name(name: str) -> str:
    """Clean and normalize business name terms."""
    cleaned = clean_text(name)
    if not cleaned:
        return ""
    
    for pattern, replacement in LEGAL_NORM.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    
    return re.sub(r"\s+", " ", cleaned).strip()

def normalize_address(address: str) -> str:
    """Clean and normalize business address."""
    cleaned = clean_text(address)
    if not cleaned:
        return ""
    
    for pattern, replacement in ADDRESS_NORM.items():
        cleaned = re.sub(pattern, replacement, cleaned)
    
    return re.sub(r"\s+", " ", cleaned).strip()
