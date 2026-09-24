"""Download the (SIL OFL-licensed) Google Fonts used by paint.py into ./fonts."""
import os
import re
import urllib.parse
import urllib.request

FAMILIES = {
    "NotoSerif-Regular": "Noto Serif:wght@400",
    "NotoSerif-Italic": "Noto Serif:ital,wght@1,400",
    "NotoNaskhArabic": "Noto Naskh Arabic:wght@400",
    "NotoSerifHebrew": "Noto Serif Hebrew:wght@400",
    "NotoSerifDevanagari": "Noto Serif Devanagari:wght@400",
    "NotoSerifBengali": "Noto Serif Bengali:wght@400",
    "NotoSerifTamil": "Noto Serif Tamil:wght@400",
    "NotoSerifThai": "Noto Serif Thai:wght@400",
    "NotoSerifJP": "Noto Serif JP:wght@400",
    "NotoSerifSC": "Noto Serif SC:wght@400",
    "NotoSerifKR": "Noto Serif KR:wght@400",
    "NotoSerifEthiopic": "Noto Serif Ethiopic:wght@400",
    "NotoSerifArmenian": "Noto Serif Armenian:wght@400",
    "NotoSerifGeorgian": "Noto Serif Georgian:wght@400",
    "NotoSansMath": "Noto Sans Math",
    "MrDafoe": "Mr Dafoe",
    "Allura": "Allura",
    "HomemadeApple": "Homemade Apple",
}

here = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(here, "fonts")
os.makedirs(out_dir, exist_ok=True)

for name, spec in FAMILIES.items():
    dest = os.path.join(out_dir, name + ".ttf")
    if os.path.exists(dest):
        continue
    css_url = "https://fonts.googleapis.com/css2?family=" + urllib.parse.quote(spec, safe=":,@;")
    css = urllib.request.urlopen(css_url).read().decode()
    urls = re.findall(r"url\((https://[^)]+\.ttf)\)", css)
    if not urls:
        raise SystemExit(f"no ttf for {spec}")
    with urllib.request.urlopen(urls[0]) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"{name}: {os.path.getsize(dest) // 1024} KB")
