"""THE LONG DAWN v3 - SOUND lane: a frugal Freesound API v2 client (search, sound info, HQ previews, originals).

Credentials never leave the headers: nothing here prints, logs or writes a key or a token.
  ~/.config/longdawn-freesound/credentials.json   {client_id, api_key}   token auth: search, info, previews
  ~/.config/longdawn-freesound/oauth.json         {access_token, refresh_token, obtained_at, expires_in}
                                                  OAuth2 bearer: /sounds/<id>/download/ (the originals)
Rate limits (60 / minute, 2,000 / day) are kept by a request ledger in music/cache/sound/ledger.json; every
JSON response is cached on disk by its request, so a re-run costs nothing.

    python freesound_v3.py search "flint steel" --cc0 --n 15
    python freesound_v3.py info 12345
    python freesound_v3.py ledger
"""
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MUSIC = os.path.dirname(HERE)
CACHE = os.path.join(MUSIC, "cache", "sound")
FS_DIR = os.path.join(CACHE, "fs")
JS_DIR = os.path.join(CACHE, "json")
LEDGER = os.path.join(CACHE, "ledger.json")
CRED = os.path.expanduser("~/.config/longdawn-freesound/credentials.json")
OAUTH = os.path.expanduser("~/.config/longdawn-freesound/oauth.json")
API = "https://freesound.org/apiv2"
PER_MIN, PER_DAY = 55, 1900
LICENCE_OK = ("creativecommons.org/publicdomain/zero", "creativecommons.org/licenses/by/")
FIELDS = ("id,name,tags,description,license,username,duration,samplerate,bitdepth,channels,type,filesize,"
          "avg_rating,num_ratings,num_downloads,previews,pack")
DESC = "loudness,spectral_centroid,pitch_salience,spectral_flatness,dynamic_range,boominess,roughness"
for d in (FS_DIR, JS_DIR):
    os.makedirs(d, exist_ok=True)


def licence_ok(lic):
    """CC0 or CC-BY only (never NonCommercial, never Sampling+)."""
    lic = (lic or "").lower()
    return any(k in lic for k in LICENCE_OK) and "-nc" not in lic and "sampling" not in lic and "nc/" not in lic


def licence_short(lic):
    lic = (lic or "").lower()
    if "publicdomain/zero" in lic:
        return "CC0"
    if "licenses/by/" in lic:
        return "CC-BY " + lic.rstrip("/").split("/")[-1]
    return "NOT-OK " + lic


# ---------------------------------------------------------------- the ledger (rate limits)
def _ledger():
    try:
        return json.load(open(LEDGER))
    except Exception:
        return {"stamps": [], "days": {}}


def _tick():
    lg = _ledger()
    now = time.time()
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    if lg["days"].get(day, 0) >= PER_DAY:
        raise RuntimeError(f"Freesound daily budget reached ({PER_DAY})")
    recent = [t for t in lg["stamps"] if now - t < 60]
    if len(recent) >= PER_MIN:
        time.sleep(60.5 - (now - recent[0]))
        now = time.time()
        recent = [t for t in recent if now - t < 60]
    recent.append(now)
    lg["stamps"] = recent
    lg["days"][day] = lg["days"].get(day, 0) + 1
    json.dump(lg, open(LEDGER, "w"))


def used_today():
    return _ledger()["days"].get(time.strftime("%Y-%m-%d", time.gmtime()), 0)


# ---------------------------------------------------------------- auth (headers only)
def _token_header():
    return {"Authorization": "Token " + json.load(open(CRED))["api_key"]}


def _oauth():
    return json.load(open(OAUTH))


def refresh_oauth(force=False):
    """refresh the bearer token when it is within 2 h of expiry (or forced); write it back, chmod 600"""
    o = _oauth()
    left = o.get("obtained_at", 0) + o.get("expires_in", 0) - time.time()
    if left > 7200 and not force:
        return left
    c = json.load(open(CRED))
    body = urllib.parse.urlencode({"grant_type": "refresh_token", "refresh_token": o["refresh_token"],
                                   "client_id": c["client_id"], "client_secret": c["api_key"]}).encode()
    _tick()
    req = urllib.request.Request(API + "/oauth2/access_token/", data=body, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            new = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"oauth refresh failed: HTTP {e.code}") from None
    new["obtained_at"] = int(time.time())
    tmp = OAUTH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(new, f)
    os.chmod(tmp, 0o600)
    os.replace(tmp, OAUTH)
    return new.get("expires_in", 0)


def _bearer_header():
    refresh_oauth()
    return {"Authorization": "Bearer " + _oauth()["access_token"]}


# ---------------------------------------------------------------- requests
def _get(url, headers, binary=False, timeout=120):
    _tick()
    req = urllib.request.Request(url, headers=dict(headers, **{"User-Agent": "longdawn-sound/1.0"}))
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            return data if binary else json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 2:
                time.sleep(30)
                continue
            try:
                body = e.read()[:300].decode("utf-8", "replace")
            except Exception:
                body = ""
            raise RuntimeError(f"HTTP {e.code} on {url.split('?')[0]}: {body}") from None
        except urllib.error.URLError as e:
            if attempt < 2:
                time.sleep(5)
                continue
            raise RuntimeError(f"network error on {url.split('?')[0]}: {e.reason}") from None


def _cached_json(url, headers):
    key = hashlib.sha1(url.encode()).hexdigest()[:20]
    p = os.path.join(JS_DIR, key + ".json")
    if os.path.exists(p):
        return json.load(open(p))
    d = _get(url, headers)
    json.dump(d, open(p, "w"))
    return d


def search(query, cc0_only=False, n=15, extra_filter="", sort="score", fields=FIELDS, page=1):
    flt = []
    if cc0_only:
        flt.append('license:"Creative Commons 0"')
    else:
        flt.append('(license:"Creative Commons 0" OR license:"Attribution")')
    if extra_filter:
        flt.append(extra_filter)
    q = urllib.parse.urlencode({"query": query, "filter": " ".join(flt), "fields": fields, "page_size": n,
                                "sort": sort, "page": page})
    d = _cached_json(f"{API}/search/text/?{q}", _token_header())
    return [s for s in d.get("results", []) if licence_ok(s.get("license"))], d.get("count", 0)


def info(sid, fields=FIELDS):
    q = urllib.parse.urlencode({"fields": fields})
    return _cached_json(f"{API}/sounds/{int(sid)}/?{q}", _token_header())


def analysis_fields(sid, names):
    """Freesound's own descriptors as fields (e.g. 'loudness,spectral_centroid,...'); {} if unavailable"""
    q = urllib.parse.urlencode({"fields": "id," + ",".join(names)})
    try:
        return _cached_json(f"{API}/sounds/{int(sid)}/?{q}", _token_header())
    except RuntimeError:
        return {}


def preview(s, kind="preview-hq-ogg"):
    """download a sound's HQ preview (token auth); cached as fs/<id>_hq.ogg"""
    p = os.path.join(FS_DIR, f"{int(s['id'])}_hq.ogg")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    url = s["previews"][kind]
    data = _get(url, _token_header(), binary=True)
    with open(p + ".part", "wb") as f:
        f.write(data)
    os.replace(p + ".part", p)
    return p


def original(s, max_mb=60):
    """download a sound's ORIGINAL file (OAuth2 bearer); cached as fs/<id>_orig.<type>"""
    ext = (s.get("type") or "wav").lower()
    p = os.path.join(FS_DIR, f"{int(s['id'])}_orig.{ext}")
    if os.path.exists(p) and os.path.getsize(p) > 1000:
        return p
    mb = (s.get("filesize") or 0) / 1e6
    if mb > max_mb:
        raise RuntimeError(f"sound {s['id']} original is {mb:.0f} MB > {max_mb} MB: use the preview")
    data = _get(f"{API}/sounds/{int(s['id'])}/download/", _bearer_header(), binary=True, timeout=600)
    with open(p + ".part", "wb") as f:
        f.write(data)
    os.replace(p + ".part", p)
    return p


def disk_mb():
    tot = 0
    for f in os.listdir(FS_DIR):
        tot += os.path.getsize(os.path.join(FS_DIR, f))
    return tot / 1e6


def brief(s):
    return (f"{s['id']:>7} {licence_short(s.get('license')):10s} {s.get('duration', 0):6.1f}s "
            f"{s.get('samplerate', 0) / 1000:.0f}k/{s.get('bitdepth', 0)}b/{s.get('channels', 0)}ch {s.get('type', ''):4s} "
            f"{(s.get('filesize') or 0) / 1e6:6.1f}MB  r{s.get('avg_rating', 0):.1f}({s.get('num_ratings', 0)}) "
            f"dl{s.get('num_downloads', 0):>6}  {s.get('username', '')[:14]:14s} {s.get('name', '')[:60]}"
            + (f"\n          L{s['loudness']:.0f} cen{(s.get('spectral_centroid') or 0):.0f} ps{(s.get('pitch_salience') or 0):.2f} "
               f"fl{(s.get('spectral_flatness') or 0):.2f} dr{(s.get('dynamic_range') or 0):.0f} boom{(s.get('boominess') or 0):.0f}"
               if s.get('loudness') is not None else ""))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--cc0", action="store_true")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--filter", default="")
    ap.add_argument("--sort", default="score")
    ap.add_argument("--tags", action="store_true")
    a = ap.parse_args()
    if a.cmd == "search":
        res, count = search(a.arg, a.cc0, a.n, a.filter, a.sort)
        print(f"# '{a.arg}' {count} hits ({len(res)} shown, licence-checked)")
        for s in res:
            print(brief(s))
            if a.tags:
                print("          tags:", " ".join(s.get("tags", [])[:18]))
    elif a.cmd == "info":
        s = info(a.arg)
        print(brief(s))
        print("tags:", " ".join(s.get("tags", [])))
        print("desc:", (s.get("description") or "")[:600].replace("\n", " "))
    elif a.cmd in ("get", "orig"):
        for sid in a.arg.split(","):
            s = info(sid)
            try:
                path = original(s) if a.cmd == "orig" else preview(s)
            except RuntimeError as e:
                print(f"{sid}: {e}")
                continue
            print(f"{sid} -> {os.path.relpath(path, MUSIC)} ({os.path.getsize(path) / 1e6:.1f} MB)")
        print(f"requests today: {used_today()} / {PER_DAY};  cached audio {disk_mb():.0f} MB")
    elif a.cmd == "refresh":
        print(f"token valid for {refresh_oauth(force=a.arg == 'force') / 3600:.1f} h")
    sys.stdout.flush()
