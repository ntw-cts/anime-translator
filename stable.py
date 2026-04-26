import sys
import cv2
import mss
import numpy as np
import easyocr
import time
import requests
from rapidfuzz import fuzz, process
from shapely import box
from symspellpy import SymSpell, Verbosity
import kenlm
import google.generativeai as genai
import pkg_resources
import ctypes
import uuid
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QLineEdit, QPushButton, 
                             QCheckBox, QPlainTextEdit, QGroupBox, QComboBox, QTabWidget)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtGui import QPainter, QPen, QColor, QFont, QFontDatabase, QTextCursor, QSurfaceFormat
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject, pyqtSlot, QThread
from OpenGL import GL
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", message="RNN module weights are not part of single contiguous chunk")
import os
import json

# --- NLLB-200 Offline Translation Engine (lazy-loaded on first use) ---
_nllb_model = None
_nllb_tokenizer = None
_nllb_device = None
_nllb_load_attempted = False
_nllb_ready = False

def get_nllb_model():
    """Lazy-load the NLLB-200-distilled-600M model (only once, on background thread)."""
    global _nllb_model, _nllb_tokenizer, _nllb_device, _nllb_load_attempted, _nllb_ready
    if _nllb_load_attempted:
        return _nllb_model, _nllb_tokenizer, _nllb_device
    _nllb_load_attempted = True
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        _nllb_device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "facebook/nllb-200-distilled-600M"
        print(f"[NLLB] Loading {model_name} on {_nllb_device.upper()}...")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(_nllb_device)
        _nllb_ready = True
        print(f"[NLLB] Model ready on {_nllb_device.upper()} — offline translation active.")
    except Exception as e:
        print(f"[NLLB] Failed to load model: {e}. Offline translation unavailable.")
        _nllb_model = None
        _nllb_tokenizer = None
        _nllb_device = "cpu"
    return _nllb_model, _nllb_tokenizer, _nllb_device

def _preload_nllb_background():
    """Kick off NLLB loading on a daemon thread so startup is never blocked."""
    import threading
    t = threading.Thread(target=get_nllb_model, daemon=True)
    t.start()

def nllb_translate(text: str) -> str:
    """Translate English text to Thai using the NLLB-200 model."""
    model, tokenizer, device = get_nllb_model()
    if model is None or tokenizer is None:
        return ""
    try:
        inputs = tokenizer(text, return_tensors="pt").to(device)
        translated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("tha_Thai"),
            max_length=400
        )
        return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
    except Exception as e:
        print(f"[NLLB] Translation error: {e}")
        return ""

# --- STEP 14: LaBSE Semantic Sanity Check (lazy-loaded on first use) ---
_labse_model = None
_labse_load_attempted = False

def get_labse_model():
    """Lazy-load the LaBSE sentence embedding model (only once)."""
    global _labse_model, _labse_load_attempted
    if _labse_load_attempted:
        return _labse_model
    _labse_load_attempted = True
    
    # REMOVED the sys.stdout swapping hack here that was breaking the UI logs!
    try:
        from sentence_transformers import SentenceTransformer
        _labse_model = SentenceTransformer('LaBSE')
        print("[INFO] LaBSE model loaded for semantic sanity checks.")
    except Exception as e:
        print(f"[WARN] LaBSE unavailable ({e}). Sanity check will be skipped.")
        _labse_model = None
    return _labse_model

def _preload_labse_background():
    """Kick off LaBSE loading on a daemon thread so startup is never blocked."""
    import threading
    t = threading.Thread(target=get_labse_model, daemon=True)
    t.start()

def labse_similarity(en_text: str, th_text: str) -> float:
    """Return cosine similarity between English source and Thai translation."""
    import numpy as np
    model = get_labse_model()
    if model is None or not en_text or not th_text:
        return 1.0  # Pass-through if model not available
    try:
        vecs = model.encode([en_text, th_text], normalize_embeddings=True)
        return float(np.dot(vecs[0], vecs[1]))
    except Exception:
        return 1.0

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    ctypes.windll.user32.SetProcessDPIAware()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

# --- SYSTEM REDIRECTION ---
class ConsoleEmitter(QObject):
    text_written = pyqtSignal(str)

class OutStream:
    def __init__(self, emitter):
        self.emitter = emitter
    def write(self, text):
        if text.strip():
            self.emitter.text_written.emit(str(text))
    def flush(self): pass
    def isatty(self): return False

class TranslationWorker(QObject):
    translation_ready = pyqtSignal(str, str)  # (sub_id, thai)
    gemini_status = pyqtSignal(bool)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.cache_hits = 0
        self.cache_misses = 0

    def google_translate(self, text):
        try:
            url = f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=th&dt=t&q={text}"
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return r.json()[0][0][0]
        except: pass
        return ""
    
    def ollama_translate_with_prompt(self, prompt):
        try:
            payload = {"model": "scb10x/typhoon-translate1.5-4b", "prompt": prompt, "stream": False}
            r = requests.post("http://localhost:11434/api/generate", json=payload, timeout=10)
            if r.status_code == 200: return r.json().get("response", "").strip()
        except: pass
        return ""

    @pyqtSlot(str, str, dict, list, str)
    def run_refinement(self, sub_id, pure_english, settings, entities, refine_mode):
        """Auto-mode quality refine step: Typhoon or Gemini, runs ~2s after NLLB fast pass."""
        try:
            anime_title = settings.get('anime_title', 'Unknown Anime')
            api_key = settings.get('gemini_api_key', '').strip()

            def build_context_prompt(en_text, anime_title, character_names):
                char_list = ", ".join(character_names[:20]) if character_names else "N/A"
                return (f"You are translating anime subtitles for '{anime_title}'.\nMain characters: {char_list}.\nRules:\n- Respond with ONLY the Thai translation. No notes, no explanation.\n- Preserve each character's speech style and personality.\n- Use natural, conversational Thai suitable for subtitles.\n- If untranslatable, respond exactly: [UNTRANSLATABLE]\n\nTranslate to Thai: {en_text}")

            def clean_llm_output(text):
                import re
                if not text: return text
                if re.match(r'^[^:\n]{0,40}:\s+', text): text = text.split(":", 1)[-1]
                text = text.strip().strip('"\'')
                bad_patterns = re.compile(r'(note[:\s]|sure[,!]|here\'?s|let me|because|context|casual|formal|i cannot|i\'m sorry|as an ai|ขออภัย|เนื่องจาก|การแปล|หมายเหตุ)', re.IGNORECASE)
                clean = [l.strip() for l in text.strip().splitlines() if l.strip() and not bad_patterns.search(l)]
                return " ".join(clean).strip() if clean else text.strip()

            refined = ""
            if refine_mode == 'typhoon':
                prompt = build_context_prompt(pure_english, anime_title, entities)
                raw = self.ollama_translate_with_prompt(prompt)
                refined = clean_llm_output(raw) if raw else ""
                if refined:
                    print(f"[TRANSLATE] Auto/Typhoon (refined): {refined}")
            elif refine_mode == 'gemini' and api_key:
                try:
                    genai.configure(api_key=api_key)
                    model_obj = genai.GenerativeModel('gemini-2.5-flash')
                    prompt = build_context_prompt(pure_english, anime_title, entities)
                    response = model_obj.generate_content(prompt)
                    refined = clean_llm_output(response.text.strip())
                    self.gemini_status.emit(True)
                    if refined:
                        print(f"[TRANSLATE] Auto/Gemini (refined): {refined}")
                except Exception:
                    self.gemini_status.emit(False)

            # Only emit if we got a valid Thai result
            def is_thai(text):
                if not text: return False
                return (sum(1 for c in text if '\u0E00' <= c <= '\u0E7F') / max(len(text), 1)) > 0.3

            if refined and is_thai(refined):
                # Save the refined result to cache under its specific engine name
                _refine_engine = "Auto-Typhoon" if refine_mode == "typhoon" else "Auto-Gemini"
                _refine_tier = 3
                try:
                    CACHE_FILE = "translations_cache.json"
                    anime_title_r = settings.get('anime_title', 'Unknown Anime')
                    cache_key_r = f"{anime_title_r}::{pure_english}"
                    cache_r = {}
                    if os.path.exists(CACHE_FILE):
                        with open(CACHE_FILE, "r", encoding="utf-8") as f: cache_r = json.load(f)
                    raw_r = cache_r.get(cache_key_r)
                    if raw_r is None:
                        cache_r[cache_key_r] = {_refine_engine: {"translation": refined, "tier": _refine_tier}}
                    else:
                        if "translation" in raw_r:
                            old_eng = raw_r.get("engine", "Google Translate")
                            cache_r[cache_key_r] = {old_eng: {"translation": raw_r["translation"], "tier": 1}}
                            raw_r = cache_r[cache_key_r]
                        raw_r[_refine_engine] = {"translation": refined, "tier": _refine_tier}
                    with open(CACHE_FILE, "w", encoding="utf-8") as f: json.dump(cache_r, f, ensure_ascii=False, indent=4)
                except Exception as ce:
                    print(f"[CACHE] Refinement cache write failed: {ce}")
                self.translation_ready.emit(sub_id + "__AUTO_REFINE__", refined)
        except Exception as e:
            print(f"[ERROR] Refinement failed: {e}")

    @pyqtSlot(str, str, dict, list)
    def run_translation(self, sub_id, pure_english, settings, entities):
        try:
            # --- TRANSLATION ENGINE TIER ---
            # Rank 3 = Typhoon 1.5 / Gemini (top quality, same rank — LaBSE breaks ties)
            # Rank 2 = NLLB-200 (offline, good quality)
            # Rank 1 = Google Translate (emergency baseline)
            ENGINE_TIER = {
                "Typhoon 1.5":      3,
                "Gemini": 3,
                "Auto-Typhoon":     3,
                "Auto-Gemini":      3,
                "NLLB-200":         2,
                "Auto-NLLB":        2,
                "Google Translate":  1,
                "Auto":             3,
            }

            # --- HELPERS ---
            def is_thai(text):
                if not text: return False
                return (sum(1 for c in text if '\u0E00' <= c <= '\u0E7F') / max(len(text), 1)) > 0.3

            def clean_llm_output(text):
                import re
                if not text: return text
                if re.match(r'^[^:\n]{0,40}:\s+', text): text = text.split(":", 1)[-1]
                text = text.strip().strip('"\'')
                bad_patterns = re.compile(r'(note[:\s]|sure[,!]|here\'?s|let me|because|context|casual|formal|i cannot|i\'m sorry|as an ai|ขออภัย|เนื่องจาก|การแปล|หมายเหตุ)', re.IGNORECASE)
                clean = [l.strip() for l in text.strip().splitlines() if l.strip() and not bad_patterns.search(l)]
                return " ".join(clean).strip() if clean else text.strip()

            def length_ratio_ok(en_text, th_text):
                if not en_text or not th_text: return False
                return 0.15 < (len(th_text) / max(len(en_text), 1)) < 5.0

            def build_context_prompt(en_text, anime_title, character_names):
                char_list = ", ".join(character_names[:20]) if character_names else "N/A"
                return (f"You are translating anime subtitles for '{anime_title}'.\nMain characters: {char_list}.\nRules:\n- Respond with ONLY the Thai translation. No notes, no explanation.\n- Preserve each character's speech style and personality.\n- Use natural, conversational Thai suitable for subtitles.\n- If untranslatable, respond exactly: [UNTRANSLATABLE]\n\nTranslate to Thai: {en_text}")

            thai_output  = ""
            CACHE_FILE   = "translations_cache.json"
            anime_title      = settings.get('anime_title', 'Unknown Anime')
            character_names  = entities
            mode             = settings.get('translation_mode', 'Auto')
            api_key          = settings.get('gemini_api_key', '').strip()
            current_tier     = ENGINE_TIER.get(mode, 1)
            cache_key = f"{anime_title}::{pure_english}"
            _cache_hit = False

            if len(pure_english) > 2:
                cache = {}
                if os.path.exists(CACHE_FILE):
                    try:
                        with open(CACHE_FILE, "r", encoding="utf-8") as f: cache = json.load(f)
                    except: pass

                # ── CACHE STRUCTURE ──────────────────────────────────────────────────────────
                # Each cache_key maps to a dict of engine→entry, e.g.:
                #   { "Typhoon 1.5": {"translation": "...", "tier": 3},
                #     "NLLB-200":    {"translation": "...", "tier": 2} }
                # Legacy entries (flat dict with a single "translation" key) are handled below.
                # ─────────────────────────────────────────────────────────────────────────────

                def _read_cache_entry(key):
                    """Return the best cached translation for this key, or '' if none."""
                    raw = cache.get(key)
                    if not raw:
                        return "", "", 0

                    # ── Legacy flat format: {"translation": "...", "engine": "...", "tier": N}
                    if "translation" in raw:
                        eng  = raw.get("engine", "Google Translate")
                        tier = ENGINE_TIER.get(eng, raw.get("tier", 1))
                        return raw["translation"], eng, tier

                    # ── New per-engine format: {engine_name: {"translation": "...", "tier": N}, ...}
                    # Collect all entries by tier
                    top_tier = max((ENGINE_TIER.get(e, 1) for e in raw), default=0)
                    top_entries = {e: v for e, v in raw.items()
                                   if ENGINE_TIER.get(e, 1) == top_tier and v.get("translation")}

                    if not top_entries:
                        return "", "", 0

                    if len(top_entries) == 1:
                        eng, v = next(iter(top_entries.items()))
                        return v["translation"], eng, top_tier

                    # ── TIE: two Rank-3 engines (Typhoon + Gemini) both cached → LaBSE decides ──
                    best_eng, best_text, best_score = "", "", -1.0
                    for eng, v in top_entries.items():
                        t = v["translation"]
                        score = labse_similarity(pure_english, t)
                        print(f"[SANITY] Cache tie-break: {eng} score={score:.3f} → {t[:40]}")
                        if score > best_score:
                            best_score, best_text, best_eng = score, t, eng
                    print(f"[SANITY] Cache tie-break winner: {best_eng} ({best_score:.3f})")
                    return best_text, best_eng, top_tier

                def _write_cache_entry(key, engine, translation, tier):
                    """
                    Write a translation to cache with strict quality gating:
                    • Lower tier  → always rejected (never downgrade).
                    • Higher tier → always accepted (clear upgrade).
                    • Same tier, same engine → always overwrite (refresh own slot).
                    • Same tier, different engine → LaBSE decides; only write if new
                      score is strictly better than the existing same-tier entry.
                    """
                    raw = cache.get(key)

                    # ── Brand-new key — just write ────────────────────────────────────────────
                    if raw is None:
                        cache[key] = {engine: {"translation": translation, "tier": tier}}
                        return

                    # ── Migrate legacy flat format ────────────────────────────────────────────
                    if "translation" in raw:
                        old_eng  = raw.get("engine", "Google Translate")
                        old_tier = ENGINE_TIER.get(old_eng, raw.get("tier", 1))
                        cache[key] = {old_eng: {"translation": raw["translation"], "tier": old_tier}}
                        raw = cache[key]

                    # ── Same engine refreshing its own slot — always allow ─────────────────────
                    if engine in raw:
                        raw[engine] = {"translation": translation, "tier": tier}
                        return

                    # ── Check whether any existing slot has the same tier ─────────────────────
                    same_tier_rivals = {e: v for e, v in raw.items()
                                        if ENGINE_TIER.get(e, v.get("tier", 1)) == tier
                                        and v.get("translation")}

                    if not same_tier_rivals:
                        # No rival at this tier — accept if tier is >= everything already stored
                        max_stored_tier = max(
                            (ENGINE_TIER.get(e, v.get("tier", 1)) for e, v in raw.items()),
                            default=0
                        )
                        if tier < max_stored_tier:
                            # Incoming is lower tier than something already stored — reject
                            print(f"[CACHE] Rejected {engine} (tier {tier}) — cache already has tier {max_stored_tier}")
                            return
                        raw[engine] = {"translation": translation, "tier": tier}
                        return

                    # ── Same tier, different engine → LaBSE tiebreaker ────────────────────────
                    new_score = labse_similarity(pure_english, translation)
                    print(f"[CACHE] Tiebreaker — incoming {engine} score={new_score:.3f}: {translation[:40]}")

                    accept = True
                    for rival_eng, rival_v in same_tier_rivals.items():
                        rival_score = labse_similarity(pure_english, rival_v["translation"])
                        print(f"[CACHE] Tiebreaker — existing {rival_eng} score={rival_score:.3f}: {rival_v['translation'][:40]}")
                        if new_score <= rival_score:
                            accept = False
                            print(f"[CACHE] Rejected {engine} — {rival_eng} is equal or better ({rival_score:.3f} >= {new_score:.3f})")
                            break

                    if accept:
                        print(f"[CACHE] Accepted {engine} ({new_score:.3f}) — better than all same-tier rivals")
                        raw[engine] = {"translation": translation, "tier": tier}

                # ── CHECK CACHE ───────────────────────────────────────────────────────────────
                cached_translation, cached_engine, cached_tier = _read_cache_entry(cache_key)

                if cached_translation:
                    # Use the cached result if it's at least as good as what the current mode can produce
                    if cached_tier >= current_tier:
                        thai_output = cached_translation
                        _cache_hit = True
                        print(f"[TRANSLATE] Cache hit ({cached_engine}, tier {cached_tier}): {thai_output}")

                # ── FUZZY FALLBACK (if exact key missed) ─────────────────────────────────────
                if not _cache_hit and not thai_output:
                    if cache:
                        match = process.extractOne(cache_key, list(cache.keys()), scorer=fuzz.ratio)
                        if match and match[1] >= 90:
                            fuzzy_trans, _, fuzzy_tier = _read_cache_entry(match[0])
                            if fuzzy_trans and fuzzy_tier >= current_tier:
                                thai_output = fuzzy_trans
                                _cache_hit = True
                                print(f"[TRANSLATE] Fuzzy cache hit (score {match[1]}): {thai_output}")

                if not _cache_hit:
                    if mode == "Auto":
                        # ── AUTO MODE: NLLB → Typhoon/Gemini (quality refine) → Google (emergency) ──

                        # Step 1: NLLB — instant offline translation (Primary)
                        nllb_result = ""
                        if _nllb_ready:
                            nllb_result = nllb_translate(pure_english)
                        if nllb_result and is_thai(nllb_result):
                            thai_output = nllb_result
                            print(f"[TRANSLATE] Auto: NLLB primary → {thai_output}")
                        else:
                            # NLLB not ready or returned garbage — skip straight to Google for now
                            thai_output = self.google_translate(pure_english)
                            print(f"[TRANSLATE] Auto: NLLB unavailable, Google emergency → {thai_output}")

                        # Step 2: Quality Refine — Typhoon or Gemini (only if text persisted ≥2s)
                        # This is triggered by the overlay via a queued call after the persist delay.
                        # We record the NLLB result as the fast output and let the overlay decide
                        # whether to enqueue a refinement call (see SubtitleOverlay logic).
                        # The `__AUTO_FAST__` marker tells on_translation_finished this is a
                        # fast first-pass result; a refine call may follow.
                        # Count cache miss here — Auto mode returns early and never reaches
                        # the 14f block below, so we must record it before the early return.
                        if thai_output:
                            self.cache_misses += 1
                        self.translation_ready.emit(sub_id + "__AUTO_FAST__", thai_output)
                        # Return early — the signal above delivers the fast result.
                        # The refine path (if triggered) will emit sub_id normally later.
                        return

                    elif mode == "Gemini" and api_key:
                        try:
                            genai.configure(api_key=api_key)
                            model_obj = genai.GenerativeModel('gemini-2.5-flash')
                            prompt = build_context_prompt(pure_english, anime_title, character_names)
                            response = model_obj.generate_content(prompt)
                            thai_output = clean_llm_output(response.text.strip())
                            self.gemini_status.emit(True)
                        except Exception:
                            self.gemini_status.emit(False)
                            thai_output = self.google_translate(pure_english)
                    elif mode == "Gemini" and not api_key:
                        # Gemini selected but no key — fall to Google
                        thai_output = self.google_translate(pure_english)
                    elif mode == "Typhoon 1.5":
                        prompt = build_context_prompt(pure_english, anime_title, character_names)
                        raw_ollama = self.ollama_translate_with_prompt(prompt)
                        thai_output = clean_llm_output(raw_ollama) if raw_ollama else self.google_translate(pure_english)
                    elif mode == "NLLB-200":
                        nllb_result = nllb_translate(pure_english)
                        thai_output = nllb_result if (nllb_result and is_thai(nllb_result)) else self.google_translate(pure_english)
                    else:  # "Google Translate" or fallback
                        thai_output = self.google_translate(pure_english)

                _bad_patterns = ("I cannot", "I'm sorry", "As an AI", "ขออภัย", "[UNTRANSLATABLE]")

                # --- STEP 14: SANITY CHECKS (only meaningful for LLM engines) ---
                # Google translate and NLLB are trusted as baseline — don't re-check their own output.
                _used_llm = mode in ("Gemini", "Typhoon 1.5") and not _cache_hit

                # 14a. Thai script verification — catches LLM output that isn't Thai
                if _used_llm and thai_output and not is_thai(thai_output):
                    self.translation_ready.emit(sub_id + "__SANITY_WARN__", "[SANITY WARN] Output contains no Thai script — falling back to Google")
                    fallback = self.google_translate(pure_english)
                    thai_output = fallback if fallback else thai_output

                # 14b. Garbage / LLM meta-text guard
                if _used_llm and thai_output and any(p in thai_output for p in _bad_patterns):
                    self.translation_ready.emit(sub_id + "__SANITY_WARN__", "[SANITY WARN] LLM returned meta-text — falling back to Google")
                    thai_output = self.google_translate(pure_english)

                # 14c. Length ratio sanity check
                if _used_llm and thai_output and not length_ratio_ok(pure_english, thai_output):
                    self.translation_ready.emit(sub_id + "__SANITY_WARN__", "[SANITY WARN] Suspicious length ratio — falling back to Google")
                    thai_output = self.google_translate(pure_english)

                # 14d. LaBSE semantic similarity check
                LABSE_THRESHOLD = 0.25
                if settings.get('labse_enabled', False) and thai_output and len(pure_english) > 4:
                    try:
                        sim_score = labse_similarity(pure_english, thai_output)
                        if sim_score < LABSE_THRESHOLD:
                            self.translation_ready.emit(sub_id + "__SANITY_WARN__", f"[SANITY WARN] Low semantic similarity ({sim_score:.2f}) — trying fallback")
                            fallback = self.google_translate(pure_english)
                            if fallback:
                                fallback_sim = labse_similarity(pure_english, fallback)
                                if fallback_sim > sim_score:
                                    self.translation_ready.emit(sub_id + "__SANITY_WARN__", f"[SANITY FIX] Replaced ({sim_score:.2f}) with fallback ({fallback_sim:.2f})")
                                    thai_output = fallback
                    except Exception:
                        pass

                # 14e. Save to cache (only clean, verified Thai output)
                _cache_engine = mode
                if mode == "Auto":
                    _cache_engine = "Auto-NLLB"  # fast pass is always NLLB; refine saves separately
                if thai_output and not any(p in thai_output for p in _bad_patterns):
                    _write_cache_entry(cache_key, _cache_engine, thai_output, ENGINE_TIER.get(_cache_engine, 1))
                    try:
                        with open(CACHE_FILE, "w", encoding="utf-8") as f: json.dump(cache, f, ensure_ascii=False, indent=4)
                    except: pass

                # 14f. Cache hit/miss stats
                if _cache_hit:
                    self.cache_hits += 1
                elif thai_output:
                    self.cache_misses += 1

            self.translation_ready.emit(sub_id, thai_output)
            
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            self.translation_ready.emit(sub_id, "") # Failsafe fallback

class DetectionWorker(QObject):
    detection_ready = pyqtSignal(str, str, list, int, str)

    def __init__(self, reader, sym_spell, kenlm_model, settings):
        super().__init__()
        self.reader = reader
        self.sym_spell = sym_spell
        self.lm = kenlm_model
        self.settings = settings

    @pyqtSlot(np.ndarray, list, int, str, dict, list)
    def run_detection(self, img, box, match_idx, initial_text, settings, entities):
        initial_text = initial_text.rstrip('_')
        winner_text = initial_text
        _correction_logs = []

        # --- STEP 8: HIGH-RES CROP OCR ---
        if len(box) == 4:
            pad_h, pad_w_left, pad_w_right = 30, 200, 30
            crop = img[max(0, box[1]-pad_h):min(img.shape[0], box[3]+pad_h),
                    max(0, box[0]-pad_w_left):min(img.shape[1], box[2]+pad_w_right)]

            if crop.size > 0 and settings.get('high_res_crop', True):
                crop_resized = cv2.resize(crop, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
                crop_gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGRA2GRAY)
                crop_gray_raw = crop_gray.copy()

                border_pixels = np.concatenate([
                    crop_gray[0, :], crop_gray[-1, :],
                    crop_gray[:, 0], crop_gray[:, -1]
                ])
                bg_brightness = np.median(border_pixels)
                if bg_brightness > 127:
                    crop_gray = cv2.bitwise_not(crop_gray)

                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
                crop_enhanced = clahe.apply(crop_gray)
                crop_final = cv2.adaptiveThreshold(
                    crop_enhanced, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    blockSize=15, C=8
                )
                crop_final = cv2.copyMakeBorder(crop_final, 20, 20, 40, 20,
                                                cv2.BORDER_CONSTANT, value=0)
                hr_results = self.reader.readtext(crop_final, detail=1)

        # --- LEADING "I" RECOVERY ---
        _i_prefixes = ('i ', "i'", 'i,', 'i.')
        if initial_text.lower().startswith(_i_prefixes) and not winner_text.lower().startswith('i'):
            winner_text = 'I ' + winner_text

        # --- STEP 10: ENTITY SHIELD SNAP ---
        if entities:
            words = winner_text.split()
            protected_words = []
            for word in words:
                if len(word) > 3:
                    match = process.extractOne(word, entities, scorer=fuzz.WRatio)
                    if match and match[1] > 85:
                        if word != match[0]:
                            _shield_key = (word, match[0])
                            if getattr(self, '_last_shield_log', None) != _shield_key:
                                _correction_logs.append(f"[SHIELD MATCH] Snapped typo '{word}' -> '{match[0]}'")
                                self._last_shield_log = _shield_key
                        protected_words.append(match[0])
                        continue
                protected_words.append(word)
            winner_text = " ".join(protected_words)

        # --- STEP 11: SYMSPELL TYPO CORRECTION ---
        words = winner_text.split()
        corrected_words = []
        for word in words:
            clean_word = "".join(filter(str.isalnum, word))
            if entities and clean_word in entities:
                corrected_words.append(word)
                continue
            if len(clean_word) < 4:
                corrected_words.append(word)
                continue
            suggestions = self.sym_spell.lookup(clean_word, Verbosity.TOP, max_edit_distance=1)
            if suggestions:
                suggestion = suggestions[0]
                if len(suggestion.term) >= len(clean_word) * 0.6:
                    if clean_word.lower() != suggestion.term.lower():
                        _sym_key = (clean_word, suggestion.term)
                        if getattr(self, '_last_sym_log', None) != _sym_key:
                            _correction_logs.append(f"[SYMSPELL FIX] Corrected '{clean_word}' -> '{suggestion.term}'")
                            self._last_sym_log = _sym_key
                    corrected_words.append(word.replace(clean_word, suggestion.term))
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        winner_text = " ".join(corrected_words)

        # --- SMART "I" RECOVERY ---
        _likely_incomplete = {
            'see', 'know', 'mean', 'think', 'thought', 'understand', 'agree',
            'disagree', 'refuse', 'wonder', 'doubt', 'believe', 'forgot',
            'remember', 'realized', 'noticed', 'heard', 'saw', 'felt', 'knew',
            'lied', 'tried', 'failed', 'won', 'lost', 'gave', 'got', 'came',
            'went', 'left', 'stayed', 'waited', 'asked', 'told', 'said',
            'promised', 'swear', 'swore', 'give up', "can't", "won't", "don't",
            "didn't", "couldn't", "wouldn't", "shouldn't", "wasn't", "haven't",
            "hadn't", "isn't", 'am', 'do', 'did', 'will', 'must', 'should',
            'could', 'would', 'might', 'need to', 'have to', 'want to', 'want it',
            'used to', 'hate this', 'love you', 'miss you', 'need you', 'trust you',
            'like you', 'hate you', 'found it', 'got it', 'did it', 'made it',
            'blew it', 'mean it', 'said it', 'knew it', 'feel it', 'see it',
            'get it', 'feel sick', 'feel fine', 'feel bad', 'feel good',
            'feel nothing', 'feel the same'
        }
        clean_text = winner_text.lower().strip().rstrip('.,!?')
        if settings.get('smart_i_recovery', True) and clean_text in _likely_incomplete:
            if getattr(self, '_last_i_log', '') != clean_text:
                _correction_logs.append(f"[I-RECOVERY] Appended 'I' to phrase: '{winner_text}'")
                self._last_i_log = clean_text
            winner_text = 'I ' + winner_text

        # --- STEP 12: CONTEXT SCORING (KenLM) ---
        if self.lm is not None:
            candidates = {initial_text, winner_text}
            fix_map = [
                ("0", "o"), ("1", "I"), ("3", "e"), ("5", "S"),
                ("6", "G"), ("8", "B"), ("9", "g"), ("|", "I"), ("`", "'"),
                ("rn", "m"), ("cl", "d"), ("vv", "w"), ("li", "h"),
                ("ti", "h"), ("iii", "m"), ("I I", "ll"),
                ("Fm", "I'm"), ("Fll", "I'll"), ("Fve", "I've"), ("Fd", "I'd"),
                ("l'm", "I'm"), ("l'll", "I'll"), ("l've", "I've"), ("l'd", "I'd"),
                (" l ", " I "), (" l'", " I'"), ("l ", "I "),
                ("dont", "don't"), ("cant", "can't"), ("wont", "won't"),
                ("youre", "you're"), ("theyre", "they're"), ("thats", "that's"),
                ("Icant", "I can't"), ("Idont", "I don't"), ("Ithink", "I think"),
                ("Im ", "I'm "), ("Imma", "I'm gonna"),
                (" ,", ","), (" .", "."), (" :", ":"), (" ;", ";"),
                ("..", "..."), ("...", "..."), (". . .", "..."), (" . .", "..."),
                ("_", "")
            ]
            auto_fixed = winner_text
            for error, fix in fix_map:
                if error in auto_fixed:
                    auto_fixed = auto_fixed.replace(error, fix)
            candidates.add(auto_fixed)

            def get_score(t):
                if not t or not t.strip():
                    return float('-inf')
                clean_t = " ".join(t.split()).lower()
                words_count = max(len(clean_t.split()), 1)
                return self.lm.score(clean_t, bos=True, eos=True) / words_count

            best_candidate = max(candidates, key=get_score)
            if best_candidate != winner_text:
                _kenlm_key = (winner_text, best_candidate)
                if getattr(self, '_last_kenlm_fix_log', None) != _kenlm_key:
                    log_type = "VETO" if best_candidate == initial_text else "AUTO-FIX"
                    _correction_logs.append(f"[KenLM {log_type}] '{winner_text}' -> '{best_candidate}'")
                    self._last_kenlm_fix_log = _kenlm_key
                winner_text = best_candidate

        sub_id = str(uuid.uuid4())
        self.detection_ready.emit(sub_id, winner_text, box, match_idx, "\n".join(_correction_logs))

# --- OVERLAY WINDOW (The Logic Core) ---
class SubtitleOverlay(QOpenGLWidget):
    start_detection = pyqtSignal(np.ndarray, list, int, str, dict, list)
    start_translation = pyqtSignal(str, str, dict, list)
    subtitle_broadcast = pyqtSignal(str, int, int, int, int, float, float)
    start_heavy_task = pyqtSignal(np.ndarray, list, int, str, dict, list)

    def __init__(self, settings_dict):
        super().__init__()
        self.settings = settings_dict
        self.reader = easyocr.Reader(['en'], gpu=True)
        self.sct = mss.mss()
        self.activity_log_buffer = []

        # 1. Standard Qt Transparency/StaysOnTop
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

        # 2. THE CULPRIT FIX: Apply the extended window style
        self.set_click_through()

        # --- SYMSPELL INITIALIZATION ---
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        # Load the default English dictionary
        self.sym_spell.load_dictionary(dictionary_path, 0, 1)
        
        # 1. CONFIGURE OPENGL SURFACE FOR TRANSPARENCY
        fmt = QSurfaceFormat()
        fmt.setAlphaBufferSize(8) # Enable alpha channel
        fmt.setSamples(4)          # Anti-aliasing for smooth lines
        self.setFormat(fmt)

        # 2. QT WINDOW FLAGS (Click-through & Transparency)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint | 
            Qt.WindowType.WindowStaysOnTopHint | 
            Qt.WindowType.Tool
        )
        
        # Logic State
        self.default_mean = [0, 0, 0, 0]
        self.fixed_mean = [0, 0, 0, 0]
        self.running_mean = [0, 0, 0, 0]
        self.raw_debug_boxes = []
        self.active_text = ""
        self.active_start_time = 0.0
        self.active_coords = [0, 0, 0, 0]
        self.best_match_idx = -1
        self.missing_frames = 0
        self.voting_buffer = [] 
        self.dim_region = None
        self.active_translation = ""  # Final verified Thai text (Step 15)
        self.VOTING_SIZE = 2
        self.dim_alpha = 0.0          # Current opacity of the dim overlay (0–255, drives fade-in)
        self.session_entities = []
        self._bg_is_bright = False
        self.current_sub_id = None
        self.translation_alpha = 0
        self.detection_busy = False
        self.translation_alpha = 0.0
        self.current_sub_id = None
        self._dim_locked = False      # True once a subtitle is committed — box freezes
        self._dim_smooth = None       # Smoothed [x,y,w,h] for the dim box (float)
        
        self.update_geometry()

        # --- STEP 15: THAI FONT LOADING ---
        self.thai_font = None
        try:
            fonts_dir = resource_path(os.path.join("assets", "fonts"))
            font_extensions = (".ttf", ".otf")
            font_files = [
                f for f in os.listdir(fonts_dir)
                if f.lower().endswith(font_extensions)
            ] if os.path.isdir(fonts_dir) else []
            if font_files:
                font_path = os.path.join(fonts_dir, font_files[0])
                font_id = QFontDatabase.addApplicationFont(font_path)
                if font_id != -1:
                    families = QFontDatabase.applicationFontFamilies(font_id)
                    if families:
                        self.thai_font = QFont(families[0], 22, QFont.Weight.Bold)
                        print(f"[INFO] Thai font loaded: {families[0]} from {font_files[0]}")
                    else:
                        print(f"[WARN] Thai font loaded but no families found: {font_path}")
                else:
                    print(f"[WARN] Failed to register Thai font: {font_path}")
            else:
                print("[WARN] No .ttf/.otf fonts found in ./assets/fonts — using system fallback.")
        except Exception as e:
            print(f"[WARN] Thai font loading failed: {e}")
        if self.thai_font is None:
            self.thai_font = QFont("Tahoma", 22, QFont.Weight.Bold)  # Tahoma has decent Thai coverage

        # --- KENLM INITIALIZATION ---
        try:
            # Get the correct path for development or for the packed .exe
            model_path = resource_path('english.bin') 
            self.kenlm_model = kenlm.Model(model_path)
            clean_name = os.path.basename(model_path)
            print(f"[INFO] KenLM model '{clean_name}' successfully initialized.")
        except OSError:
            print(f"[WARN] KenLM model 'english.bin' not found. Context scoring will be disabled.")
            self.kenlm_model = None

        # --- QTHREAD SETUP ---
        self.detection_thread = QThread()
        self.translation_thread = QThread()   # Fast: NLLB / Google / cache hits
        self.refinement_thread = QThread()    # Slow: Typhoon / Gemini — never blocks fast thread

        self.detection_worker = DetectionWorker(self.reader, self.sym_spell, self.kenlm_model, self.settings)
        self.translation_worker = TranslationWorker(self.settings)
        self.refinement_worker = TranslationWorker(self.settings)  # Separate instance for slow engines

        self.detection_worker.moveToThread(self.detection_thread)
        self.translation_worker.moveToThread(self.translation_thread)
        self.refinement_worker.moveToThread(self.refinement_thread)

        # Connect slots
        self.start_detection.connect(self.detection_worker.run_detection)
        self.detection_worker.detection_ready.connect(self.on_detection_finished)
        self.translation_worker.translation_ready.connect(self.on_translation_finished)
        self.refinement_worker.translation_ready.connect(self.on_translation_finished)
        self.start_translation.connect(self.translation_worker.run_translation)

        self.detection_thread.start()
        self.translation_thread.start()
        self.refinement_thread.start()
        
        # Logic loop remains on a timer, Rendering follows GPU sync
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_ocr)
        self.timer.start(250)

        # --- STEP 3: FADE-IN TIMER ---
        # Ticks at 30fps to animate the dim overlay alpha independently of OCR.
        self._fade_timer = QTimer()
        self._fade_timer.timeout.connect(self._tick_fade)
        self._fade_timer.start(33)  # ~30 fps

    def _tick_fade(self):
        """Advance dim overlay alpha. Fades in slowly while pending, snaps to full when translation arrives."""
        if self.dim_region is None or not self.active_text:
            # Snap to invisible instantly when subtitle clears
            if self.dim_alpha != 0.0:
                self.dim_alpha = 0.0
                self.update()
            return

        if self.active_translation:
            # Translation is ready — skip animation, go opaque immediately
            if self.dim_alpha != 255.0:
                self.dim_alpha = 255.0
                self.update()
        else:
            # Translation pending — slowly fade in to obscure_alpha (loading cue)
            target = float(self.settings.get('obscure_alpha', 180))
            if self.dim_alpha < target:
                self.dim_alpha = min(self.dim_alpha + 20.0, target)
                self.update()

    def set_click_through(self):
        """
        Applies Windows-specific flags to make the overlay invisible to the 
        OCR capture and mouse events.
        """
        # CAST TO INT: This is required to make it compatible with ctypes
        hwnd = int(self.winId()) 
        
        # GWL_EXSTYLE = -20
        # WS_EX_TRANSPARENT = 0x20 (Click-through)
        # WS_EX_LAYERED = 0x80000 (Required for transparency)
        
        # Get current style
        style = ctypes.windll.user32.GetWindowLongW(hwnd, -20)
        
        # Set new style with transparency and layered flags
        ctypes.windll.user32.SetWindowLongW(hwnd, -20, style | 0x20 | 0x80000)

    @pyqtSlot(str, str)
    def on_translation_finished(self, sub_id, thai):
        # Sanity log passthrough — not a real translation result
        if "__SANITY_WARN__" in sub_id:
            print(thai)
            return

        # AUTO MODE: fast first-pass result from NLLB
        if "__AUTO_FAST__" in sub_id:
            real_id = sub_id.replace("__AUTO_FAST__", "")
            if real_id != self.current_sub_id:
                return  # Stale
            self.active_translation = thai
            if thai:
                print(f"[TRANSLATE] Auto/NLLB (fast): {thai}")
            self.update()

            # Schedule quality refinement in 2 seconds if text is still on screen
            def _maybe_refine():
                if self.current_sub_id != real_id:
                    return  # Subtitle changed — skip refinement
                mode = self.settings.get('translation_mode', 'Auto')
                api_key = self.settings.get('gemini_api_key', '').strip()
                refine_mode = None
                if mode == 'Auto':
                    # Gemini key takes priority — if provided, use it for the quality refine pass
                    if api_key:
                        refine_mode = 'gemini'
                    else:
                        # No Gemini key — fall back to Typhoon if it's running locally
                        try:
                            r = requests.get("http://localhost:11434/api/tags", timeout=1)
                            models = [m.get('name','') for m in r.json().get('models', [])]
                            if any('typhoon' in m.lower() for m in models):
                                refine_mode = 'typhoon'
                        except Exception:
                            pass
                if not refine_mode:
                    return  # No refiner available
                anime_title = self.settings.get('anime_title', 'Unknown Anime')
                character_names = self.session_entities
                safe_settings = dict(self.settings)
                safe_entities = list(self.session_entities)
                # Emit a refinement translation request with special marker
                from PyQt6.QtCore import QMetaObject, Q_ARG
                _Qt = __import__('PyQt6.QtCore', fromlist=['Qt'])
                QMetaObject.invokeMethod(
                    self.refinement_worker, "run_refinement",
                    _Qt.Qt.ConnectionType.QueuedConnection,
                    Q_ARG(str, real_id),
                    Q_ARG(str, self.active_text),
                    Q_ARG(dict, safe_settings),
                    Q_ARG(list, safe_entities),
                    Q_ARG(str, refine_mode)
                )

            from PyQt6.QtCore import QTimer
            QTimer.singleShot(2000, _maybe_refine)
            return

        # AUTO MODE: quality refinement result from Typhoon / Gemini (arrives ~2s after NLLB fast pass)
        if "__AUTO_REFINE__" in sub_id:
            real_id = sub_id.replace("__AUTO_REFINE__", "")
            if real_id != self.current_sub_id:
                print(f"[TRANSLATE] Typhoon/Gemini refine discarded (subtitle already changed)")
                return
            if thai:
                self.active_translation = thai
                print(f"[TRANSLATE] Auto/Typhoon (refined, applied): {thai}")
                self.update()
            return

        # Ignore stale results from a previous subtitle
        if sub_id != self.current_sub_id:
            return

        self.active_translation = thai
        # Log the translation inline so it appears after TEXT in the activity log
        if thai:
            print(f"[TRANSLATE] {thai}")
        self.update()

    @pyqtSlot(str, str, list, int, str)
    def on_detection_finished(self, sub_id, winner_text, box, match_idx, correction_logs):
        self.detection_busy = False  # Release lock so next frame can be captured

        current_time = time.time()

        # --- CASE A: No text detected ---
        if not winner_text or winner_text.strip() == "":
            if self.active_text:
                self.missing_frames += 1
                if self.missing_frames >= self.settings.get('max_missing', 4):
                    duration = current_time - self.active_start_time
                    if duration >= self.settings.get('min_duration', 0.6):
                        self.emit_sub(current_time)
                    self.active_text = ""
                    self.active_translation = ""
                    self.current_sub_id = None
                    self.voting_buffer.clear()
                    self.missing_frames = 0
                    self.dim_alpha = 0.0
                    self._dim_locked = False
                    self._dim_smooth = None
                    # Snap cyan box back to default centre position
                    if self.settings.get('detection_mode') != 'Fixed':
                        self.running_mean = list(self.default_mean)
                    self.update()
            return

        self.missing_frames = 0

        # --- MULTI-FRAME VOTING ---
        self.voting_buffer.append(winner_text)

        if len(self.voting_buffer) < self.VOTING_SIZE:
            return  # Wait for buffer to fill before committing

        # Pick the most-agreed-upon candidate
        best_score = -1
        voted_text = self.voting_buffer[0]
        for candidate in self.voting_buffer:
            score = sum(fuzz.token_set_ratio(candidate, other) for other in self.voting_buffer)
            if score > best_score:
                best_score = score
                voted_text = candidate

        # Prefer the KenLM-corrected current frame if it's close
        if fuzz.token_set_ratio(winner_text, voted_text) > 80:
            voted_text = winner_text

        # --- SOFT UPDATE: same subtitle, longer/corrected version ---
        if self.active_text:
            sim = max(
                fuzz.partial_ratio(voted_text, self.active_text),
                fuzz.token_set_ratio(voted_text, self.active_text)
            )
            if sim > self.settings.get('similarity', 70):
                if len(voted_text) > len(self.active_text):
                    self.active_text = voted_text
                    self.active_coords = box
                    # Temporarily unlock so the larger box can be absorbed, then re-lock
                    self._dim_locked = False
                    self._update_dim_region(box)
                    self._dim_locked = True
                self.voting_buffer.clear()
                return

        # --- NEW SUBTITLE COMMIT ---
        if self.active_text != voted_text:
            # Finalize old subtitle if it ran long enough
            if self.active_text and (current_time - self.active_start_time) > self.settings.get('min_duration', 0.6):
                self.emit_sub(current_time)

            self.active_text = voted_text
            self.active_coords = box
            self.active_start_time = current_time
            self.active_translation = ""
            self.dim_alpha = 0.0       # Reset fade so new subtitle fades in fresh
            self.current_sub_id = sub_id  # Lock ID — only this translation result is valid
            self._dim_locked = True    # Freeze the dim box — noisy OCR cannot jitter it
            self._force_snap = True
            self.update()  # Render dim immediately

            # LOG in strict order: header → corrections → TEXT → COORDS
            # [TRANSLATE] is emitted later by on_translation_finished once it arrives
            start_local = datetime.fromtimestamp(self.active_start_time).strftime("%H:%M:%S.%f")[:-3]
            print(f"---- SUBTITLE DETECTED AT {start_local} ----")
            if correction_logs:
                print(correction_logs)
            print(f"TEXT:  {self.active_text}")
            print(f"{box}")

            # Request translation — pass snapshots to prevent race conditions
            safe_settings = self.settings.copy()
            safe_entities = list(self.session_entities)
            from PyQt6.QtCore import QMetaObject, Qt as _Qt, Q_ARG
            QMetaObject.invokeMethod(
                self.translation_worker, "run_translation",
                _Qt.ConnectionType.QueuedConnection,
                Q_ARG(str, sub_id),
                Q_ARG(str, self.active_text),
                Q_ARG(dict, safe_settings),
                Q_ARG(list, safe_entities)
            )

        self.voting_buffer.clear()

    def closeEvent(self, event):
        """Ensure all background threads are safely killed before exiting."""
        if hasattr(self, 'timer'):
            self.timer.stop()
        if hasattr(self, '_fade_timer'):
            self._fade_timer.stop()

        # Quit threads and wait for them to cleanly exit
        self.detection_thread.quit()
        self.translation_thread.quit()
        self.refinement_thread.quit()
        
        self.detection_thread.wait()
        self.translation_thread.wait()
        self.refinement_thread.wait()
        
        event.accept()

    def initializeGL(self):
        """Sets up OpenGL state."""
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def resizeGL(self, w, h):
        """Handle window resizing."""
        # The viewport is handled automatically by QOpenGLWidget
        pass

    def paintGL(self):
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

    def paintGL(self):
        GL.glClearColor(0, 0, 0, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        display_style = self.settings.get('display_style', 'Adaptive')

        if not self.settings.get('debug_mode', True):

            if display_style == 'Fixed':
                # ── FIXED STYLE: static VN-style textbox pinned to bottom of overlay ──
                if self.active_text:
                    from PyQt6.QtCore import QRect
                    W = self.width()
                    H = self.height()
                    BOX_H = 90
                    BOX_Y = H - BOX_H
                    PADDING = 16

                    painter.setPen(Qt.PenStyle.NoPen)

                    if self.active_translation:
                        painter.setBrush(QColor(0, 0, 0, 220))
                        painter.drawRect(0, BOX_Y, W, BOX_H)
                        painter.setFont(self.thai_font)
                        painter.setPen(QColor(255, 255, 255, 255))
                        painter.drawText(
                            QRect(PADDING, BOX_Y, W - PADDING * 2, BOX_H),
                            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                            self.active_translation
                        )
                    else:
                        alpha = int(min(self.dim_alpha, 180))
                        if alpha > 0:
                            painter.setBrush(QColor(0, 0, 0, alpha))
                            painter.drawRect(0, BOX_Y, W, BOX_H)

            else:
                # ── ADAPTIVE STYLE: original behaviour — box follows the subtitle ──
                if self.dim_region is not None:
                    dx, dy, dw, dh = self.dim_region
                    painter.setPen(Qt.PenStyle.NoPen)

                    current_alpha = int(self.dim_alpha)
                    if current_alpha <= 0:
                        pass  # Nothing to draw yet — fade hasn't started
                    elif self.active_translation:
                        # --- TRANSLATION READY: blackout box (faded in) + white Thai text ---
                        painter.setBrush(QColor(0, 0, 0, min(current_alpha, 255)))
                        painter.drawRect(int(dx), int(dy), int(dw), int(dh))

                        painter.setFont(self.thai_font)
                        painter.setPen(QColor(255, 255, 255, min(current_alpha, 255)))
                        text_rect = painter.boundingRect(
                            int(dx), int(dy), int(dw), int(dh),
                            Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                            self.active_translation
                        )
                        if text_rect.height() > int(dh):
                            dy_adjusted = int(dy) - (text_rect.height() - int(dh)) // 2
                            draw_rect_h = text_rect.height() + 8
                            painter.setPen(Qt.PenStyle.NoPen)
                            painter.setBrush(QColor(0, 0, 0, min(current_alpha, 255)))
                            painter.drawRect(int(dx), dy_adjusted, int(dw), draw_rect_h)
                            painter.setPen(QColor(255, 255, 255, min(current_alpha, 255)))
                            from PyQt6.QtCore import QRect
                            painter.drawText(
                                QRect(int(dx), dy_adjusted, int(dw), draw_rect_h),
                                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                                self.active_translation
                            )
                        else:
                            from PyQt6.QtCore import QRect
                            painter.drawText(
                                QRect(int(dx), int(dy), int(dw), int(dh)),
                                Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                                self.active_translation
                            )
                    else:
                        # --- TRANSLATION PENDING: partial dim slowly fading in ---
                        painter.setBrush(QColor(0, 0, 0, current_alpha))
                        painter.drawRect(int(dx), int(dy), int(dw), int(dh))

        # --- DEBUG VISUALS (only when debug boxes are ON) ---
        if self.settings.get('debug_mode', True):
            mx, my, mw, mh = self.running_mean
            painter.setPen(QPen(QColor(0, 255, 255, 150), 2))
            painter.setBrush(QColor(0, 255, 255, 30))
            painter.drawRect(int(mx), int(my), int(mw), int(mh))

            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(255, 255, 0, 180), 1))
            for item in self.raw_debug_boxes:
                b = item['box']
                painter.drawRect(b[0], b[1], b[2]-b[0], b[3]-b[1])

            if self.best_match_idx != -1 and self.best_match_idx < len(self.raw_debug_boxes):
                try:
                    painter.setPen(QPen(QColor(0, 255, 0), 3))
                    b = self.raw_debug_boxes[self.best_match_idx]['box']
                    painter.drawRect(b[0], b[1], b[2]-b[0], b[3]-b[1])
                    if self.active_text:
                        painter.setFont(QFont("Arial", 12, QFont.Weight.Bold))
                        painter.setPen(QColor(0, 255, 0))
                        painter.drawText(b[0], b[1]-10, self.active_text)
                except (IndexError, KeyError):
                    pass

        painter.end()

    def update_geometry(self):
        monitor = self.sct.monitors[1]
        h_percent = self.settings['capture_height'] / 100.0
        c_height = int(monitor["height"] * h_percent)
        c_top = monitor["height"] - c_height
        self.region = {"top": c_top, "left": 0, "width": monitor["width"], "height": c_height}
        self.setGeometry(0, c_top, monitor["width"], c_height)

        mw = int(monitor["width"] * 0.4)
        mh = int(c_height * 0.5)
        mx = (monitor["width"] - mw) // 2
        my = (c_height - mh) // 2 
        self.default_mean = [mx, my, mw, mh]

        target_w, target_h = 20, c_height
        target_x, target_y = (monitor["width"] - target_w) // 2, 0 
        self.fixed_mean = [target_x, target_y, target_w, target_h]

        if self.settings.get('detection_mode') == 'Fixed':
            self.running_mean = list(self.fixed_mean)
        elif not self.active_text:
            self.running_mean = list(self.default_mean)

    def _update_dim_region(self, raw_box):
        """
        Smoothly update dim_region from a raw OCR box [x1,y1,x2,y2].

        Rules:
        • Once a subtitle is committed (_dim_locked=True), the box is FROZEN —
          noisy background updates cannot move it at all.
        • On first detection (not locked yet), snap immediately and add padding.
        • A small lerp factor keeps the box from jumping on each OCR result
          before it locks in.
        • The box is only allowed to GROW vertically (absorbs multi-line variance)
          and moves horizontally via lerp — it never shrinks while a subtitle is live.
        """
        PAD_X = 12   # horizontal padding each side (px)
        PAD_Y = 10   # vertical padding each side (px)
        LERP  = 0.25 # how fast the box drifts toward target (0=frozen, 1=instant)

        x1, y1, x2, y2 = raw_box
        # Apply padding
        tx = x1 - PAD_X
        ty = y1 - PAD_Y
        tw = (x2 - x1) + PAD_X * 2
        th = (y2 - y1) + PAD_Y * 2

        if self._dim_locked:
            # Subtitle is committed — box is completely frozen, ignore OCR jitter
            return

        if self._dim_smooth is None:
            # First detection — snap immediately
            self._dim_smooth = [float(tx), float(ty), float(tw), float(th)]
        else:
            sx, sy, sw, sh = self._dim_smooth
            # Lerp position
            nx = sx + LERP * (tx - sx)
            ny = sy + LERP * (ty - sy)
            # Width: lerp toward target
            nw = sw + LERP * (tw - sw)
            # Height: only grow, never shrink while subtitle is live (absorbs line count variance)
            nh = max(sh, sy + sh - ny + LERP * (th - sh)) if self.active_text else th
            self._dim_smooth = [nx, ny, nw, nh]

        sx, sy, sw, sh = self._dim_smooth
        self.dim_region = [int(sx), int(sy), int(sw), int(sh)]

    def run_ocr(self):
        if self.settings.get('detection_mode') == 'Adaptive':
            self.run_ocr_adaptive()
        else:
            self.run_ocr_fixed()

        self.update()

    # --- ADAPTIVE LOGIC ---
    def run_ocr_adaptive(self):
        # PREVENT OVERLAPPING OCR RUNS:
        if self.detection_busy:
            return
        
        # --- STEP 1: SCREEN CAPTURE ---
        screenshot = self.sct.grab(self.region)
        img = np.array(screenshot)

        # --- STEP 2: REGION DETECTION (Fast EasyOCR pass) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Bright background detection
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        bg_brightness = np.median(border_pixels)
        # Hysteresis: only switch mode when brightness crosses a wider gap
        # to prevent flickering when brightness hovers near the threshold
        if self._bg_is_bright and bg_brightness < 100:
            print(f"[OCR MODE] Switched to DARK background mode (Standard) — Brightness: {bg_brightness}")
            self._bg_is_bright = False
        elif not self._bg_is_bright and bg_brightness > 155:
            print(f"[OCR MODE] Switched to BRIGHT background mode (Inverting colors) — Brightness: {bg_brightness}")
            self._bg_is_bright = True
        if self._bg_is_bright:
            gray = cv2.bitwise_not(gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, self.settings['ocr_thresh'], 255, cv2.THRESH_BINARY)
        results = self.reader.readtext(thresh, detail=1)

        sw = self.region["width"]
        prob_threshold = self.settings['prob_limit'] / 100.0

        token_list = []
        valid_boxes = []
        for (bbox, text, prob) in results:
            if prob < prob_threshold:
                continue
            x1 = int(min(bbox[0][0], bbox[3][0]))
            y1 = int(min(bbox[0][1], bbox[1][1]))
            x2 = int(max(bbox[1][0], bbox[2][0]))
            y2 = int(max(bbox[2][1], bbox[3][1]))
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if cx < sw * 0.05 or cx > sw * 0.95:
                continue

            # --- THE "I" FIX (kept — smarter than original plan) ---
            bw, bh = x2 - x1, y2 - y1
            if not (bw > bh * 0.05 and (bw > 2 or bh > 15)):
                continue

            token_list.append({'x': x1, 'x2': x2, 'y': y1, 'y2': y2,
                                'cx': cx, 'cy': cy, 'text': text.strip()})
            valid_boxes.append([x1, y1, x2, y2])

        merged = self.proximity_merge_adaptive(valid_boxes)

        final_data = []
        for m_box in merged:
            pad = 10
            parts = [t for t in token_list
                     if (m_box[0] - pad) <= t['cx'] <= (m_box[2] + pad)
                     and (m_box[1] - pad) <= t['cy'] <= (m_box[3] + pad)]
            if parts:
                token_heights = sorted([p['y2'] - p['y'] for p in parts])
                median_h = token_heights[len(token_heights) // 2]
                row_bucket = max(8, int(median_h * 0.6))
                parts.sort(key=lambda p: (((p['y'] + p['y2']) // 2) // row_bucket, p['x']))
                joined_text = " ".join([p['text'] for p in parts])
                final_data.append({'box': m_box, 'text': joined_text})

        self.raw_debug_boxes = final_data

        # --- STEP 4: HALLUCINATION FILTER ---
        # Use the running_mean (cyan box) to find the best matching text region via IoU.
        best_iou, self.best_match_idx, winner_text = -1, -1, ""
        mx, my, mw, mh = self.running_mean
        mean_cx, mean_cy = mx + mw / 2, my + mh / 2

        for i, item in enumerate(self.raw_debug_boxes):
            box = item['box']
            ix1, iy1 = max(mx, box[0]), max(my, box[1])
            ix2, iy2 = min(mx+mw, box[2]), min(my+mh, box[3])

            # --- CENTER HIT BONUS (kept — smarter than plain IoU) ---
            center_hit = (box[0] <= mean_cx <= box[2]) and (box[1] <= mean_cy <= box[3])

            if (ix2 > ix1 and iy2 > iy1) or center_hit:
                if center_hit and not (ix2 > ix1 and iy2 > iy1):
                    iou = 0.21
                else:
                    inter = (ix2 - ix1) * (iy2 - iy1)
                    union = (mw * mh) + ((box[2]-box[0]) * (box[3]-box[1])) - inter
                    iou = inter / union if union > 0 else 0

                if iou > 0.02:
                    if len(item['text']) > len(winner_text):
                        best_iou, self.best_match_idx, winner_text = iou, i, item['text']
                    elif len(item['text']) == len(winner_text) and iou > best_iou:
                        best_iou, self.best_match_idx, winner_text = iou, i, item['text']

        current_time = time.time()

        if self.best_match_idx != -1:
            box = self.raw_debug_boxes[self.best_match_idx]['box']

            # --- STEP 6: UX TRIGGER (DIMMING) ---
            self._update_dim_region(box)

            # --- RUBBER BAND SPATIAL MEAN ---
            # Current mean: [x, y, w, h]
            m_left, m_top, m_width, m_height = self.running_mean
            m_right, m_bottom = m_left + m_width, m_top + m_height

            # Target box from OCR: [x1, y1, x2, y2]
            b_left, b_top, b_right, b_bottom = box

            # Get tracking speed from slider (0.01 to 1.0)
            alpha = self.settings.get('alpha', 10) / 100.0

            if getattr(self, '_force_snap', False):
                new_left, new_top, new_right, new_bottom = b_left, b_top, b_right, b_bottom
                self._force_snap = False
            else:
                # RUBBER BAND LOGIC:
                # If target is SMALLER (shrink), snap immediately.
                # If target is LARGER (expand), drift slowly via alpha.
                new_left   = b_left   if b_left   > m_left   else (alpha * b_left   + (1.0 - alpha) * m_left)
                new_right  = b_right  if b_right  < m_right  else (alpha * b_right  + (1.0 - alpha) * m_right)
                new_top    = b_top    if b_top    > m_top    else (alpha * b_top    + (1.0 - alpha) * m_top)
                new_bottom = b_bottom if b_bottom < m_bottom else (alpha * b_bottom + (1.0 - alpha) * m_bottom)

            # Update the running mean for the next frame
            self.running_mean = [new_left, new_top, new_right - new_left, new_bottom - new_top]

        else:
            # --- NO MATCH FOUND ---
            self.dim_region = None
            self._dim_smooth = None
            self._dim_locked = False
            if not self.active_text:
                # Snap cyan box back to default when screen is known-empty
                self.running_mean = list(self.default_mean)

        # RIGHT BEFORE emitting the signal to the worker, lock it:
        self.detection_busy = True
        # Determine the box to pass (must be a list)
        box_to_pass = self.dim_region if self.dim_region is not None else []
        
        # Pass all 6 arguments: img, box, match_idx, initial_text, settings, entities
        self.start_detection.emit(img, box_to_pass, self.best_match_idx, winner_text, self.settings, self.session_entities)

    # --- FIXED LOGIC ---
    def run_ocr_fixed(self):
        # PREVENT OVERLAPPING OCR RUNS:
        if self.detection_busy:
            return
        
        # --- STEP 1: SCREEN CAPTURE ---
        screenshot = self.sct.grab(self.region)
        img = np.array(screenshot)

        # --- STEP 2: REGION DETECTION (Fast EasyOCR pass) ---
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

        # Bright background detection
        border_pixels = np.concatenate([gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]])
        if np.median(border_pixels) > 127:
            gray = cv2.bitwise_not(gray)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, thresh = cv2.threshold(gray, self.settings['ocr_thresh'], 255, cv2.THRESH_BINARY)
        thresh_padded = cv2.copyMakeBorder(thresh, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        results = self.reader.readtext(thresh_padded, detail=1)

        sw = self.region["width"]
        prob_threshold = self.settings['prob_limit'] / 100.0

        token_list = []
        valid_boxes = []
        for (bbox, text, prob) in results:
            if prob < prob_threshold:
                continue
            # Subtract the 50px padding added by copyMakeBorder
            x1 = int(min(bbox[0][0], bbox[3][0])) - 50
            y1 = int(min(bbox[0][1], bbox[1][1])) - 50
            x2 = int(max(bbox[1][0], bbox[2][0])) - 50
            y2 = int(max(bbox[2][1], bbox[3][1])) - 50
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if cx < sw * 0.05 or cx > sw * 0.95:
                continue
            token_list.append({'x': x1, 'x2': x2, 'y': y1, 'y2': y2, 'cx': cx, 'cy': cy, 'text': text.strip()})
            valid_boxes.append([x1, y1, x2, y2])

        merged = self.proximity_merge_fixed(valid_boxes)
        final_data = []
        for m_box in merged:
            parts = [t for t in token_list if m_box[0]-5 <= t['cx'] <= m_box[2]+5 and m_box[1]-5 <= t['cy'] <= m_box[3]+5]
            if parts:
                token_heights = sorted([p['y2'] - p['y'] for p in parts])
                median_h = token_heights[len(token_heights) // 2]
                row_bucket = max(8, int(median_h * 0.6))
                parts.sort(key=lambda p: (int(p['cy']) // row_bucket, p['x']))
                joined_text = " ".join([p['text'] for p in parts])
                final_data.append({'box': m_box, 'text': joined_text})

        self.raw_debug_boxes = final_data

        # --- STEP 4: HALLUCINATION FILTER ---
        # In Fixed mode the running_mean is the fixed centre column — anything overlapping wins.
        self.best_match_idx = -1
        winner_text = ""
        mx, my, mw, mh = self.running_mean

        for i, item in enumerate(self.raw_debug_boxes):
            box = item['box']
            ix1, iy1 = max(mx, box[0]), max(my, box[1])
            ix2, iy2 = min(mx+mw, box[2]), min(my+mh, box[3])
            if ix2 > ix1 and iy2 > iy1:
                if len(item['text']) > len(winner_text):
                    self.best_match_idx = i
                    winner_text = item['text']

        current_time = time.time()

        if self.best_match_idx != -1:
            box = self.raw_debug_boxes[self.best_match_idx]['box']

            # --- STEP 6: UX TRIGGER (DIMMING) ---
            self._update_dim_region(box)

        else:
            self.dim_region = None
            self._dim_smooth = None
            self._dim_locked = False
            # State transitions handled exclusively in on_detection_finished

        # RIGHT BEFORE emitting the signal to the worker, lock it:
        self.detection_busy = True
        # Get the correct bounding box format for adaptive mode
        box_to_pass = self.raw_debug_boxes[self.best_match_idx]['box'] if self.best_match_idx != -1 else []
        
        self.start_detection.emit(img, box_to_pass, self.best_match_idx, winner_text, self.settings, self.session_entities)

    def emit_sub(self, end_time):
        self.subtitle_broadcast.emit(self.active_text, *self.active_coords, self.active_start_time, end_time)
        end_local = datetime.fromtimestamp(end_time).strftime("%H:%M:%S.%f")[:-3]
        print(f"--- SUBTITLE ENDED AT {end_local} ---\n")

    def proximity_merge_adaptive(self, boxes):
        if not boxes: return []
        # Initial vertical sort to process top-to-bottom
        boxes.sort(key=lambda x: (x[1] // 10, x[0])) 
        
        merged_clusters = []
        while boxes:
            curr = boxes.pop(0)
            cluster = [curr]
            changed = True
            while changed:
                changed = False
                for i in range(len(boxes)-1, -1, -1):
                    # Distance math
                    dist_x = max(0, boxes[i][0] - max(c[2] for c in cluster), min(c[0] for c in cluster) - boxes[i][2])
                    dist_y = max(0, boxes[i][1] - max(c[3] for c in cluster), min(c[1] for c in cluster) - boxes[i][3])
                    
                    curr_h = max(c[3]-c[1] for c in cluster)
                    adaptive_y = max(40, int(curr_h * 1.5))

                    if dist_x < 180 and dist_y < adaptive_y:
                        cluster.append(boxes.pop(i))
                        changed = True
            
            # --- Sort words in the cluster left-to-right ---
            cluster.sort(key=lambda x: (x[1] // 10, x[0])) 
            
            m_box = [min(c[0] for c in cluster), min(c[1] for c in cluster), 
                     max(c[2] for c in cluster), max(c[3] for c in cluster)]
            merged_clusters.append(m_box)
            
        return merged_clusters

    def proximity_merge_fixed(self, boxes):
        if not boxes: return []
        boxes.sort(key=lambda x: x[1]) 
        merged = []
        while boxes:
            curr = boxes.pop(0)
            changed = True
            while changed:
                changed = False
                for i in range(len(boxes)-1, -1, -1):
                    dx = max(0, boxes[i][0] - curr[2], curr[0] - boxes[i][2])
                    dy = max(0, boxes[i][1] - curr[3], curr[1] - boxes[i][3])
                    if dx < 80 and dy < 25:
                        nxt = boxes.pop(i)
                        curr = [min(curr[0], nxt[0]), min(curr[1], nxt[1]), max(curr[2], nxt[2]), max(curr[3], nxt[3])]
                        changed = True
            merged.append(curr)
        return merged
    
    def fetch_anime_characters(self, anime_name):
        # We only query for titles (to confirm the show) and character names
        query = '''
        query ($search: String) {
          Media (search: $search, type: ANIME) {
            title { english romaji }
            characters (sort: [ROLE, RELEVANCE], perPage: 25) {
              nodes { name { full } }
            }
          }
        }
        '''
        url = 'https://graphql.anilist.co'
        try:
            print(f"[SYSTEM] Manually fetching characters for '{anime_name}'...")
            response = requests.post(url, json={'query': query, 'variables': {'search': anime_name}}, timeout=5)

            if response.status_code == 500:
                print("[ERROR] AniList is currently down or having server issues (Error 500). Please try again in a few minutes.")
                return []
            
            if response.status_code == 200:
                data = response.json()['data']['Media'] 
                
                # Get the full names for the log
                full_character_names = [c['name']['full'] for c in data['characters']['nodes'] if c['name'].get('full')]
                
                # Split names into fragments for the actual OCR Shield (e.g. "Sousou no Frieren" -> "Frieren")
                fragments = []
                for name in full_character_names:
                    fragments.extend(name.split())
                
                # Update the shield with unique fragments
                self.session_entities = list(set([f for f in fragments if f]))
                
                # Detailed Log output as requested
                anime_title = data['title'].get('english') or data['title'].get('romaji')

                # --- Pass anime context into shared settings for translation prompt ---
                self.settings['anime_title']      = anime_title or 'Unknown Anime'
                self.settings['session_entities'] = full_character_names  # full names for prompt context

                print(f"---- ENTITY SHIELD UPDATED ----")
                print(f"ANIME: {anime_title}")
                print(f"FETCHED CHARACTERS:")
                for name in sorted(full_character_names):
                    print(f" • {name}")
                print(f"TOTAL SHIELD FRAGMENTS: {len(self.session_entities)}")
                print(f"-------------------------------")
            else:
                print(f"[ERROR] AniList API Error: {response.status_code}")
        except Exception as e:
            print(f"[ERROR] Fetch failed: {e}")
        return []

class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Subtitle OCR Controller")
        self.resize(600, 350)  # height adjusted after init via adjustSize()
        self.activity_log_buffer = []
        
        # 1. Define defaults and settings first
        self.defaults = {
            'capture_height': 25, 
            'ocr_thresh': 235, 
            'similarity': 70,
            'max_missing': 4, 
            'min_duration': 0.6, 
            'alpha': 10,       
            'prob_limit': 45,  
            'debug_mode': False,
            'show_console': False,
            'detection_mode': 'Fixed',
            'obscure_enabled': False,
            'obscure_alpha': 200,
            'high_res_crop': True,
            'smart_i_recovery': True,
            'translation_mode': 'Auto',
            'gemini_api_key': '',
            'labse_enabled': True,
            'anime_title': 'Unknown Anime',
            'session_entities': [],
            'display_style': 'Adaptive',
        }
        self.settings = dict(self.defaults)
        
        # 2. Initialize the storage dictionary BEFORE building the UI
        self.slider_widgets = {} 
        
        # 3. Now build the UI
        self.init_ui()
        
    def init_ui(self):
        # 1. Create the layout and tabs FIRST
        self.main_layout = QVBoxLayout(self)
        self.tabs = QTabWidget()
        
        # ==========================================
        # --- TAB 1: BASIC SETTINGS (For Normal Users)
        # ==========================================
        self.tab_basic = QWidget()
        basic_layout = QVBoxLayout(self.tab_basic)

        # --- TRANSLATION MODE DROPDOWN ---
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("<b>Translation Engine:</b>"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Auto", "Google Translate (Fast)", "Typhoon 1.5 (Accurate)", "Gemini (Balance)", "NLLB-200 (Offline)"])
        self.mode_combo.setCurrentText("Auto")
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        basic_layout.addLayout(mode_layout)

        # --- TRANSLATION DISPLAY STYLE DROPDOWN ---
        style_layout = QHBoxLayout()
        style_layout.addWidget(QLabel("<b>Translation Display Style:</b>"))
        self.style_combo = QComboBox()
        self.style_combo.addItems(["Adaptive", "Fixed"])
        self.style_combo.setCurrentText("Adaptive")
        self.style_combo.setToolTip(
            "Adaptive: translation box follows the subtitle position on screen.\n"
            "Fixed: translation always appears in a static box at the bottom of the screen."
        )
        self.style_combo.currentTextChanged.connect(self.on_display_style_changed)
        style_layout.addWidget(self.style_combo)
        basic_layout.addLayout(style_layout)

        # --- TOGGLEABLE GEMINI BOX ---
        self.gemini_group = QGroupBox("Google Gemini Configuration (Optional)")
        g_layout = QVBoxLayout()

        key_row = QHBoxLayout()
        self.api_key_input = QLineEdit()
        self.api_key_input.setPlaceholderText("Paste Gemini API Key here...")
        self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key_input.textChanged.connect(lambda: self.settings.update({'gemini_api_key': self.api_key_input.text()}))

        self.toggle_key_btn = QPushButton("👁")
        self.toggle_key_btn.setFixedWidth(30)
        self.toggle_key_btn.setCheckable(True)
        self.toggle_key_btn.clicked.connect(self.toggle_api_visibility)

        self.verify_key_btn = QPushButton("Verify Key")
        self.verify_key_btn.setFixedWidth(80)
        self.verify_key_btn.clicked.connect(self.verify_gemini_key)
        self.api_key_input.returnPressed.connect(self.verify_gemini_key)

        key_row.addWidget(self.api_key_input)
        key_row.addWidget(self.toggle_key_btn)
        key_row.addWidget(self.verify_key_btn)
        g_layout.addLayout(key_row)

        self.api_status_label = QLabel("")  # "Valid Key" feedback
        g_layout.addWidget(self.api_status_label)

        self.gemini_group.setLayout(g_layout)
        self.gemini_group.setVisible(True)  # Visible for Auto (default) and Gemini modes
        basic_layout.addWidget(self.gemini_group)
        
        # 2. Character Entity Shield (Specific Feature)
        shield_group = QGroupBox("Character Entity Shield")
        shield_layout = QHBoxLayout()
        self.anime_search = QLineEdit()
        self.anime_search.setPlaceholderText("Enter Anime Name (Optional but recommended)")
        self.anime_search.returnPressed.connect(self.run_shield_fetch)

        btn_fetch = QPushButton("Fetch Characters")
        btn_fetch.setFixedWidth(120)
        btn_fetch.clicked.connect(self.run_shield_fetch)

        shield_layout.addWidget(self.anime_search)
        shield_layout.addWidget(btn_fetch)
        shield_group.setLayout(shield_layout)
        basic_layout.addWidget(shield_group)

        # Mode Selector
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Tracking Logic:"))
        self.mode_selector = QComboBox()
        self.mode_selector.addItems(["Fixed", "Adaptive"])
        self.mode_selector.currentTextChanged.connect(lambda v: self.update_setting('detection_mode', v))
        mode_layout.addWidget(self.mode_selector)
        basic_layout.addLayout(mode_layout)

        # Detection Zone Height
        self.add_control(basic_layout, "Detection Zone Height (%)", 'capture_height', 5, 100, "Vertical slice of the screen to monitor.")
        
        # Smart "I" Recovery Checkbox
        self.smart_i_cb = QCheckBox("Enable Smart 'I' Recovery")
        self.smart_i_cb.setChecked(self.settings['smart_i_recovery'])
        self.smart_i_cb.setToolTip("Automatically adds 'I' to the start of likely incomplete sentences (e.g. 'love you' -> 'I love you')")
        self.smart_i_cb.toggled.connect(lambda v: self.settings.update({'smart_i_recovery': v}))
        basic_layout.addWidget(self.smart_i_cb)

        # Hide Overlay Checkbox
        self.hide_overlay_cb = QCheckBox("Hide Visual Overlay")
        self.hide_overlay_cb.toggled.connect(self.toggle_hide_overlay)
        basic_layout.addWidget(self.hide_overlay_cb)

        basic_layout.addStretch() # Pushes basic items to the top
        self.tabs.addTab(self.tab_basic, "Basic Settings")

        # ==========================================
        # --- TAB 2: ADVANCED TUNING
        # ==========================================
        self.tab_advanced = QWidget()
        advanced_layout = QVBoxLayout(self.tab_advanced)

        checkbox_row_widget = QWidget()
        checkbox_row = QHBoxLayout(checkbox_row_widget)
        checkbox_row.setContentsMargins(0, 0, 0, 5)

        self.debug_cb = QCheckBox("Debug Boxes")
        self.debug_cb.setChecked(self.settings['debug_mode'])
        self.debug_cb.toggled.connect(self.toggle_debug)

        self.console_cb = QCheckBox("Show Activity Log (Console)")
        self.console_cb.setChecked(self.settings['show_console'])
        self.console_cb.toggled.connect(self.toggle_console)

        checkbox_row.addWidget(self.console_cb)
        checkbox_row.addWidget(self.debug_cb)

        # Add the row to the top of the advanced layout
        advanced_layout.addWidget(checkbox_row_widget)
        
        self.add_control(advanced_layout, "White Text Sensitivity", 'ocr_thresh', 50, 255, "Sensitivity to white text vs dark backgrounds.")
        self.add_control(advanced_layout, "New Subtitle Detection Sensitivity (%)", 'similarity', 0, 100, "How strict the app is when deciding if text changed.")
        self.add_control(advanced_layout, "Flicker Protection (Frames)", 'max_missing', 1, 20, "How long text stays on screen if OCR misses a frame.")
        self.add_control(advanced_layout, "Ignore Background Clutter (Strictness %)", 'prob_limit', 0, 100, "Minimum OCR confidence to consider valid text.")
        self.add_control(advanced_layout, "Tracking Box Expand Speed", 'alpha', 1, 100, "1 = Slow Drift, 100 = Instant Snap.")

        if self.settings['detection_mode'] == 'Fixed':
            self.slider_widgets['alpha']['row'].setVisible(False)
        
        # Global Reset Button
        btn_reset_adv = QPushButton("Reset Advanced Tuning to Defaults")
        btn_reset_adv.setStyleSheet("background-color: #aa3333; color: white; padding: 5px; font-weight: bold;")
        btn_reset_adv.clicked.connect(self.reset_advanced_defaults)
        advanced_layout.addWidget(btn_reset_adv)

        # --- STEP 14: SANITY CHECK (Cache stats + Clear) ---
        sanity_group = QGroupBox("Translation Cache")
        sanity_layout = QVBoxLayout()
        
        # Cache stats row
        cache_stats_row = QHBoxLayout()
        self.cache_stats_label = QLabel("Cache: — entries | — hits / — misses (—%)")
        self.cache_stats_label.setStyleSheet("color: #aaaaaa; font-size: 9pt;")
        cache_stats_row.addWidget(self.cache_stats_label)
        cache_stats_row.addStretch()
        
        # Refresh Button
        refresh_stats_btn = QPushButton("↻")
        refresh_stats_btn.setFixedWidth(28)
        refresh_stats_btn.setToolTip("Refresh cache hit/miss statistics")
        # FIX: Connect the click event
        refresh_stats_btn.clicked.connect(self.refresh_cache_stats) 
        cache_stats_row.addWidget(refresh_stats_btn)
        
        sanity_layout.addLayout(cache_stats_row)

        # Clear Cache Button
        btn_clear_cache = QPushButton("🗑 Clear Translation Cache")
        btn_clear_cache.setStyleSheet("background-color: #555533; color: white; padding: 4px; font-weight: bold;")
        # FIX: Connect the click event
        btn_clear_cache.clicked.connect(self.clear_translation_cache)
        
        sanity_layout.addWidget(btn_clear_cache)
        sanity_group.setLayout(sanity_layout)

        advanced_layout.addStretch()
        advanced_layout.addWidget(sanity_group)
        self.tabs.addTab(self.tab_advanced, "Advanced Tuning")

        self.main_layout.addWidget(self.tabs)

        # ==========================================
        # --- FOOTER (Always Visible)
        # ==========================================
        self.console_label = QLabel("<b>Activity Log:</b>")
        self.main_layout.addWidget(self.console_label)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setStyleSheet("background: #111; color: #eee; font-family: Consolas; font-size: 10pt;")
        self.main_layout.addWidget(self.console)
        
        # Hide console initially if debug mode is off
        if not self.settings['show_console']:
            self.console_label.hide()
            self.console.hide()

        self.setLayout(self.main_layout)

        # Start Output Logging
        self.emitter = ConsoleEmitter()
        self.emitter.text_written.connect(self.log)
        sys.stdout = OutStream(self.emitter)

        # 4. MOVE THIS TO THE VERY END
        # Now when KenLM initializes, it will find the console and print there!
        self.overlay = SubtitleOverlay(self.settings)
        self.toggle_console(self.settings['show_console'])
        # Connect Worker to UI for key verification
        self.overlay.translation_worker.gemini_status.connect(self.update_api_status)
        # Auto-refresh cache stats label after every translation result
        self.overlay.translation_worker.translation_ready.connect(lambda *_: self.refresh_cache_stats())
        self.overlay.refinement_worker.translation_ready.connect(lambda *_: self.refresh_cache_stats())
        self.overlay.show()
        try:
            hwnd = int(self.overlay.winId())
            ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
        except Exception as e:
            print(f"[WARN] SetWindowDisplayAffinity failed: {e}")

        # Start LaBSE loading in background — app is fully usable while it loads
        if self.settings.get('labse_enabled', True):
            _preload_labse_background()

        # Start NLLB loading in background so it's ready when Auto mode needs it
        _preload_nllb_background()

        # Prime the cache stats display now that UI is fully built
        self.refresh_cache_stats()

        # Fit window height to actual content — no blank space at the bottom
        self.resize(self.width(), self.sizeHint().height())

    def closeEvent(self, event):
        """When the control panel closes, safely shut down the overlay threads."""
        print("[INFO] Control Panel closing, cleaning up overlay...")
        if hasattr(self, 'overlay'):
            # This triggers the SubtitleOverlay.closeEvent you already wrote
            self.overlay.close() 
        event.accept()

    def add_control(self, parent_layout, label, key, v_min, v_max, tooltip=""):
        row_widget = QWidget()
        row = QHBoxLayout(row_widget)
        row.setContentsMargins(0, 0, 0, 0)

        lbl = QLabel(f"{label}:")
        lbl.setToolTip(tooltip)
        row.addWidget(lbl)
        
        sld = QSlider(Qt.Orientation.Horizontal)
        sld.setRange(v_min, v_max)
        sld.setValue(int(self.settings[key]))
        sld.setToolTip(tooltip)
        
        txt = QLineEdit(str(self.settings[key]))
        txt.setFixedWidth(45)
        
        btn = QPushButton("Reset")
        btn.setFixedWidth(50)

        sld.valueChanged.connect(lambda v: self.sync(key, v, txt))
        txt.editingFinished.connect(lambda: self.sync_txt(key, txt, sld))
        btn.clicked.connect(lambda: self.sync(key, self.defaults[key], txt, sld))

        row.addWidget(sld)
        row.addWidget(txt)
        row.addWidget(btn)
        
        parent_layout.addWidget(row_widget)
        
        # Store references so the Global Reset button can update them
        self.slider_widgets[key] = {'row': row_widget, 'slider': sld, 'text': txt}

    def reset_advanced_defaults(self):
        adv_keys = ['ocr_thresh', 'similarity', 'max_missing', 'prob_limit', 'alpha']
        for key in adv_keys:
            val = self.defaults[key]
            self.settings[key] = val
            if key in self.slider_widgets:
                self.slider_widgets[key]['slider'].setValue(val)
                self.slider_widgets[key]['text'].setText(str(val))
        print("[SYSTEM] Advanced settings reset to default.")

    def sync(self, key, val, txt, sld=None):
        self.settings[key] = val
        txt.setText(str(val))
        if sld: sld.setValue(int(val))
        if key == 'capture_height': self.overlay.update_geometry()

    def sync_txt(self, key, txt, sld):
        try:
            val = int(txt.text())
            sld.setValue(val)
        except: pass

    def update_setting(self, key, val):
        old_val = self.settings.get(key)
        self.settings[key] = val
        
        if key == 'detection_mode' and old_val != val:
            if val == 'Fixed':
                self.slider_widgets['alpha']['row'].setVisible(False)
                if hasattr(self, 'overlay'):
                    self.overlay.running_mean = list(self.overlay.fixed_mean)
            else:
                self.slider_widgets['alpha']['row'].setVisible(True)
                if hasattr(self, 'overlay'):
                    self.overlay.running_mean = list(self.overlay.default_mean)
                    self.overlay.best_match_idx = -1

    def on_mode_changed(self, text):
        # Map display label → internal mode key
        _mode_map = {
            "Auto":                        "Auto",
            "Google Translate (Fast)":     "Google Translate",
            "Typhoon 1.5 (Accurate)":      "Typhoon 1.5",
            "Gemini (Balance)":  "Gemini",
            "NLLB-200 (Offline)":          "NLLB-200",
        }
        self.settings['translation_mode'] = _mode_map.get(text, text)
        # Show Gemini key box for "Auto" and "Gemini" modes
        is_gemini_relevant = self.settings['translation_mode'] in ("Gemini", "Auto")
        self.gemini_group.setVisible(is_gemini_relevant)
        if self.settings['translation_mode'] == "Auto":
            self.gemini_group.setTitle("Google Gemini Configuration (Optional)")
        else:
            self.gemini_group.setTitle("Google Gemini Configuration")

        # Force the layout to reflow and resize the window correctly.
        self.gemini_group.updateGeometry()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, self._refit_window)

    def on_display_style_changed(self, text):
        self.settings['display_style'] = text
        if hasattr(self, 'overlay'):
            self.overlay.update()

    def _refit_window(self):
        """Shrink or expand height only — never touch the width."""
        if not self.settings.get('show_console', False):
            self.resize(self.width(), self.sizeHint().height())

    def verify_gemini_key(self):
        """Actively tests the Gemini API key and updates the status label immediately."""
        import threading
        key = self.api_key_input.text().strip()
        if not key:
            self.api_status_label.setText("⚠ Please enter a key first")
            self.api_status_label.setStyleSheet("color: #ffaa00; font-weight: bold;")
            return
        self.api_status_label.setText("⏳ Checking...")
        self.api_status_label.setStyleSheet("color: #aaaaaa; font-weight: bold;")
        self.verify_key_btn.setEnabled(False)

        def _check():
            try:
                import google.generativeai as genai
                genai.configure(api_key=key)
                for model_name in ('gemini-2.5-flash', 'gemini-1.5-flash', 'gemini-2.0-flash'):
                    try:
                        model = genai.GenerativeModel(model_name)
                        model.generate_content("Hi")
                        print(f"[GEMINI] Key verification: success using {model_name}")
                        break
                    except Exception as e:
                        print(f"[GEMINI] {model_name} failed: {e}")
                        continue
                else:
                    raise RuntimeError("All models failed")
                valid = True
            except Exception as e:
                valid = False
                print(f"[GEMINI] Key verification failed: {e}")
            # Use the existing gemini_status signal to safely update UI from thread
            if hasattr(self, 'overlay') and hasattr(self.overlay, 'translation_worker'):
                self.overlay.translation_worker.gemini_status.emit(valid)
            else:
                # Overlay not started yet — use a QTimer to call back on main thread
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self._on_verify_done(valid))

        threading.Thread(target=_check, daemon=True).start()

    def _on_verify_done(self, is_valid):
        self.verify_key_btn.setEnabled(True)
        self.update_api_status(is_valid)

    def update_api_status(self, is_valid):
        self.verify_key_btn.setEnabled(True)
        if is_valid:
            self.api_status_label.setText("✔ Valid Key")
            self.api_status_label.setStyleSheet("color: #39ff14; font-weight: bold;")
        else:
            self.api_status_label.setText("✘ Key Error / Quota Full")
            self.api_status_label.setStyleSheet("color: #ff4444; font-weight: bold;")

    def toggle_api_visibility(self):
        """Toggles masked dots vs plain text."""
        if self.toggle_key_btn.isChecked():
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Normal)
            self.toggle_key_btn.setText("🔒")
        else:
            self.api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
            self.toggle_key_btn.setText("👁")

    def sync_gemini_settings(self):
        """Updates the shared settings dictionary with the API key from the UI."""
        self.settings['gemini_api_key'] = self.api_key_input.text().strip()

    def toggle_debug(self, val):
        self.settings['debug_mode'] = val
        if hasattr(self, 'overlay'):
            self.overlay.update() # This updates the cyan/green boxes
        print(f"[SYSTEM] Debug Boxes {'Enabled' if val else 'Disabled'}.")

    def toggle_console(self, val):
        self.settings['show_console'] = val
        if hasattr(self, 'console'):
            self.console_label.setVisible(val)
            self.console.setVisible(val)

            if not val:
                # Use a timer so Qt finishes collapsing the widget before we measure height.
                # Without this, sizeHint() still returns the old expanded height.
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(0, lambda: self.resize(self.width(), self.sizeHint().height()))
            else:
                self.resize(self.width(), 750)
        print(f"[SYSTEM] Activity Log {'Visible' if val else 'Hidden'}.")

    def toggle_hide_overlay(self, hidden):
        if hasattr(self, 'overlay'):
            if hidden:
                self.overlay.hide()
                print("[SYSTEM] Overlay Hidden: OCR running in background.")
            else:
                self.overlay.show()
                # Re-apply window affinity to keep it off screen-recordings
                hwnd = int(self.overlay.winId())
                ctypes.windll.user32.SetWindowDisplayAffinity(hwnd, 0x11)
                print("[SYSTEM] Overlay Visible.")

    def run_shield_fetch(self):
        name = self.anime_search.text()
        if name and hasattr(self, 'overlay'):
            # Just trigger the fetch; the overlay handles the rest
            self.overlay.fetch_anime_characters(name)

    # --- STEP 14: SANITY CHECK HELPERS ---
    def clear_translation_cache(self):
        """Delete the translation cache file and reset in-memory hit/miss counters."""
        import os
        if os.path.exists("translations_cache.json"):
            try:
                os.remove("translations_cache.json")
                print("[CACHE] Translation cache cleared.")
            except Exception as e:
                print(f"[CACHE] Could not delete cache: {e}")
        else:
            print("[CACHE] Cache was already empty — nothing to clear.")

        # FIX: Changed 'heavy_worker' to 'translation_worker'
        if hasattr(self, 'overlay') and hasattr(self.overlay, 'translation_worker'):
            self.overlay.translation_worker.cache_hits = 0
            self.overlay.translation_worker.cache_misses = 0
        if hasattr(self, 'overlay') and hasattr(self.overlay, 'refinement_worker'):
            self.overlay.refinement_worker.cache_hits = 0
            self.overlay.refinement_worker.cache_misses = 0
            
        self.refresh_cache_stats()

    def refresh_cache_stats(self):
        """Read in-memory hit/miss counters from the worker and update the stats label."""
        import os, json
        try:
            # Cache file entry count
            cache_size = 0
            if os.path.exists("translations_cache.json"):
                with open("translations_cache.json", "r", encoding="utf-8") as cf:
                    cache_size = len(json.load(cf))
                    
            # In-memory counters from both live workers
            hits   = 0
            misses = 0
            
            # FIX: Changed 'heavy_worker' to 'translation_worker'
            if hasattr(self, 'overlay') and hasattr(self.overlay, 'translation_worker'):
                hits   += self.overlay.translation_worker.cache_hits
                misses += self.overlay.translation_worker.cache_misses
            if hasattr(self, 'overlay') and hasattr(self.overlay, 'refinement_worker'):
                hits   += self.overlay.refinement_worker.cache_hits
                misses += self.overlay.refinement_worker.cache_misses
                
            total = hits + misses
            rate  = round(hits / total * 100, 1) if total > 0 else 0.0
            
            self.cache_stats_label.setText(
                f"Cache: {cache_size} entries | {hits} hits / {misses} misses  ({rate}%)"
            )
        except Exception:
            self.cache_stats_label.setText("Cache: (stats unavailable)")

    @pyqtSlot(str)
    def log(self, msg):
        # 1. Color Palette
        palette = {
            "TEXT": "#39ff14",     # Neon Green
            "FIX": "#f2cc60",      # Gold
            "VETO": "#ff7b72",     # Red
            "SPELL": "#d2a8ff",    # Purple
            "TRANSLATE": "#00ffff",    # Cyan
            "SHIELD": "#ffa657",   # Orange
            "LABEL": "#58a6ff",    # Soft Blue
            "SYSTEM": "#8b949e",   # Muted Gray
            "SANITY": "#ff9900"    # Amber — for sanity warnings/fixes
        }

        final_html = msg 
        msg_s = msg.strip()

        # CASE 1: System Boundaries (Dashes) -> Gray & Italic
        if "---" in msg_s and ("SUBTITLE" in msg_s or "ENTITY" in msg_s or msg_s.count("-") > 10):
            final_html = f'<div style="color:{palette["SYSTEM"]}; font-style: italic;">{msg_s}</div>'

        # CASE 2: The Final TEXT: Output -> Label Blue, Content Green
        elif msg_s.startswith("TEXT:"):
            content = msg_s.replace("TEXT:", "").strip()
            final_html = (f'<span style="color:{palette["LABEL"]}; font-weight:bold;">TEXT: </span>'
                          f'<span style="color:{palette["TEXT"]}; font-weight:bold;">{content}</span>')

        # CASE 3: Actual Coordinates [x, y, w, h] -> Label Blue, Content White
        # Strict check: starts with [ and the first character inside is a digit
        elif msg_s.startswith("[") and len(msg_s) > 1 and msg_s[1].isdigit():
            final_html = (f'<span style="color:{palette["LABEL"]}; font-weight:bold;">COORDS: </span>'
                          f'<span style="color:#ffffff;">{msg_s}</span>')

        # CASE 4: Action Tags ([KenLM], [SYMSPELL], [TRANSLATE]) -> Colors + Arrow ➔
        elif "[" in msg_s and "]" in msg_s:
            tag_part = msg_s[msg_s.find("[")+1 : msg_s.find("]")]
            message_part = msg_s[msg_s.find("]")+1:]
            
            t_col = palette["SYSTEM"]
            if "FIX" in tag_part: t_col = palette["FIX"]
            elif "VETO" in tag_part: t_col = palette["VETO"]
            elif "SHIELD" in tag_part: t_col = palette["SHIELD"]
            elif "SYMSPELL" in tag_part: t_col = palette["SPELL"]
            elif "TRANSLATE" in tag_part: t_col = palette["TRANSLATE"] # Add this check
            elif "SANITY" in tag_part: t_col = palette["SANITY"]

            # Format the arrow logs professionally
            if "->" in message_part:
                before, after = message_part.split("->", 1)
                message_part = f'{before} <span style="color:{palette["LABEL"]}; font-weight:bold;">➔</span> <span style="color:{palette["TEXT"]}; font-weight:bold;">{after}</span>'

            final_html = (f'<div style="margin-left: 10px;">'
                        f'<span style="color:{t_col}; font-weight:bold;">[{tag_part}]</span>'
                        f'<span style="color:#ffffff;">{message_part}</span>'
                        f'</div>')

        # CASE 5: General Metadata (START:, ANIME:, etc.) -> Label Blue, Content White
        elif ":" in msg_s:
            try:
                label, data = msg_s.split(":", 1)
                final_html = (f'<span style="color:{palette["LABEL"]}; font-weight:bold;">{label}:</span>'
                              f'<span style="color:#ffffff;">{data}</span>')
            except ValueError:
                final_html = f'<span style="color:#ffffff;">{msg_s}</span>'

        # Push to UI
        self.console.appendHtml(final_html)
        self.console.moveCursor(QTextCursor.MoveOperation.End)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ControlPanel()
    window.show()
    sys.exit(app.exec())