# 🎌 Anime Subtitle Translator (EN → TH)

A real-time screen OCR tool that captures English anime subtitles and translates them to Thai — displayed as a transparent overlay on top of your video player.

---

## ⬇️ Download

**[→ Download the latest release](https://github.com/ntw-cts/anime-translator/releases/latest)**

No Python installation required. Just download, extract, and run `AnimeTranslator.exe`.

---

## 🚀 First-Time Setup

1. Download and extract the `.zip` from the [Releases](https://github.com/ntw-cts/anime-translator/releases/latest) page
2. Run `AnimeTranslator.exe`
3. If Windows shows a SmartScreen warning → click **"More info"** → **"Run anyway"**
4. On first launch, the app will automatically download `english.bin` (~931 MB) for OCR context scoring — click **OK** to allow it, or **Cancel** to skip (app still works without it)
5. EasyOCR will also download its models on first run — this is automatic

> 💡 All downloads only happen **once**. After that the app launches instantly.

---

## ✨ Features

- **Real-time OCR** — Captures and reads on-screen subtitles automatically using EasyOCR
- **Multiple translation engines** — Choose based on your needs (speed, quality, or offline)
- **Transparent overlay** — Thai translation appears directly over your screen, no window switching
- **Character Entity Shield** — Fetches character names from AniList to prevent names from being mistranslated
- **Smart caching** — Translations are saved locally so repeated subtitles load instantly
- **LaBSE tie-breaking** — When two engines produce different results, semantic similarity picks the better one

---

## 🔄 Translation Engines

| Engine | Speed | Requires |
|---|---|---|
| **Auto** ⭐ | Smart | Nothing — recommended for most users |
| **Google Translate** | Fast | Nothing |
| **Gemini** | Balanced | Free Gemini API key |
| **Typhoon 1.5** | Accurate | Local [Ollama](https://ollama.com) + Typhoon model |
| **NLLB-200** | Offline | Python + ~1-2 GB download on first use |

**Recommended: leave it on Auto** — it picks the best available engine automatically.

---

## 🔑 Getting a Free Gemini API Key (Optional)

Improves **Auto** mode quality and enables the **Gemini** engine:

1. Go to [aistudio.google.com](https://aistudio.google.com)
2. Sign in and generate a free API key
3. Paste it into the **Google Gemini Configuration** box in the app
4. Click **Verify Key** — you should see ✔ Valid Key

---

## ⚙️ Settings

**Basic tab:**
- **Translation Engine** — pick your engine or leave on Auto
- **Google Gemini Configuration** — API key input (optional in Auto, required for Gemini engine)
- **Character Entity Shield** — type an anime name and press Enter or click Fetch to protect character names from mistranslation
- **Detection Zone Height** — how much of the bottom screen to scan for subtitles

**Advanced tab:**
- OCR threshold, similarity sensitivity, detection mode (Fixed / Adaptive), and more

---

## ⚠️ NLLB-200 Offline Mode

NLLB-200 requires additional packages (~1-2 GB) that are **not** bundled in the app. When you first select it, the app will ask to install them automatically via pip.

**This requires Python to be installed on your machine.** Download it from [python.org](https://python.org) if needed.

All other engines (Auto, Google Translate, Gemini) work without Python.

---

## 🦬 Typhoon 1.5 Setup

Typhoon runs locally via Ollama — no API key needed, but requires initial setup:

1. Download and install [Ollama](https://ollama.com)
2. Open a terminal and run:
   ```bash
   ollama pull scb10x/typhoon-translate1.5-4b
   ```
3. Make sure Ollama is running before starting the app
4. Select **Typhoon 1.5 (Accurate)** in the Translation Engine dropdown

> 💡 Typhoon runs fully on your machine — no internet needed after the initial model download.

---

## 📁 File Structure

After first run your folder will look like this:

```
AnimeTranslator.exe
english.bin               ← auto-downloaded on first run (~931 MB)
translations_cache.json   ← auto-created, stores past translations
```

---

## 🛠 System Requirements

- Windows 10 / 11
- Internet connection (for first-time downloads and online translation engines)
- GPU recommended but not required (used by EasyOCR and NLLB-200)

---

## 🧑‍💻 Build from Source

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
python -m venv venv
venv\Scripts\activate
pip install -r requirements_clean.txt
python stable.py
```

---

## 📄 License

MIT License — free to use, modify, and distribute.
