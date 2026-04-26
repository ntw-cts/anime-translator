# 🎌 Anime Subtitle Translator (EN → TH)

A real-time screen OCR tool that captures English anime subtitles and translates them to Thai — displayed as a transparent overlay on top of your video player.

---

## 🚀 Quick Start

### 1 — Clone the repo

```bash
git clone https://github.com/ntw-cts/anime-translator.git
cd anime-translator
```

### 2 — Install Python 3.11

> ⚠️ **Python 3.12 and newer are NOT supported.** The `kenlm` dependency does not build on Python 3.12+.
> Download Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/).

Create a virtual environment using Python 3.11:

```bash
py -3.11 -m venv venv
```

Then activate it:

```bash
venv\Scripts\activate
```

> ⚠️ If you see a script execution error in PowerShell, run this once to fix it, then activate again:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

### 3 — Install kenlm

`kenlm` has no PyPI wheel so it must be installed separately first. This requires [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) to be installed.

```bash
pip install https://github.com/kpu/kenlm/archive/master.zip
```

### 4 — Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 5 — Download `english.bin` (required for OCR accuracy)

This file is too large to include in the repo. Download it and place it in the **root of the project folder** (same folder as `stable.py`):

**[→ Download english.bin (~931 MB) from Google Drive](https://drive.google.com/file/d/1nXKi0XKLBoXbJbADaiXhGrZB17Ue11qx/view?usp=sharing)**

Your folder should look like this:

```
anime-translator/
├── stable.py
├── requirements.txt
├── english.bin        ← place here
└── assets/
```

### 6 — Run the app

```bash
python stable.py
```

---

## ✨ Features

- **Real-time OCR** — Captures and reads on-screen subtitles automatically using EasyOCR
- **Multiple translation engines** — Choose based on your needs (speed, quality, or offline)
- **Transparent overlay** — Thai translation appears directly over your screen, no window switching
- **Character Entity Shield** — Fetches character names from AniList to prevent names from being mistranslated
- **Smart caching** — Translations are saved locally so repeated subtitles load instantly
- **KenLM context scoring** — Picks the most natural OCR reading using a language model
- **LaBSE tie-breaking** — When two engines produce different results, semantic similarity picks the better one
- **SymSpell correction** — Auto-fixes OCR typos before translation

---

## 🔄 Translation Engines

| Engine | Speed | Quality | Requires |
|---|---|---|---|
| **Auto** ⭐ | Smart | Best available | Nothing — recommended for most users |
| **Google Translate** | Fast | Good | Nothing |
| **Gemini** | Balanced | High | Free Gemini API key |
| **NLLB-200** | Medium | Good | ~1.2 GB, downloads on first use |
| **Typhoon 1.5** | Accurate | Highest | Manual Ollama setup (see below) |

**Recommended: leave it on Auto** — it uses NLLB-200 as the fast pass, then refines with Typhoon or Gemini if available, and falls back to Google Translate if needed.

---

## 🦬 Typhoon 1.5 Setup (Optional)

1. Download and install [Ollama](https://ollama.com)
2. Open a terminal and run:
   ```bash
   ollama pull scb10x/typhoon-translate1.5-4b
   ```
3. Make sure Ollama is running before starting the app
4. Select **Typhoon 1.5** in the Translation Engine dropdown

> 💡 Typhoon runs fully on your machine — no internet needed after the initial model download (~2.4 GB).

---

## 🔑 Getting a Free Gemini API Key (Optional)

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

## 🛠 System Requirements

- Windows 10 / 11
- **Python 3.11** (3.12+ not supported)
- [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) (required for kenlm)
- ~2 GB free disk space
- Internet connection (required for online translation engines)
- GPU recommended but not required (used by EasyOCR and NLLB-200 if available)

---

## 📄 License

MIT License — free to use, modify, and distribute.
