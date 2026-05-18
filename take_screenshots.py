"""
Capture dashboard panel screenshots for the README.
Requires: pip install playwright && python -m playwright install chromium
"""
from pathlib import Path
from playwright.sync_api import sync_playwright
import time

DASHBOARD = "http://localhost:7432/dashboard.html"
OUT_DIR   = Path("docs/screenshots")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PANELS = [
    ("overview",   "◉Overview"),
    ("models",     "⬡Model Performance"),
    ("mofa",       "⬢MOFA+ Factors"),
    ("shap",       "◈SHAP Attribution"),
    ("targets",    "⬥Therapeutic Targets"),
    ("findings",   "★Findings & Implications"),
    ("did",        "⚖Causal DiD Layer"),
    ("ablation",   "◍Modality Ablation"),
]

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1440, "height": 860})
    page.goto(DASHBOARD, wait_until="networkidle")
    time.sleep(1)

    for slug, label in PANELS:
        # Click the nav item whose text contains the label (strip the icon)
        label_text = label[1:].strip()  # strip icon char
        page.evaluate(f"""
            Array.from(document.querySelectorAll('.nav-item')).find(
                el => el.textContent.includes('{label_text}')
            )?.click();
        """)
        time.sleep(0.6)

        out = OUT_DIR / f"panel_{slug}.png"
        page.screenshot(path=str(out), full_page=False)
        print(f"  Saved: {out}")

    browser.close()

print("All screenshots saved to docs/screenshots/")
