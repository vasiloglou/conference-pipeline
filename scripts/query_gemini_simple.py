#!/usr/bin/env python3
"""
Simple Gemini query using existing Chrome instance via CDP (Chrome DevTools Protocol).

This connects to your already-running Chrome instead of launching a new instance,
so it uses your existing login session.
"""

import argparse
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


def query_gemini_via_cdp(
    question: str,
    gemini_url: str = "https://gemini.google.com/u/2/app?pageId=none",
    timeout: int = 60000,
    model: str | None = None,
    add_notebooklm: bool = False,
    notebook_name: str | None = None,
) -> str:
    """
    Query Gemini by connecting to existing Chrome instance.

    Args:
        question: Question to ask
        gemini_url: Gemini URL to use
        timeout: Timeout in milliseconds

    Returns:
        Response text
    """
    print("=" * 80)
    print("Connecting to your running Chrome browser...")
    print(f"Question: {question}")
    print(f"URL: {gemini_url}\n")
    print("⚠️  IMPORTANT: Make sure Chrome is already running and you're logged into Gemini!")
    print("=" * 80)
    print()

    with sync_playwright() as p:
        # Connect to existing Chrome instance via CDP
        # Chrome must be started with remote debugging enabled
        try:
            browser = p.chromium.connect_over_cdp("http://localhost:9222")
            print("✓ Connected to Chrome")
        except Exception as e:
            print("\n❌ Could not connect to Chrome!")
            print("\nYou need to start Chrome with remote debugging enabled:")
            print("\n  /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\")
            print("    --remote-debugging-port=9222 &")
            print("\nOr use the helper script: ./scripts/start_chrome_debug.sh")
            raise

        try:
            # Get existing contexts and pages
            contexts = browser.contexts
            if not contexts:
                raise Exception("No browser contexts found. Is Chrome running?")

            context = contexts[0]

            # Create new page or use existing
            if len(context.pages) > 0:
                page = context.pages[0]
                print(f"✓ Using existing tab")
            else:
                page = context.new_page()
                print(f"✓ Created new tab")

            def switch_model(target_model: str) -> bool:
                """Open the model picker and choose the desired variant."""
                target_model = target_model.lower()
                try:
                    page.wait_for_selector('[data-test-id="bard-mode-menu-button"]', timeout=10000)
                except Exception:
                    pass
                try:
                    page.evaluate(
                        """() => {
                            const app = document.querySelector('#app-root[aria-hidden="true"]');
                            if (app) app.removeAttribute('aria-hidden');
                        }"""
                    )
                except Exception:
                    pass
                # Buttons that currently expose the model menu
                def find_model_button():
                    # Try obvious buttons
                    candidates = [
                        page.locator('button[aria-label*="Model"]'),
                        page.locator('button[aria-label*="Gemini"]'),
                        page.locator("button:has-text('Pro')"),
                        page.locator("button:has-text('Flash')"),
                        page.locator("button:has-text('Gemini')"),
                    ]
                    for cand in candidates:
                        if cand.count() > 0 and cand.first.is_visible():
                            return cand.first
                    # Fallback: any element with 'Pro' text and a clickable ancestor
                    text_nodes = page.get_by_text("Pro", exact=False)
                    for i in range(min(text_nodes.count(), 5)):
                        node = text_nodes.nth(i)
                        btn_ancestor = node.locator("xpath=ancestor::button[1]")
                        if btn_ancestor.count() > 0 and btn_ancestor.first.is_visible():
                            return btn_ancestor.first
                        role_btn = node.locator("xpath=ancestor::*[@role='button'][1]")
                        if role_btn.count() > 0 and role_btn.first.is_visible():
                            return role_btn.first
                    return None

                openers = [
                    page.locator('button[aria-label*="Model"]'),
                    page.locator('button[aria-label*="Gemini"]'),
                    page.locator("button:has-text('Pro')"),
                    page.locator("button:has-text('Flash')"),
                    page.locator("button:has-text('Gemini')"),
                    page.locator('div[role="button"]').filter(has_text="Gemini"),
                    # Scoped guesses: first form/input bar container
                    page.locator("form button:has-text('Pro')"),
                    page.locator("form button:has-text('Flash')"),
                    page.locator("form button:has-text('Gemini')"),
                    # Explicit mode picker from provided DOM
                    page.locator('[aria-label="Open mode picker"] button'),
                    page.locator('[data-test-id="bard-mode-menu-button"] button'),
                    page.locator('[data-test-id="bard-mode-menu-button"]'),
                ]
                if target_model in ("fast", "flash"):
                    option_texts = [
                        "Fast",  # menu item text
                        "Fast Answers quickly",
                        "Gemini 1.5 Flash",
                        "Gemini 1.5 Flash Experimental",
                        "1.5 Flash",
                        "Flash",
                    ]
                else:
                    option_texts = [
                        "Pro",  # menu item text
                        "Pro Thinks longer for advanced math & code",
                        "Gemini 2.0 Pro",
                        "Gemini 1.5 Pro",
                        "Gemini Pro",
                        "2.0 Pro",
                        "1.5 Pro",
                    ]

                target_label = "Fast" if target_model in ("fast", "flash") else "Pro"

                def get_mode_label() -> str:
                    try:
                        return page.evaluate(
                            """() => {
                                const labelContainer = document.querySelector('[data-test-id="logo-pill-label-container"]');
                                if (labelContainer) {
                                    const span = labelContainer.querySelector('span');
                                    return (span ? span.innerText : labelContainer.innerText).trim();
                                }
                                const container = document.querySelector('[data-test-id="bard-mode-menu-button"]');
                                if (container) {
                                    return (container.innerText || '').trim();
                                }
                                return '';
                            }"""
                        )
                    except Exception:
                        return ""

                def matches_target(label: str) -> bool:
                    return bool(label) and label.strip().lower().startswith(target_label.lower())

                current_label = get_mode_label()
                if matches_target(current_label):
                    print(f"✓ Model already set to {current_label}")
                    return True

                def click_button_with_text(text: str) -> bool:
                    btn = page.locator(f"button:has-text('{text}')")
                    if btn.count() > 0 and btn.first.is_visible():
                        btn.first.click(force=True, timeout=2000)
                        page.wait_for_timeout(300)
                        return True
                    # Try nearest button ancestor of the text node
                    node = page.get_by_text(text, exact=True)
                    if node.count() > 0:
                        handle = node.first
                        button_ancestor = handle.locator("xpath=ancestor::button[1]")
                        if button_ancestor.count() > 0 and button_ancestor.first.is_visible():
                            button_ancestor.first.click(force=True, timeout=2000)
                            page.wait_for_timeout(300)
                            return True
                    return False

                def try_select_option() -> bool:
                    current_label = get_mode_label()
                    if matches_target(current_label):
                        print(f"✓ Model already set to {current_label}")
                        return True

                    # Wait briefly for menu items to appear
                    try:
                        page.locator("[role='menuitem']").first.wait_for(timeout=2000)
                    except Exception:
                        pass

                    for text in option_texts:
                        menu_opt = page.locator(f"[role='menuitem']:has-text('{text}')")
                        if menu_opt.count() > 0 and menu_opt.first.is_visible():
                            menu_opt.first.click(force=True, timeout=2000)
                            page.wait_for_timeout(500)
                            selected = get_mode_label()
                            if matches_target(selected):
                                print(f"✓ Switched model to {selected}")
                                return True
                        opt = page.get_by_text(text, exact=False)
                        if opt.count() > 0:
                            opt.first.click(force=True, timeout=2000)
                            page.wait_for_timeout(500)
                            selected = get_mode_label()
                            if matches_target(selected):
                                print(f"✓ Switched model to {selected}")
                                return True

                    # Menu is open; try to click by index for known layout
                    try:
                        items = page.locator("[role='menuitem']")
                        if items.count() > 0:
                            idx = 0 if target_model in ("fast", "flash") else 2
                            if items.count() > idx:
                                items.nth(idx).click(force=True, timeout=2000)
                                page.wait_for_timeout(500)
                                selected = get_mode_label()
                                if matches_target(selected):
                                    print(f"✓ Switched model to {selected} via menu index {idx}")
                                    return True
                    except Exception:
                        pass

                    # JS fallback: match menu item by text prefix
                    try:
                        clicked = page.evaluate(
                            """(label) => {
                                const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
                                const match = items.find(el => (el.innerText || '').trim().startsWith(label));
                                if (match) { match.click(); return true; }
                                return false;
                            }""",
                            target_label,
                        )
                        if clicked:
                            page.wait_for_timeout(500)
                            selected = get_mode_label()
                            if matches_target(selected):
                                print(f"✓ Switched model to {selected} via JS")
                                return True
                    except Exception:
                        pass

                    return False

                def js_open_and_select() -> bool:
                    try:
                        opened = page.evaluate(
                            """() => {
                                const container = document.querySelector('[data-test-id="bard-mode-menu-button"]')
                                    || document.querySelector('[aria-label="Open mode picker"]');
                                if (!container) return false;
                                const btn = container.querySelector('button') || container;
                                btn.click();
                                return true;
                            }"""
                        )
                        if not opened:
                            return False
                        page.wait_for_timeout(500)
                        clicked = page.evaluate(
                            """(label) => {
                                const items = Array.from(document.querySelectorAll('[role="menuitem"]'));
                                const match = items.find(el => (el.innerText || '').trim().startsWith(label));
                                if (match) { match.click(); return true; }
                                return false;
                            }""",
                            target_label,
                        )
                        if clicked:
                            page.wait_for_timeout(500)
                            selected = get_mode_label()
                            if matches_target(selected):
                                print(f"✓ Switched model to {selected} via JS")
                                return True
                    except Exception:
                        pass
                    return False

                extra = find_model_button()
                scoped = page.locator("form button:has-text('Pro')")
                scoped2 = page.locator("form button:has-text('Flash')")
                candidate_openers = openers + ([extra] if extra else []) + (
                    [scoped] if scoped.count() > 0 else []
                ) + ([scoped2] if scoped2.count() > 0 else [])

                for opener in candidate_openers:
                    try:
                        # Dismiss any overlays/backdrops before attempting click
                        for _ in range(3):
                            page.keyboard.press("Escape")
                            page.wait_for_timeout(100)

                        opener.click(force=True, timeout=2000)
                        page.wait_for_timeout(500)

                        # If the opener was just text, ensure dropdown is open by trying a second click on text
                        if not page.locator("text=Flash").is_visible() and not page.locator("text=Pro").is_visible():
                            click_button_with_text("Pro")
                            click_button_with_text("Flash")

                        if try_select_option():
                            return True

                        page.keyboard.press("Escape")
                    except Exception:
                        continue

                # Final JS attempt: click the mode picker container and select by label
                if js_open_and_select():
                    return True

                print(f"⚠️  Could not switch model to '{target_model}', continuing with current model.")
                try:
                    current_label = get_mode_label()
                    menu_count = page.evaluate("() => document.querySelectorAll('[role=\"menuitem\"]').length")
                    print(f"   Debug mode label: {current_label!r}; menu items: {menu_count}")
                    debug_buttons = page.locator("button").all_inner_texts()
                    print("   Debug available buttons (sample):", debug_buttons[:10])
                    page.screenshot(path="gemini_model_picker_debug.png", full_page=True)
                    print("   Saved screenshot: gemini_model_picker_debug.png")
                except Exception:
                    pass
                return False

            # Navigate to Gemini
            print(f"📍 Navigating to {gemini_url}...")
            page.goto(gemini_url, timeout=timeout, wait_until="domcontentloaded")
            time.sleep(3)

            # Check if logged in
            if "accounts.google.com" in page.url:
                raise Exception("Not logged in! Please log into Gemini in Chrome first.")

            # Optionally switch model (e.g., to Flash/fast)
            switch_success = False
            if model:
                switch_success = switch_model(model)

            # Prompt-level fallback if switch failed
            if model and not switch_success and model.lower() in ("fast", "flash"):
                question = f"Use Gemini 1.5 Flash (fast) for this answer. {question}"

            def add_notebooklm_source(name: str | None) -> None:
                try:
                    page.wait_for_selector('button[aria-label="Open upload file menu"]', timeout=10000)
                except Exception:
                    pass
                def select_notebook(target_name: str) -> bool:
                    if not target_name:
                        return False

                    def confirm_notebook_selection() -> None:
                        confirm_selectors = [
                            'button:has-text("Add")',
                            'button:has-text("Insert")',
                            'button:has-text("Select")',
                            'button:has-text("Open")',
                            'button:has-text("Use")',
                            'button:has-text("Attach")',
                            'button[aria-label*="Add"]',
                        ]
                        for selector in confirm_selectors:
                            try:
                                btn = page.locator(selector)
                                if btn.count() > 0 and btn.first.is_visible():
                                    btn.first.click(force=True, timeout=2000)
                                    break
                            except Exception:
                                continue
                        try:
                            page.wait_for_timeout(500)
                        except Exception:
                            pass
                        try:
                            page.keyboard.press("Escape")
                        except Exception:
                            pass
                        try:
                            page.wait_for_selector('[data-test-id="notebook-item-title"]', state="detached", timeout=5000)
                        except Exception:
                            pass

                    # Direct NotebookLM picker list items
                    try:
                        list_item = page.locator('.mat-mdc-list-item:has-text("'+target_name+'")')
                        if list_item.count() > 0 and list_item.first.is_visible():
                            list_item.first.click(force=True, timeout=2000)
                            confirm_notebook_selection()
                            print(f"✓ Selected NotebookLM: {target_name}")
                            return True
                        titles = page.locator('[data-test-id="notebook-item-title"]')
                        if titles.count() > 0:
                            for i in range(titles.count()):
                                txt = titles.nth(i).inner_text().strip()
                                if target_name.lower() in txt.lower():
                                    titles.nth(i).click(force=True, timeout=2000)
                                    confirm_notebook_selection()
                                    print(f"✓ Selected NotebookLM: {txt}")
                                    return True
                    except Exception:
                        pass

                    # Optional search box if present
                    search_selectors = [
                        'input[placeholder*="Search"]',
                        'input[aria-label*="Search"]',
                        'input[type="search"]',
                        'input[placeholder*="notebook"]',
                    ]
                    for selector in search_selectors:
                        try:
                            inp = page.locator(selector)
                            if inp.count() > 0 and inp.first.is_visible():
                                inp.first.click()
                                inp.first.fill(target_name)
                                time.sleep(0.2)
                                inp.first.press("Enter")
                                time.sleep(0.5)
                                break
                        except Exception:
                            continue

                    selectors = [
                        f'[role="option"]:has-text("{target_name}")',
                        f'[role="menuitem"]:has-text("{target_name}")',
                        f'[role="button"]:has-text("{target_name}")',
                        f'button:has-text("{target_name}")',
                        f'a:has-text("{target_name}")',
                    ]
                    for selector in selectors:
                        try:
                            el = page.locator(selector)
                            if el.count() > 0 and el.first.is_visible():
                                el.first.click(force=True, timeout=2000)
                                print(f"✓ Selected NotebookLM: {target_name}")
                                return True
                        except Exception:
                            continue

                    try:
                        clicked = page.evaluate(
                            """(name) => {
                                const nodes = Array.from(document.querySelectorAll('[role="option"], [role="menuitem"], [role="button"], button, a, div'));
                                const target = name.toLowerCase();
                                const match = nodes.find(el => ((el.innerText || '').trim().toLowerCase().includes(target)));
                                if (!match) return false;
                                match.click();
                                return true;
                            }""",
                            target_name,
                        )
                        if clicked:
                            confirm_notebook_selection()
                            print(f"✓ Selected NotebookLM: {target_name}")
                            return True
                    except Exception:
                        pass

                    def select_in_picker_frame() -> bool:
                        frame = None
                        for fr in page.frames:
                            if "docs.google.com/picker" in fr.url:
                                frame = fr
                                break
                        if not frame:
                            return False

                        try:
                            titles = frame.locator('[data-test-id="notebook-item-title"]')
                            if titles.count() > 0:
                                for i in range(titles.count()):
                                    txt = titles.nth(i).inner_text().strip()
                                    if target_name.lower() in txt.lower():
                                        titles.nth(i).click()
                                        print(f"✓ Selected NotebookLM in picker: {txt}")
                                        return True
                        except Exception:
                            pass

                        # Try search inside picker
                        for selector in ['input[type="search"]', 'input[type="text"]', 'input[aria-label*="Search"]']:
                            try:
                                inp = frame.locator(selector)
                                if inp.count() > 0 and inp.first.is_visible():
                                    inp.first.click()
                                    inp.first.fill(target_name)
                                    time.sleep(0.2)
                                    inp.first.press("Enter")
                                    time.sleep(0.5)
                                    break
                            except Exception:
                                continue

                        item_selectors = [
                            f'text="{target_name}"',
                            f'[role="listitem"]:has-text("{target_name}")',
                            f'[role="option"]:has-text("{target_name}")',
                            f'div:has-text("{target_name}")',
                        ]
                        for selector in item_selectors:
                            try:
                                el = frame.locator(selector)
                                if el.count() > 0 and el.first.is_visible():
                                    el.first.click()
                                    time.sleep(0.3)
                                    break
                            except Exception:
                                continue

                        # Confirm selection if button exists
                        for selector in ['button:has-text("Select")', 'button:has-text("Open")', 'button:has-text("Insert")']:
                            try:
                                btn = frame.locator(selector)
                                if btn.count() > 0 and btn.first.is_visible():
                                    btn.first.click()
                                    print(f"✓ Selected NotebookLM in picker: {target_name}")
                                    return True
                            except Exception:
                                continue

                        # If there is no confirm button, try clicking the row directly
                        try:
                            el = frame.get_by_text(target_name, exact=False)
                            if el.count() > 0:
                                el.first.click()
                                print(f"✓ Selected NotebookLM in picker: {target_name}")
                                return True
                        except Exception:
                            pass
                        return False

                    if select_in_picker_frame():
                        return True

                    try:
                        notebook_titles = page.locator('[data-test-id="notebook-item-title"]').all_inner_texts()
                        if notebook_titles:
                            print(f"⚠️  Could not select NotebookLM: {target_name}. Notebook titles: {notebook_titles[:10]}")
                        else:
                            texts = page.evaluate(
                                """() => {
                                    const nodes = Array.from(document.querySelectorAll('[role="option"], [role="menuitem"], [role="button"], button, a, div'));
                                    const out = [];
                                    for (const el of nodes) {
                                        const t = (el.innerText || '').trim();
                                        if (t && t.length < 80) out.push(t);
                                        if (out.length >= 10) break;
                                    }
                                    return out;
                                }"""
                            )
                            print(f"⚠️  Could not select NotebookLM: {target_name}. Sample items: {texts}")
                    except Exception:
                        print(f"⚠️  Could not select NotebookLM: {target_name}")
                    try:
                        page.keyboard.press("Escape")
                    except Exception:
                        pass
                    return False

                # Open the "+" menu
                open_selectors = [
                    'button[aria-label="Open upload file menu"]',
                    'button[aria-label*="Open upload file menu"]:not([aria-label*="Close"])',
                    'button.upload-card-button:not([aria-label*="Close"])',
                    'button.upload-card-button.open',
                    '[data-test-id="open-upload-menu"]',
                ]
                def is_menu_open() -> bool:
                    if page.locator('[aria-label="Close upload file menu"]').count() > 0:
                        return page.locator('[aria-label="Close upload file menu"]').is_visible()
                    if page.locator('[data-test-id="local-images-files-uploader-button"]').count() > 0:
                        return page.locator('[data-test-id="local-images-files-uploader-button"]').is_visible()
                    if page.locator('[data-test-id="uploader-drive-button"]').count() > 0:
                        return page.locator('[data-test-id="uploader-drive-button"]').is_visible()
                    return False

                opened = is_menu_open()
                if not opened:
                    for selector in open_selectors:
                        try:
                            btn = page.locator(selector)
                            for i in range(btn.count()):
                                if btn.nth(i).is_visible():
                                    btn.nth(i).click(force=True, timeout=2000)
                                    opened = True
                                    break
                            if opened:
                                break
                        except Exception:
                            continue
                if not opened:
                    try:
                        opened = page.evaluate(
                            """() => {
                                const btn = document.querySelector('button[aria-label="Open upload file menu"]')
                                    || document.querySelector('button.upload-card-button.open')
                                    || document.querySelector('.upload-card-button.open');
                                if (!btn) return false;
                                btn.click();
                                return true;
                            }"""
                        )
                    except Exception:
                        opened = False
                if opened:
                    try:
                        page.wait_for_selector('[data-test-id="local-images-files-uploader-button"], [data-test-id="uploader-drive-button"]', timeout=3000)
                    except Exception:
                        pass
                if not opened and not is_menu_open():
                    print("⚠️  Could not open upload menu for NotebookLM.")
                    return

                # Try direct NotebookLM option
                notebook_selectors = [
                    '[data-test-id="notebooks-import-button"]',
                    'button:has-text("NotebookLM")',
                ]
                # If the notebook list is already visible, select directly.
                if name and page.locator('[data-test-id="notebook-item-title"]').count() > 0:
                    if select_notebook(name):
                        return
                for selector in notebook_selectors:
                    try:
                        nb = page.locator(selector)
                        if nb.count() > 0 and nb.first.is_visible():
                            nb.first.click(force=True, timeout=2000)
                            print("✓ Opened NotebookLM picker")
                            try:
                                page.wait_for_selector('iframe[src*="docs.google.com/picker"]', timeout=5000)
                            except Exception:
                                pass
                            page.wait_for_timeout(1000)
                            if name:
                                select_notebook(name)
                            return
                    except Exception:
                        continue

                # If NotebookLM isn't shown, go through "Add from Drive"
                drive_selectors = [
                    '[data-test-id="uploader-drive-button"]',
                    'button:has-text("Add from Drive")',
                    'button[aria-label*="Add from Drive"]',
                ]
                for selector in drive_selectors:
                    drive_btn = page.locator(selector)
                    if drive_btn.count() > 0 and drive_btn.first.is_visible():
                        try:
                            drive_btn.first.click(force=True, timeout=2000)
                            page.wait_for_timeout(500)
                            for selector in notebook_selectors:
                                nb = page.locator(selector)
                                if nb.count() > 0 and nb.first.is_visible():
                                    nb.first.click(force=True, timeout=2000)
                                    print("✓ Opened NotebookLM picker")
                                    try:
                                        page.wait_for_selector('iframe[src*="docs.google.com/picker"]', timeout=5000)
                                    except Exception:
                                        pass
                                    page.wait_for_timeout(1000)
                                    if name:
                                        select_notebook(name)
                                    return
                        except Exception:
                            pass

                    try:
                        menu_items = page.locator('button[role="menuitem"], .mat-mdc-list-item')
                        sample = menu_items.all_inner_texts()[:8] if menu_items.count() > 0 else []
                        print(f"⚠️  Could not find NotebookLM option in the upload menu. Menu sample: {sample}")
                    except Exception:
                        print("⚠️  Could not find NotebookLM option in the upload menu.")

            # Find input
            print("🔍 Looking for chat input...")
            input_selectors = [
                'textarea[aria-label*="Message Gemini"]',
                'textarea[aria-label*="Message"]',
                'textarea[placeholder*="Message"]',
                'textarea[placeholder*="Ask"]',
                'div[aria-label*="Message Gemini"][contenteditable="true"]',
                '.ql-editor',
                'div[contenteditable="true"]',
            ]

            input_element = None
            for selector in input_selectors:
                try:
                    candidate = page.wait_for_selector(selector, timeout=5000)
                    if candidate and candidate.is_visible():
                        input_element = candidate
                        print(f"✓ Found input: {selector}")
                        break
                except PlaywrightTimeout:
                    continue
                except Exception:
                    continue

            if not input_element:
                raise Exception("Could not find chat input")

            if add_notebooklm or notebook_name:
                add_notebooklm_source(notebook_name)

            # Type question
            print(f"⌨️  Typing question...")
            input_element.click()
            time.sleep(0.5)
            input_element.type(question, delay=50)
            time.sleep(0.5)

            # Submit
            submit_selectors = [
                'button[aria-label*="Send message"]',
                'button[aria-label*="Send"]',
                'button[aria-label*="Reply"]',
                'button[type="submit"]',
                'button:has-text("Send")',
                'button:has-text("Reply")'
            ]

            print("🔍 Looking for send button...")
            submitted = False
            for selector in submit_selectors:
                try:
                    submit_btn = page.query_selector(selector)
                    if submit_btn and submit_btn.is_visible() and submit_btn.is_enabled():
                        print(f"✓ Found send button: {selector}")
                        try:
                            submit_btn.click(timeout=2000, force=True)
                            submitted = True
                            break
                        except Exception as e:
                            print(f"   Click failed ({selector}): {e}. Will try next selector.")
                except Exception:
                    continue

            if not submitted:
                print("⌨️  Trying Enter key...")
                page.keyboard.press("Enter")
                submitted = True

            print("⏳ Waiting for response...")
            time.sleep(2)

            def get_last_model_text():
                return page.evaluate("""
                    () => {
                        const nodes = document.querySelectorAll('[data-message-author="model"], .model-response, .response-container, markdown-element');
                        if (nodes.length === 0) return '';
                        return nodes[nodes.length - 1].innerText || '';
                    }
                """)

            # Wait for response to complete with a stabilizing poll
            print("⏳ Waiting for response to complete...")
            stable_count = 0
            last_text = ""
            first_non_empty_ts = None

            for i in range(120):
                try:
                    current_text = get_last_model_text()
                    if current_text:
                        if first_non_empty_ts is None:
                            first_non_empty_ts = time.time()
                        if current_text == last_text:
                            stable_count += 1
                        else:
                            stable_count = 0
                        # Require stability AND either length threshold or elapsed time
                        long_enough = len(current_text) >= 80
                        waited_long = first_non_empty_ts and (time.time() - first_non_empty_ts) > 15
                        if stable_count >= 5 and (long_enough or waited_long):
                            print("✓ Response complete!")
                            break
                    else:
                        stable_count = 0
                    last_text = current_text

                    if i % 5 == 0 and current_text:
                        print(f"   Response length: {len(current_text)} chars...")
                    time.sleep(1)
                except Exception:
                    time.sleep(1)

            # Small extra wait to catch late tokens
            time.sleep(3)
            final_text = get_last_model_text().strip()

            # Extract response
            print("📝 Extracting response...")
            response_text = final_text or "Error: Could not extract response"

            # Note: We don't close the browser since we're connected to existing Chrome
            return response_text

        except Exception as e:
            raise Exception(f"Query failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Query Gemini via existing Chrome (CDP connection)"
    )
    parser.add_argument("question", nargs="?", help="Question to ask")
    parser.add_argument(
        "--url",
        default="https://gemini.google.com/u/2/app?pageId=none",
        help="Gemini URL"
    )
    parser.add_argument(
        "--model",
        choices=["fast", "flash", "pro"],
        help="Optional model selector (fast/flash or pro). Default: leave current UI setting."
    )
    parser.add_argument(
        "--add-notebooklm",
        action="store_true",
        help="Open the + menu and select NotebookLM before sending the prompt."
    )
    parser.add_argument(
        "--notebook-name",
        type=str,
        help="NotebookLM notebook name to attach after opening the picker."
    )

    args = parser.parse_args()

    if not args.question:
        print("=== Query Google Gemini ===")
        args.question = input("Enter your question: ").strip()
        if not args.question:
            print("Error: Question cannot be empty")
            return 1

    try:
        response = query_gemini_via_cdp(
            question=args.question,
            gemini_url=args.url,
            model=args.model,
            add_notebooklm=args.add_notebooklm,
            notebook_name=args.notebook_name,
        )

        print("\n" + "=" * 80)
        print("GEMINI RESPONSE:")
        print("=" * 80)
        print(response)
        print("=" * 80)

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
