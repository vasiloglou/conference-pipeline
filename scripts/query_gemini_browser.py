#!/usr/bin/env python3
"""
Query Google Gemini via browser automation.

This script opens a Chrome browser, navigates to gemini.google.com,
asks a question, and returns the response.
"""

import argparse
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout


def query_gemini(
    question: str,
    headless: bool = False,
    timeout: int = 60000,
    wait_for_complete: int = 5000,
    chrome_profile_dir: str = None,
    gemini_url: str = "https://gemini.google.com",
    model: str | None = None,
    add_notebooklm: bool = False,
    notebook_name: str | None = None,
) -> str:
    """
    Query Google Gemini and return the response.

    Args:
        question: Question to ask Gemini
        headless: Run in headless mode (default: False for visibility)
        timeout: Navigation timeout in milliseconds
        wait_for_complete: Time to wait after response starts (ms)
        chrome_profile_dir: Path to Chrome user data directory (uses existing profile)
        gemini_url: Gemini URL to use (default: https://gemini.google.com)

    Returns:
        Gemini's response text

    Raises:
        Exception: If automation fails
    """
    print(f"Opening Chrome and navigating to Gemini...")
    print(f"Question: {question}")
    print(f"URL: {gemini_url}\n")

    with sync_playwright() as p:
        # If using existing profile, use launch_persistent_context
        if chrome_profile_dir:
            print(f"📂 Using Chrome profile: {chrome_profile_dir}")

            # Verify profile exists
            profile_path = Path(chrome_profile_dir)
            if not profile_path.exists():
                raise Exception(f"Chrome profile directory not found: {chrome_profile_dir}")

            context = p.chromium.launch_persistent_context(
                user_data_dir=str(profile_path),
                headless=headless,
                channel="chrome",
                args=[
                    '--disable-blink-features=AutomationControlled',
                ],
                viewport={'width': 1280, 'height': 1024}
            )

            # Get the first page or create new one
            if len(context.pages) > 0:
                page = context.pages[0]
            else:
                page = context.new_page()
        else:
            # Launch fresh browser
            browser = p.chromium.launch(
                headless=headless,
                channel="chrome",  # Use installed Chrome
                args=[
                    '--disable-blink-features=AutomationControlled',
                    '--disable-dev-shm-usage',
                    '--no-sandbox'
                ]
            )

            # Create browser context with viewport
            context = browser.new_context(
                viewport={'width': 1280, 'height': 1024},
                user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            )

            page = context.new_page()

        try:
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
                def find_model_button():
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
                    page.locator("form button:has-text('Pro')"),
                    page.locator("form button:has-text('Flash')"),
                    page.locator("form button:has-text('Gemini')"),
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

                        # If menu didn't appear, poke the text buttons directly
                        if not page.locator("text=Flash").is_visible() and not page.locator("text=Pro").is_visible():
                            click_button_with_text("Pro")
                            click_button_with_text("Flash")

                        if try_select_option():
                            return True

                        page.keyboard.press("Escape")
                    except Exception:
                        continue

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

            # Wait a bit for page to fully load
            time.sleep(3)

            # Check if login is required
            if "accounts.google.com" in page.url:
                print("\n⚠️  Login required!")
                print("Please log in to your Google account in the browser window.")
                print("The script will wait for you to complete login...")
                print("(You have 2 minutes)")

                # Wait for redirect back to Gemini after login (max 2 minutes)
                try:
                    page.wait_for_url("**/gemini.google.com**", timeout=120000)
                    print("✓ Login successful!")
                    time.sleep(2)
                except PlaywrightTimeout:
                    raise Exception("Login timeout - please try again")

            # Try multiple possible selectors for the chat input
            input_selectors = [
                # Current Gemini compose box variants
                'textarea[aria-label*="Message Gemini"]',
                'textarea[aria-label*="Message"]',
                'textarea[placeholder*="Message"]',
                'textarea[placeholder*="Ask"]',
                'div[aria-label*="Message Gemini"][contenteditable="true"]',
                '[data-test-id="chat-input"]',
                # Legacy fallbacks
                'div[contenteditable="true"]',
                '.ql-editor',
                'rich-textarea',
            ]

            print("🔍 Looking for chat input...")
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
                # Take screenshot for debugging
                screenshot_path = Path("gemini_debug.png")
                page.screenshot(path=str(screenshot_path))
                raise Exception(f"Could not find chat input. Screenshot saved to {screenshot_path}")

            # Optionally switch model (e.g., to Flash/fast)
            switch_success = False
            if model:
                target = model.lower()

                def switch_model(target_model: str) -> bool:
                    def find_model_button():
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

                    def click_button_with_text(text: str) -> bool:
                        btn = page.locator(f"button:has-text('{text}')")
                        if btn.count() > 0 and btn.first.is_visible():
                            btn.first.click(force=True, timeout=2000)
                            page.wait_for_timeout(300)
                            return True
                        node = page.get_by_text(text, exact=True)
                        if node.count() > 0:
                            handle = node.first
                            button_ancestor = handle.locator("xpath=ancestor::button[1]")
                            if button_ancestor.count() > 0 and button_ancestor.first.is_visible():
                                button_ancestor.first.click(force=True, timeout=2000)
                                page.wait_for_timeout(300)
                                return True
                        return False

                    option_labels = ["1.5 Flash", "Gemini 1.5 Flash", "Gemini Flash", "Flash"] if target_model in ("fast", "flash") else ["2.0 Pro", "Gemini 2.0 Pro", "1.5 Pro", "Gemini 1.5 Pro", "Gemini Pro", "Pro"]

                    openers = [
                        page.locator('button[aria-label*="Model"]'),
                        page.locator('button[aria-label*="Gemini"]'),
                        page.locator("button:has-text('Pro')"),
                        page.locator("button:has-text('Flash')"),
                        page.locator("button:has-text('Gemini')"),
                        page.locator('div[role="button"]').filter(has_text="Gemini"),
                        page.locator("form button:has-text('Pro')"),
                        page.locator("form button:has-text('Flash')"),
                        page.locator("form button:has-text('Gemini')"),
                    ]

                    extra = find_model_button()
                    scoped = page.locator("form button:has-text('Pro')")
                    scoped2 = page.locator("form button:has-text('Flash')")
                    candidate_openers = openers + ([extra] if extra else []) + ([scoped] if scoped.count() > 0 else []) + ([scoped2] if scoped2.count() > 0 else [])

                    for opener in candidate_openers:
                        try:
                            for _ in range(3):
                                page.keyboard.press("Escape")
                                page.wait_for_timeout(100)

                            if not opener.is_visible():
                                continue
                            opener.click(force=True, timeout=2000)
                            page.wait_for_timeout(500)

                            if not page.locator("text=Flash").is_visible() and not page.locator("text=Pro").is_visible():
                                click_button_with_text("Pro")
                                click_button_with_text("Flash")

                            for label in option_labels:
                                opt = page.get_by_text(label, exact=False)
                                if opt.count() > 0:
                                    opt.first.click(force=True, timeout=2000)
                                    page.wait_for_timeout(500)
                                    print(f"✓ Switched model to {label}")
                                    return True
                        except Exception:
                            continue
                    print(f"⚠️  Could not switch model to '{target_model}', continuing with current model.")
                    try:
                        debug_buttons = page.locator("button").all_inner_texts()
                        print("   Debug available buttons (sample):", debug_buttons[:10])
                        page.screenshot(path="gemini_model_picker_debug.png", full_page=True)
                        print("   Saved screenshot: gemini_model_picker_debug.png")
                    except Exception:
                        pass
                    return False

                switch_success = switch_model(target)

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

                        for selector in ['button:has-text("Select")', 'button:has-text("Open")', 'button:has-text("Insert")']:
                            try:
                                btn = frame.locator(selector)
                                if btn.count() > 0 and btn.first.is_visible():
                                    btn.first.click()
                                    print(f"✓ Selected NotebookLM in picker: {target_name}")
                                    return True
                            except Exception:
                                continue

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

            if add_notebooklm or notebook_name:
                add_notebooklm_source(notebook_name)

            # Type the question
            print(f"⌨️  Typing question...")
            input_element.click()
            time.sleep(0.5)
            input_element.type(question, delay=50)  # Type with slight delay
            time.sleep(0.5)

            # Try to find and click submit button
            submit_selectors = [
                'button[aria-label*="Send message"]',
                'button[aria-label*="Send"]',
                'button[aria-label*="Reply"]',
                'button[type="submit"]',
                '[data-test-id="send-button"]',
                'button:has-text("Send")',
                'button:has-text("Reply")',
                'button svg'  # Often the send button has an icon
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

            # If no button found, try Enter key
            if not submitted:
                print("⌨️  Trying Enter key...")
                page.keyboard.press("Enter")

            print("⏳ Waiting for response...")

            # Wait for response to start appearing
            time.sleep(2)

            # Try to detect when response is complete
            # Look for response container
            response_selectors = [
                '[data-message-author="model"]',
                '[data-test-id="response"]',
                '.model-response',
                '.response-container',
                'markdown-element'
            ]

            response_element = None
            for selector in response_selectors:
                try:
                    response_element = page.wait_for_selector(selector, timeout=10000)
                    if response_element:
                        print(f"✓ Found response container: {selector}")
                        break
                except:
                    continue

            def get_last_model_text():
                return page.evaluate("""
                    () => {
                        const nodes = document.querySelectorAll('[data-message-author="model"], .model-response, .response-container, markdown-element');
                        if (nodes.length === 0) return '';
                        return nodes[nodes.length - 1].innerText || '';
                    }
                """)

            # Wait for response to complete
            print("⏳ Waiting for response to complete...")
            stable_count = 0
            last_text = ""
            first_non_empty_ts = None
            max_wait = 120  # seconds

            for i in range(max_wait):
                try:
                    current_text = get_last_model_text()
                    if current_text:
                        if first_non_empty_ts is None:
                            first_non_empty_ts = time.time()
                        if current_text == last_text:
                            stable_count += 1
                        else:
                            stable_count = 0

                        long_enough = len(current_text) >= 80
                        waited_long = first_non_empty_ts and (time.time() - first_non_empty_ts) > 15
                        if stable_count >= 5 and (long_enough or waited_long):
                            print("✓ Response complete!")
                            break
                    else:
                        stable_count = 0

                    last_text = current_text
                    time.sleep(1)

                    if i % 5 == 0 and current_text:
                        print(f"   Response length: {len(current_text)} chars...")

                except Exception as e:
                    print(f"   Waiting... ({i}/{max_wait})")
                    time.sleep(1)

            # Small extra wait to catch late tokens
            time.sleep(3)

            # Extract final response text
            print("📝 Extracting response...")
            response_text = get_last_model_text().strip()

            if not response_text:
                # Last resort: take screenshot and dump all text
                screenshot_path = Path("gemini_response.png")
                page.screenshot(path=str(screenshot_path))
                all_text = page.evaluate("() => document.body.innerText")
                raise Exception(f"Could not extract response properly. Screenshot: {screenshot_path}\n\nPage text:\n{all_text[:500]}...")

            return response_text

        finally:
            # Keep browser open for a moment so user can see result
            if not headless:
                print("\n✓ Done! Closing browser in 3 seconds...")
                time.sleep(3)

            # Close browser/context depending on mode
            if chrome_profile_dir:
                context.close()
            else:
                browser.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Query Google Gemini via browser automation")
    parser.add_argument(
        "question",
        nargs="?",
        help="Question to ask Gemini"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (no visible browser)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60000,
        help="Navigation timeout in milliseconds (default: 60000)"
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Chrome profile directory to use (for existing login). Default: ~/Library/Application Support/Google/Chrome/Default"
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://gemini.google.com",
        help="Gemini URL (default: https://gemini.google.com). Use https://gemini.google.com/u/2/app for specific account"
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
    parser.add_argument(
        "--auto-profile",
        action="store_true",
        help="Automatically use default Chrome profile (~/Library/Application Support/Google/Chrome/Default)"
    )

    args = parser.parse_args()

    if not args.question:
        # Interactive mode
        print("=== Query Google Gemini ===")
        args.question = input("Enter your question: ").strip()
        if not args.question:
            print("Error: Question cannot be empty")
            return 1

    # Determine profile directory
    profile_dir = args.profile
    if args.auto_profile and not profile_dir:
        # Use default Chrome profile on macOS
        default_profile = Path.home() / "Library/Application Support/Google/Chrome/Default"
        if default_profile.exists():
            profile_dir = str(default_profile)
            print(f"Using default Chrome profile: {profile_dir}")
        else:
            print(f"Warning: Default profile not found at {default_profile}")

    try:
        response = query_gemini(
            question=args.question,
            headless=args.headless,
            timeout=args.timeout,
            chrome_profile_dir=profile_dir,
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
