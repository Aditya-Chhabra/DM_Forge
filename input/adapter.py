import asyncio
import os
import re
import time
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class InputAdapter:
    def __init__(self) -> None:
        self._url_cache: dict[str, str] = {}
        self._blocked_host_until: dict[str, float] = {}

    def resolve_post(self, raw_input: str) -> str:
        value = (raw_input or "").strip()
        if not value:
            raise ValueError("Please provide a LinkedIn post or profile URL.")
        if self._is_url(value):
            return self._fetch_latest_public_post(value)
        return self._clean_text(value)

    def extract_addressee_name(self, raw_input: str) -> str:
        value = (raw_input or "").strip()
        if not self._is_url(value):
            return ""
        parsed = urlparse(value)
        path_parts = [part for part in (parsed.path or "").split("/") if part]
        if not path_parts:
            return ""
        candidate = ""
        first = path_parts[0].lower()
        if first in {"in", "company", "school"} and len(path_parts) > 1:
            candidate = path_parts[1]
        elif first == "posts" and len(path_parts) > 1:
            candidate = path_parts[1].split("_", 1)[0]
        elif len(path_parts) > 1:
            candidate = path_parts[1]
        candidate = re.sub(r"[-_]+", " ", candidate).strip()
        candidate = re.sub(r"\d+", "", candidate).strip()
        candidate = re.sub(r"[^A-Za-z\s]", " ", candidate).strip()
        candidate = re.sub(r"\s+", " ", candidate).strip()
        if len(candidate) < 2:
            return ""
        words = [word.capitalize() for word in candidate.split() if len(word) > 1]
        if not words:
            return ""
        return " ".join(words[:3])

    def _is_url(self, value: str) -> bool:
        parsed = urlparse(value)
        return bool(parsed.scheme and parsed.netloc and "linkedin.com" in parsed.netloc.lower())

    def _fetch_latest_public_post(self, profile_url: str) -> str:
        normalized_url = self._normalize_url(profile_url)
        cached = self._url_cache.get(normalized_url)
        if cached:
            return cached
        candidates: list[str] = []
        if self._should_try_crawl4ai(normalized_url):
            crawl4ai_text = self._try_crawl4ai(profile_url)
            if crawl4ai_text:
                candidates.append(crawl4ai_text)

        urls_to_try = [profile_url]
        if normalized_url != profile_url:
            urls_to_try.append(normalized_url)

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        for url in urls_to_try:
            response = self._fetch_url_html(url, headers=headers)
            if response is None:
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            meta_candidate = self._extract_meta_candidate(soup)
            if meta_candidate:
                candidates.append(meta_candidate)
            json_ld_candidate = self._extract_json_ld_candidate(soup)
            if json_ld_candidate:
                candidates.append(json_ld_candidate)
            candidates.extend(self._extract_candidate_blocks(soup))

        best = self._select_best_candidate(candidates)
        if not best:
            fallback = self._extract_text_from_linkedin_url(profile_url)
            if fallback:
                self._url_cache[normalized_url] = fallback
                return fallback
            raise ValueError(
                "Could not extract post content from this LinkedIn URL due to access limits. "
                "Please paste the post text directly."
            )
        cleaned = self._clean_text(best)
        self._url_cache[normalized_url] = cleaned
        return cleaned

    def _should_try_crawl4ai(self, url: str) -> bool:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        use_for_linkedin = os.getenv("DM_FORGE_USE_CRAWL4AI_FOR_LINKEDIN", "0") == "1"
        if "linkedin.com" in host and not use_for_linkedin:
            return False
        if self._is_host_temporarily_blocked(host):
            return False
        return True

    def _try_crawl4ai(self, url: str) -> str | None:
        try:
            from crawl4ai import AsyncWebCrawler
        except Exception:
            return None

        async def crawl() -> str | None:
            crawler = AsyncWebCrawler()
            result = await crawler.arun(url=url)
            markdown = getattr(result, "markdown", None)
            markdown_v2 = getattr(result, "markdown_v2", None)
            raw_markdown = getattr(markdown_v2, "raw_markdown", None) if markdown_v2 else None
            content = raw_markdown or markdown
            return content if isinstance(content, str) and content.strip() else None

        try:
            return asyncio.run(crawl())
        except Exception:
            return None

    def _clean_text(self, text: str) -> str:
        compact = re.sub(r"\s+", " ", text).strip()
        if len(compact) < 20:
            raise ValueError("Input is too short. Please provide a fuller post.")
        return compact

    def _extract_candidate_blocks(self, soup: BeautifulSoup) -> list[str]:
        selectors = (
            "article",
            "div.feed-shared-update-v2",
            "div.update-components-text",
            "div[data-urn]",
            "main",
            "section",
            "p",
        )
        blocks: list[str] = []
        for selector in selectors:
            for node in soup.select(selector):
                text = node.get_text(" ", strip=True)
                if len(text.split()) >= 18:
                    blocks.append(text)
        return blocks

    def _select_best_candidate(self, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        scored = [(self._score_candidate(text), text) for text in candidates]
        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_text = scored[0]
        if best_score < 1.5:
            return None
        return best_text

    def _score_candidate(self, text: str) -> float:
        lowered = text.lower()
        words = len(lowered.split())
        if words < 18:
            return 0.0
        post_signals = (
            "hiring",
            "launched",
            "learned",
            "building",
            "project",
            "role",
            "internship",
            "team",
            "experience",
            "customers",
            "shipped",
        )
        noise_signals = (
            "sign in",
            "join now",
            "cookie",
            "privacy policy",
            "terms",
            "linkedin corporation",
            "followers",
            "connections",
            "see more",
        )
        signal_score = sum(1 for token in post_signals if token in lowered)
        noise_penalty = sum(1 for token in noise_signals if token in lowered)
        length_score = min(words / 60.0, 2.0)
        return signal_score + length_score - (noise_penalty * 0.8)

    def _extract_text_from_linkedin_url(self, profile_url: str) -> str | None:
        parsed = urlparse(profile_url)
        path_parts = [part for part in (parsed.path or "").split("/") if part]
        if len(path_parts) < 2:
            return None
        if path_parts[0] not in {"posts", "feed", "pulse"}:
            return None
        slug = path_parts[-1]
        if not slug:
            return None
        lowered_slug = slug.lower()
        topic_map: list[tuple[tuple[str, ...], str]] = [
            (("hiring", "job", "role", "intern", "opening", "vacancy", "shortlist"), "hiring and candidate selection"),
            (("raise", "fundraise", "investor", "seed", "series", "funding"), "fundraising and investor outreach"),
            (("launch", "released", "shipped", "product"), "product launch and execution"),
            (("learned", "lesson", "journey", "experience"), "career lessons and practical experience"),
            (("agency", "clients", "growth", "bd", "pipeline"), "business development and client growth"),
        ]
        detected_topics: list[str] = []
        for keywords, label in topic_map:
            if any(keyword in lowered_slug for keyword in keywords):
                detected_topics.append(label)
        if detected_topics:
            unique_topics = ", ".join(dict.fromkeys(detected_topics))
            inferred = f"LinkedIn post appears to discuss {unique_topics}."
            return inferred if len(inferred) >= 20 else None
        readable_tokens: list[str] = []
        for token in re.findall(r"[a-zA-Z]{3,20}", lowered_slug):
            if token in {"share", "activity", "posts", "post", "linkedin", "member", "desktop"}:
                continue
            if token not in readable_tokens:
                readable_tokens.append(token)
            if len(readable_tokens) >= 4:
                break
        if not readable_tokens:
            return None
        compact = " ".join(readable_tokens)
        inferred = f"LinkedIn post likely mentions {compact}."
        return inferred if len(inferred) >= 20 else None

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.scheme and parsed.netloc:
            return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        return url

    def _fetch_url_html(self, url: str, headers: dict[str, str]) -> requests.Response | None:
        parsed = urlparse(url)
        host = (parsed.netloc or "").lower()
        if self._is_host_temporarily_blocked(host):
            return None
        backoffs = [0.0, 0.6, 1.5]
        for idx, delay in enumerate(backoffs):
            if delay > 0:
                time.sleep(delay)
            try:
                response = requests.get(url, headers=headers, timeout=20)
            except requests.RequestException:
                continue
            if response.status_code == 429:
                if idx == len(backoffs) - 1:
                    self._mark_host_blocked(host, seconds=120)
                continue
            if response.status_code >= 400:
                continue
            return response
        return None

    def _extract_meta_candidate(self, soup: BeautifulSoup) -> str | None:
        selectors = [
            ('meta[property="og:description"]', "content"),
            ('meta[name="description"]', "content"),
            ('meta[property="og:title"]', "content"),
        ]
        values: list[str] = []
        for selector, attr in selectors:
            node = soup.select_one(selector)
            value = node.get(attr) if node else None
            if value and isinstance(value, str):
                values.append(value.strip())
        merged = " ".join(values).strip()
        return merged if len(merged.split()) >= 8 else None

    def _extract_json_ld_candidate(self, soup: BeautifulSoup) -> str | None:
        scripts = soup.select('script[type="application/ld+json"]')
        chunks: list[str] = []
        for script in scripts:
            text = script.get_text(" ", strip=True)
            if len(text.split()) < 10:
                continue
            lowered = text.lower()
            if "articlebody" in lowered or "headline" in lowered or "description" in lowered:
                chunks.append(text)
        merged = " ".join(chunks).strip()
        return merged if len(merged.split()) >= 12 else None

    def _is_host_temporarily_blocked(self, host: str) -> bool:
        if not host:
            return False
        blocked_until = self._blocked_host_until.get(host, 0.0)
        return blocked_until > time.monotonic()

    def _mark_host_blocked(self, host: str, seconds: float) -> None:
        if not host:
            return
        now = time.monotonic()
        self._blocked_host_until[host] = now + max(seconds, 0.0)
