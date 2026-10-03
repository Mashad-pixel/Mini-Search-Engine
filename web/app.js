"use strict";

(() => {
  const TOP_K = 10;
  const SUGGEST_MIN_CHARS = 2;
  const SUGGEST_DELAY_MS = 160;
  const SKELETON_COUNT = 5;

  const input = document.getElementById("query");
  const button = document.getElementById("search-btn");
  const statusEl = document.getElementById("status");
  const resultsEl = document.getElementById("results");
  const modeToggle = document.getElementById("mode-toggle");
  const facetsEl = document.getElementById("facets");
  const facetList = document.getElementById("facet-list");
  const facetClear = document.getElementById("facet-clear");
  const searchWrap = document.getElementById("search-wrap");
  const suggestionsEl = document.getElementById("suggestions");

  const selectedCategories = new Set();

  let requestId = 0;
  let controller = null;

  let suggestTimer = null;
  let suggestController = null;
  let suggestRequestId = 0;
  let suggestions = [];
  let activeIndex = -1;

  let resultsGeneration = 0;
  const docControllers = new Set();

  let hasScrolledToResults = false;

  // ---------------------------------------------------------------- //
  // Theme
  // ---------------------------------------------------------------- //
  (() => {
    const key = "mini-search-theme";
    const stored = localStorage.getItem(key);
    if (stored) document.documentElement.dataset.theme = stored;
    const btn = document.getElementById("theme-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      const current =
        document.documentElement.dataset.theme ||
        (window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
      const next = current === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      localStorage.setItem(key, next);
    });
  })();

  // ---------------------------------------------------------------- //
  // Helpers
  // ---------------------------------------------------------------- //
  function currentMode() {
    const checked = modeToggle.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : "hybrid";
  }

  function setStatus(message, kind) {
    statusEl.textContent = message;
    statusEl.className = kind ? `status ${kind}` : "status";
  }

  function setSnippet(element, html) {
    const template = document.createElement("template");
    template.innerHTML = html;

    const fragment = document.createDocumentFragment();
    for (const node of template.content.childNodes) {
      if (node.nodeType === Node.ELEMENT_NODE && node.nodeName === "MARK") {
        const mark = document.createElement("mark");
        mark.textContent = node.textContent;
        fragment.appendChild(mark);
      } else {
        fragment.appendChild(document.createTextNode(node.textContent));
      }
    }
    element.replaceChildren(fragment);
  }

  function renderSkeletons() {
    const fragment = document.createDocumentFragment();
    for (let i = 0; i < SKELETON_COUNT; i += 1) {
      const card = document.createElement("article");
      card.className = "result skeleton";
      card.setAttribute("aria-hidden", "true");

      const title = document.createElement("div");
      title.className = "skeleton-line skeleton-title";

      const line1 = document.createElement("div");
      line1.className = "skeleton-line";

      const line2 = document.createElement("div");
      line2.className = "skeleton-line short";

      card.append(title, line1, line2);
      fragment.appendChild(card);
    }
    resultsEl.replaceChildren(fragment);
  }

  function renderResults(hits) {
    const fragment = document.createDocumentFragment();
    const top = Math.max(...hits.map((h) => Number(h.score)), 1e-9);

    hits.forEach((hit) => {
      const card = document.createElement("article");
      card.className = "result";
      card.dataset.docId = hit.doc_id;

      const header = document.createElement("div");
      header.className = "result-header";

      const title = document.createElement("h2");
      title.className = "result-title";
      const link = document.createElement("a");
      link.href = `/document.html?id=${encodeURIComponent(hit.doc_id)}`;
      link.target = "_blank";
      link.rel = "noopener";
      link.textContent = hit.doc_id;
      title.appendChild(link);

      const score = document.createElement("span");
      score.className = "result-score";
      score.title = "Score, relative to the best result";
      const bar = document.createElement("span");
      bar.className = "score-bar";
      const fill = document.createElement("i");
      fill.style.width = `${Math.max(4, Math.round((Number(hit.score) / top) * 100))}%`;
      bar.appendChild(fill);
      const num = document.createElement("span");
      num.textContent = Number(hit.score).toFixed(3);
      score.append(bar, num);
      header.append(title, score);

      const snippet = document.createElement("p");
      snippet.className = "result-snippet";
      setSnippet(snippet, hit.snippet);

      const toggle = document.createElement("button");
      toggle.type = "button";
      toggle.className = "result-toggle";
      toggle.textContent = "Read full text";
      toggle.setAttribute("aria-expanded", "false");

      card.append(header, snippet, toggle);
      fragment.appendChild(card);
    });

    resultsEl.replaceChildren(fragment);
  }

  function scrollToResultsOnce() {
    if (hasScrolledToResults) return;
    hasScrolledToResults = true;
    const top = statusEl.getBoundingClientRect().top + window.scrollY - 90;
    window.scrollTo({ top, behavior: "smooth" });
  }

  // ---------------------------------------------------------------- //
  // Inline full-text expansion
  // ---------------------------------------------------------------- //
  function cancelDocumentFetches() {
    resultsGeneration += 1;
    for (const docController of docControllers) {
      docController.abort();
    }
    docControllers.clear();
  }

  async function loadDocument(card, body, docId) {
    const docController = new AbortController();
    docControllers.add(docController);
    const generation = resultsGeneration;

    body.dataset.state = "loading";
    body.classList.remove("error");
    body.textContent = "Loading…";
    body.hidden = false;
    card.classList.add("expanded");

    try {
      const response = await fetch(`/document/${encodeURIComponent(docId)}`, {
        signal: docController.signal,
      });
      if (response.status === 404) {
        throw new Error("this document is no longer available");
      }
      if (!response.ok) {
        throw new Error(`server responded with status ${response.status}`);
      }

      const data = await response.json();
      if (generation !== resultsGeneration) return;
      body.textContent = data.text;
      body.dataset.state = "loaded";
    } catch (error) {
      if (error.name === "AbortError" || generation !== resultsGeneration) return;
      body.textContent = `Could not load the document: ${error.message}.`;
      body.classList.add("error");
      body.dataset.state = "error";
    } finally {
      docControllers.delete(docController);
    }
  }

  function syncExpanded(card) {
    const btn = card.querySelector(".result-toggle");
    if (!btn) return;
    const open = card.classList.contains("expanded");
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "Hide full text" : "Read full text";
  }

  function toggleDocument(card) {
    toggleDocumentInner(card);
    syncExpanded(card);
  }

  function toggleDocumentInner(card) {
    const docId = card.dataset.docId;
    let body = card.querySelector(".result-body");

    if (!body) {
      body = document.createElement("div");
      body.className = "result-body";
      card.appendChild(body);
      loadDocument(card, body, docId);
      return;
    }

    const state = body.dataset.state;
    if (state === "loading") return;
    if (state === "error") {
      loadDocument(card, body, docId);
      return;
    }
    body.hidden = !body.hidden;
    card.classList.toggle("expanded", !body.hidden);
  }

  // ---------------------------------------------------------------- //
  // Facets
  // ---------------------------------------------------------------- //
  function buildFacet(item) {
    const label = document.createElement("label");
    label.className = "facet";

    const checkbox = document.createElement("input");
    checkbox.type = "checkbox";
    checkbox.value = item.category;

    const name = document.createElement("span");
    name.textContent = item.category;

    const count = document.createElement("span");
    count.className = "facet-count";
    count.textContent = String(item.count);

    if (selectedCategories.has(item.category)) {
      checkbox.checked = true;
      label.classList.add("selected");
    }
    label.append(checkbox, name, count);

    checkbox.addEventListener("change", () => {
      label.classList.toggle("selected", checkbox.checked);
      if (checkbox.checked) {
        selectedCategories.add(item.category);
      } else {
        selectedCategories.delete(item.category);
      }
      facetClear.hidden = selectedCategories.size === 0;
      if (input.value.trim()) runSearch();
    });

    return label;
  }

  async function loadFacets() {
    try {
      const response = await fetch("/categories");
      if (!response.ok) return;
      const categories = await response.json();
      if (!Array.isArray(categories) || categories.length === 0) return;
      facetList.replaceChildren(...categories.map(buildFacet));
      facetsEl.hidden = false;
    } catch {
      /* facets are optional */
    }
  }

  // ---------------------------------------------------------------- //
  // Autocomplete
  // ---------------------------------------------------------------- //
  function isSuggestionsOpen() {
    return !suggestionsEl.hidden && suggestions.length > 0;
  }

  function hideSuggestions() {
    clearTimeout(suggestTimer);
    suggestTimer = null;
    if (suggestController) {
      suggestController.abort();
      suggestController = null;
    }
    suggestRequestId += 1;

    suggestions = [];
    activeIndex = -1;
    suggestionsEl.replaceChildren();
    suggestionsEl.hidden = true;
    input.setAttribute("aria-expanded", "false");
    input.removeAttribute("aria-activedescendant");
  }

  function setActive(index) {
    const items = suggestionsEl.children;
    activeIndex = index;
    for (let i = 0; i < items.length; i += 1) {
      const selected = i === index;
      items[i].classList.toggle("active", selected);
      items[i].setAttribute("aria-selected", selected ? "true" : "false");
    }
    if (index >= 0 && items[index]) {
      input.setAttribute("aria-activedescendant", items[index].id);
      if (items[index].scrollIntoView) {
        items[index].scrollIntoView({ block: "nearest" });
      }
    } else {
      input.removeAttribute("aria-activedescendant");
    }
  }

  function moveActive(delta) {
    const count = suggestions.length;
    if (count === 0) return;
    if (activeIndex === -1) {
      setActive(delta > 0 ? 0 : count - 1);
    } else {
      setActive((activeIndex + delta + count) % count);
    }
  }

  function renderSuggestions(items) {
    if (items.length === 0) {
      hideSuggestions();
      return;
    }

    suggestions = items;
    activeIndex = -1;

    const fragment = document.createDocumentFragment();
    items.forEach((item, index) => {
      const option = document.createElement("div");
      option.className = "suggestion";
      option.id = `suggestion-${index}`;
      option.setAttribute("role", "option");
      option.setAttribute("aria-selected", "false");
      option.dataset.index = String(index);
      option.textContent = item.title;
      fragment.appendChild(option);
    });

    suggestionsEl.replaceChildren(fragment);
    suggestionsEl.hidden = false;
    input.setAttribute("aria-expanded", "true");
    input.removeAttribute("aria-activedescendant");
  }

  async function fetchSuggestions(prefix) {
    if (suggestController) suggestController.abort();
    suggestController = new AbortController();
    const thisController = suggestController;
    const currentId = ++suggestRequestId;

    try {
      const params = new URLSearchParams({ q: prefix });
      const response = await fetch(`/suggest?${params.toString()}`, {
        signal: thisController.signal,
      });
      if (!response.ok) throw new Error(`status ${response.status}`);

      const items = await response.json();
      if (currentId !== suggestRequestId || input.value.trim() !== prefix) return;
      renderSuggestions(Array.isArray(items) ? items : []);
    } catch (error) {
      if (error.name === "AbortError" || currentId !== suggestRequestId) return;
      hideSuggestions();
    } finally {
      if (suggestController === thisController) suggestController = null;
    }
  }

  function selectSuggestion(index) {
    const item = suggestions[index];
    if (!item) return;
    input.value = item.title.replace(/\u2026\s*$/, "").trim();
    input.focus();
    runSearch();
  }

  // ---------------------------------------------------------------- //
  // Search
  // ---------------------------------------------------------------- //
  let restoring = false;

  function syncUrl(query) {
    if (restoring) return;
    const params = new URLSearchParams({ q: query, mode: currentMode() });
    for (const category of selectedCategories) params.append("category", category);
    const next = `?${params.toString()}`;
    if (next !== window.location.search) history.pushState(null, "", next);
  }

  function applyUrl() {
    const params = new URLSearchParams(window.location.search);
    const q = (params.get("q") || "").trim();
    selectedCategories.clear();
    params.getAll("category").forEach((c) => selectedCategories.add(c));
    for (const box of facetList.querySelectorAll("input[type=checkbox]")) {
      box.checked = selectedCategories.has(box.value);
      box.parentElement.classList.toggle("selected", box.checked);
    }
    facetClear.hidden = selectedCategories.size === 0;
    for (const radio of modeToggle.querySelectorAll('input[name="mode"]')) {
      if (radio.value === params.get("mode")) radio.checked = true;
    }
    input.value = q;
    if (q) {
      restoring = true;
      runSearch();
      restoring = false;
    } else {
      cancelDocumentFetches();
      resultsEl.replaceChildren();
      setStatus("", "");
      document.body.classList.remove("searched");
    }
  }

  window.addEventListener("popstate", applyUrl);

  async function runSearch() {
    hideSuggestions();
    const query = input.value.trim();
    if (!query) {
      input.focus();
      return;
    }

    document.body.classList.add("searched");
    syncUrl(query);
    cancelDocumentFetches();

    if (controller) controller.abort();
    controller = new AbortController();
    const currentId = ++requestId;

    button.disabled = true;
    renderSkeletons();
    setStatus("Searching…", "loading");
    scrollToResultsOnce();

    const startedAt = performance.now();

    try {
      const params = new URLSearchParams({
        q: query,
        top_k: String(TOP_K),
        mode: currentMode(),
      });
      for (const category of selectedCategories) {
        params.append("category", category);
      }

      const response = await fetch(`/search?${params.toString()}`, {
        signal: controller.signal,
      });
      if (!response.ok) {
        throw new Error(`server responded with status ${response.status}`);
      }

      const hits = await response.json();
      if (currentId !== requestId) return;

      const elapsed = Math.round(performance.now() - startedAt);

      if (hits.length === 0) {
        resultsEl.replaceChildren();
        setStatus(`No results for “${query}”.`, "");
        if (currentMode() !== "semantic") {
          const retry = document.createElement("button");
          retry.type = "button";
          retry.className = "status-action";
          retry.textContent = "Search by meaning instead";
          retry.addEventListener("click", () => {
            modeToggle.querySelector('input[value="semantic"]').checked = true;
            runSearch();
          });
          statusEl.append(retry);
        }
        return;
      }

      const label = hits.length === 1 ? "result" : "results";
      setStatus(`${hits.length} ${label} · ${elapsed} ms`, "");
      renderResults(hits);
    } catch (error) {
      if (error.name === "AbortError" || currentId !== requestId) return;
      resultsEl.replaceChildren();
      setStatus(`Something went wrong: ${error.message}`, "error");
    } finally {
      if (currentId === requestId) button.disabled = false;
    }
  }

  // ---------------------------------------------------------------- //
  // Events
  // ---------------------------------------------------------------- //
  button.addEventListener("click", runSearch);

  document.getElementById("examples")?.addEventListener("click", (event) => {
    const chip = event.target.closest("button[data-query]");
    if (!chip) return;
    input.value = chip.dataset.query;
    runSearch();
  });

  input.addEventListener("keydown", (event) => {
    if (event.isComposing) return;

    switch (event.key) {
      case "Enter":
        event.preventDefault();
        if (isSuggestionsOpen() && activeIndex >= 0) {
          selectSuggestion(activeIndex);
        } else {
          runSearch();
        }
        break;
      case "ArrowDown":
        if (isSuggestionsOpen()) {
          event.preventDefault();
          moveActive(1);
        }
        break;
      case "ArrowUp":
        if (isSuggestionsOpen()) {
          event.preventDefault();
          moveActive(-1);
        }
        break;
      case "Escape":
        if (isSuggestionsOpen()) {
          event.preventDefault();
          hideSuggestions();
        } else if (input.value) {
          input.value = "";
        }
        break;
      case "Tab":
        hideSuggestions();
        break;
      default:
        break;
    }
  });

  input.addEventListener("input", () => {
    const prefix = input.value.trim();
    clearTimeout(suggestTimer);
    suggestTimer = null;

    if (prefix.length < SUGGEST_MIN_CHARS) {
      hideSuggestions();
      return;
    }
    suggestTimer = setTimeout(() => {
      suggestTimer = null;
      fetchSuggestions(prefix);
    }, SUGGEST_DELAY_MS);
  });

  // "/" focuses the search box from anywhere on the page.
  document.addEventListener("keydown", (event) => {
    if (event.key !== "/" || event.ctrlKey || event.metaKey || event.altKey) return;
    const active = document.activeElement;
    if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
    event.preventDefault();
    input.focus();
    input.select();
  });

  suggestionsEl.addEventListener("mousedown", (event) => event.preventDefault());

  suggestionsEl.addEventListener("mouseover", (event) => {
    const option = event.target.closest(".suggestion");
    if (option) setActive(Number(option.dataset.index));
  });

  suggestionsEl.addEventListener("click", (event) => {
    const option = event.target.closest(".suggestion");
    if (option) selectSuggestion(Number(option.dataset.index));
  });

  document.addEventListener("click", (event) => {
    if (!searchWrap.contains(event.target)) hideSuggestions();
  });

  resultsEl.addEventListener("click", (event) => {
    const card = event.target.closest(".result");
    if (!card || !resultsEl.contains(card)) return;
    if (event.target.closest("a") || event.target.closest(".result-body")) return;
    const selection = window.getSelection();
    if (selection && selection.toString().length > 0) return;
    toggleDocument(card);
  });

  modeToggle.addEventListener("change", () => {
    if (input.value.trim()) runSearch();
  });

  facetClear.addEventListener("click", () => {
    selectedCategories.clear();
    for (const checkbox of facetList.querySelectorAll("input[type=checkbox]")) {
      checkbox.checked = false;
      checkbox.parentElement.classList.remove("selected");
    }
    facetClear.hidden = true;
    if (input.value.trim()) runSearch();
  });

(async () => {
    const params = new URLSearchParams(window.location.search);
    const q = (params.get("q") || "").trim();
    params.getAll("category").forEach((c) => selectedCategories.add(c));
    for (const radio of modeToggle.querySelectorAll('input[name="mode"]')) {
      if (radio.value === params.get("mode")) radio.checked = true;
    }
    if (q) input.value = q;
    await loadFacets();
    facetClear.hidden = selectedCategories.size === 0;
    if (q) runSearch();
  })();
})();