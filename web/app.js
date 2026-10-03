"use strict";

(() => {
  const TOP_K = 10;
  const SUGGEST_MIN_CHARS = 2;
  const SUGGEST_DELAY_MS = 300;

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

  // Full-text fetches for expanded result cards. `resultsGeneration` changes
  // whenever the result list is replaced, so late responses are ignored.
  let resultsGeneration = 0;
  const docControllers = new Set();

  function currentMode() {
    const checked = modeToggle.querySelector('input[name="mode"]:checked');
    return checked ? checked.value : "hybrid";
  }

  function setStatus(message, kind) {
    statusEl.textContent = message;
    statusEl.className = kind ? `status ${kind}` : "status";
  }

  // The API returns HTML-escaped text where <mark> is the only markup.
  // Rebuild it node by node so nothing else can ever be injected.
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

  function renderResults(hits) {
    const fragment = document.createDocumentFragment();

    for (const hit of hits) {
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
      score.textContent = `score ${Number(hit.score).toFixed(3)}`;

      header.append(title, score);

      const snippet = document.createElement("p");
      snippet.className = "result-snippet";
      setSnippet(snippet, hit.snippet);

      card.append(header, snippet);
      fragment.appendChild(card);
    }

    resultsEl.replaceChildren(fragment);
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
    body.textContent = "Loading\u2026";
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
      if (generation !== resultsGeneration) {
        return;
      }
      body.textContent = data.text;
      body.dataset.state = "loaded";
    } catch (error) {
      if (error.name === "AbortError" || generation !== resultsGeneration) {
        return;
      }
      body.textContent = `Could not load the document: ${error.message}.`;
      body.classList.add("error");
      body.dataset.state = "error";
    } finally {
      docControllers.delete(docController);
    }
  }

  function toggleDocument(card) {
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
    if (state === "loading") {
      return;
    }
    if (state === "error") {
      loadDocument(card, body, docId);
      return;
    }
    body.hidden = !body.hidden;
    card.classList.toggle("expanded", !body.hidden);
  }

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

    label.append(checkbox, name, count);

    checkbox.addEventListener("change", () => {
      label.classList.toggle("selected", checkbox.checked);
      if (checkbox.checked) {
        selectedCategories.add(item.category);
      } else {
        selectedCategories.delete(item.category);
      }
      facetClear.hidden = selectedCategories.size === 0;
      if (input.value.trim()) {
        runSearch();
      }
    });

    return label;
  }

  async function loadFacets() {
    try {
      const response = await fetch("/categories");
      if (!response.ok) {
        return;
      }
      const categories = await response.json();
      if (!Array.isArray(categories) || categories.length === 0) {
        return;
      }
      facetList.replaceChildren(...categories.map(buildFacet));
      facetsEl.hidden = false;
    } catch (error) {
      // Facets are optional; search still works without them.
    }
  }

  // ---------------------------------------------------------------- //
  // Autocomplete
  // ---------------------------------------------------------------- //
  function isSuggestionsOpen() {
    return !suggestionsEl.hidden && suggestions.length > 0;
  }

  // Closes the dropdown and cancels any pending or in-flight suggestion
  // request so a late response cannot reopen it.
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
    if (count === 0) {
      return;
    }
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
    if (suggestController) {
      suggestController.abort();
    }
    suggestController = new AbortController();
    const thisController = suggestController;
    const currentId = ++suggestRequestId;

    try {
      const params = new URLSearchParams({ q: prefix });
      const response = await fetch(`/suggest?${params.toString()}`, {
        signal: thisController.signal,
      });
      if (!response.ok) {
        throw new Error(`Server responded with status ${response.status}`);
      }

      const items = await response.json();
      if (currentId !== suggestRequestId || input.value.trim() !== prefix) {
        return;
      }
      renderSuggestions(Array.isArray(items) ? items : []);
    } catch (error) {
      if (error.name === "AbortError" || currentId !== suggestRequestId) {
        return;
      }
      // Suggestions are optional; fail quietly and keep search usable.
      hideSuggestions();
    } finally {
      if (suggestController === thisController) {
        suggestController = null;
      }
    }
  }

  function selectSuggestion(index) {
    const item = suggestions[index];
    if (!item) {
      return;
    }
    input.value = item.title.replace(/\u2026\s*$/, "").trim();
    input.focus();
    runSearch(); // also closes the dropdown
  }

  async function runSearch() {
    hideSuggestions();
    const query = input.value.trim();
    if (!query) {
      input.focus();
      return;
    }

    document.body.classList.add("searched");
    cancelDocumentFetches();

    if (controller) {
      controller.abort();
    }
    controller = new AbortController();
    const currentId = ++requestId;

    button.disabled = true;
    resultsEl.replaceChildren();
    setStatus("Searching\u2026", "loading");

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
        throw new Error(`Server responded with status ${response.status}`);
      }

      const hits = await response.json();
      if (currentId !== requestId) {
        return;
      }

      if (hits.length === 0) {
        setStatus(`No results found for \u201c${query}\u201d.`, "");
        return;
      }

      setStatus(`${hits.length} result${hits.length === 1 ? "" : "s"}`, "");
      renderResults(hits);
    } catch (error) {
      if (error.name === "AbortError" || currentId !== requestId) {
        return;
      }
      setStatus(`Something went wrong: ${error.message}`, "error");
    } finally {
      if (currentId === requestId) {
        button.disabled = false;
      }
    }
  }

  button.addEventListener("click", runSearch);

  input.addEventListener("keydown", (event) => {
    if (event.isComposing) {
      return;
    }

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

  // Keep focus in the input while clicking inside the dropdown.
  suggestionsEl.addEventListener("mousedown", (event) => {
    event.preventDefault();
  });

  suggestionsEl.addEventListener("mouseover", (event) => {
    const option = event.target.closest(".suggestion");
    if (option) {
      setActive(Number(option.dataset.index));
    }
  });

  suggestionsEl.addEventListener("click", (event) => {
    const option = event.target.closest(".suggestion");
    if (option) {
      selectSuggestion(Number(option.dataset.index));
    }
  });

  document.addEventListener("click", (event) => {
    if (!searchWrap.contains(event.target)) {
      hideSuggestions();
    }
  });

  // Click a card (not its link, not the open text) to expand or collapse it.
  resultsEl.addEventListener("click", (event) => {
    const card = event.target.closest(".result");
    if (!card || !resultsEl.contains(card)) {
      return;
    }
    if (event.target.closest("a") || event.target.closest(".result-body")) {
      return;
    }
    const selection = window.getSelection();
    if (selection && selection.toString().length > 0) {
      return; // the user is selecting text, not toggling
    }
    toggleDocument(card);
  });

  modeToggle.addEventListener("change", () => {
    if (input.value.trim()) {
      runSearch();
    }
  });

  facetClear.addEventListener("click", () => {
    selectedCategories.clear();
    for (const checkbox of facetList.querySelectorAll("input[type=checkbox]")) {
      checkbox.checked = false;
      checkbox.parentElement.classList.remove("selected");
    }
    facetClear.hidden = true;
    if (input.value.trim()) {
      runSearch();
    }
  });

  loadFacets();
})();