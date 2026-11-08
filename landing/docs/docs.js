(() => {
  const store = {
    get(key) { try { return localStorage.getItem(key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem(key, value); } catch {} },
  };
  const root = document.documentElement;

  const copyOnClick = (btn, pre) => {
    const label = btn.textContent;
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "Copied";
      } catch {
        btn.textContent = "Select and copy";
      }
      btn.classList.add("done");
      setTimeout(() => { btn.textContent = label; btn.classList.remove("done"); }, 1400);
    });
  };
  for (const btn of document.querySelectorAll(".copy")) copyOnClick(btn, btn.closest(".code").querySelector("pre"));
  for (const btn of document.querySelectorAll(".copy-prompt")) {
    const pre = document.getElementById(btn.dataset.target)?.querySelector("pre");
    if (pre) copyOnClick(btn, pre);
  }
  for (const panel of document.querySelectorAll(".tab-panel")) {
    const btn = panel.querySelector(".copy");
    if (btn) panel.append(btn);
  }

  const selectTab = (label) => {
    for (const group of document.querySelectorAll(".tabs")) {
      if (!group.querySelector(`.tab-panel[data-tab="${label}"]`)) continue;
      for (const b of group.querySelectorAll("[role=tab]")) b.setAttribute("aria-selected", String(b.dataset.tab === label));
      for (const p of group.querySelectorAll(".tab-panel")) p.hidden = p.dataset.tab !== label;
    }
  };
  for (const b of document.querySelectorAll(".tab-list [role=tab]")) {
    b.addEventListener("click", () => {
      const top = b.getBoundingClientRect().top;
      selectTab(b.dataset.tab);
      store.set("cascade-docs-tab", b.dataset.tab);
      window.scrollBy(0, b.getBoundingClientRect().top - top);
    });
  }
  const savedTab = store.get("cascade-docs-tab");
  if (savedTab) selectTab(savedTab);

  document.querySelector(".theme-btn")?.addEventListener("click", () => {
    const dark = root.dataset.theme
      ? root.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store.set("cascade-docs-theme", root.dataset.theme);
  });

  const sidebar = document.querySelector(".sidebar");
  const scrim = document.querySelector(".scrim");
  const menuBtn = document.querySelector(".menu-btn");
  const setMenu = (open) => {
    sidebar.classList.toggle("open", open);
    scrim.hidden = !open;
    menuBtn.setAttribute("aria-expanded", String(open));
  };
  menuBtn?.addEventListener("click", () => setMenu(!sidebar.classList.contains("open")));
  scrim?.addEventListener("click", () => setMenu(false));

  const tocLinks = [...document.querySelectorAll(".toc a")];
  if (tocLinks.length && "IntersectionObserver" in window) {
    const byId = new Map(tocLinks.map((a) => [decodeURIComponent(a.hash.slice(1)), a]));
    const targets = [...byId.keys()].map((id) => document.getElementById(id)).filter(Boolean);
    const visible = new Set();
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) e.isIntersecting ? visible.add(e.target) : visible.delete(e.target);
      const first = targets.find((t) => visible.has(t));
      if (!first) return;
      for (const a of tocLinks) a.classList.toggle("active", a === byId.get(first.id));
    }, { rootMargin: "-64px 0px -65% 0px" });
    targets.forEach((t) => io.observe(t));
  }

  const dialog = document.querySelector("dialog.search");
  const input = dialog.querySelector("input");
  const list = dialog.querySelector(".results");
  let active = 0;
  const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const mark = (text, terms) => {
    let out = escapeHtml(text);
    for (const t of terms) {
      if (!t) continue;
      out = out.replace(new RegExp(`(${t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "ig"), "<mark>$1</mark>");
    }
    return out;
  };
  const render = () => {
    const index = window.DOCS_INDEX || [];
    const terms = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const hits = terms.length
      ? index
          .map((e) => {
            const hay = `${e.t} ${e.g}`.toLowerCase();
            if (!terms.every((t) => hay.includes(t))) return null;
            const title = e.t.toLowerCase();
            const score = (title.startsWith(terms[0]) ? 0 : 2) + (title.includes(terms.join(" ")) ? 0 : 1) + (e.u.includes("#") ? 1 : 0);
            return { e, score };
          })
          .filter(Boolean)
          .sort((a, b) => a.score - b.score)
          .slice(0, 30)
          .map((h) => h.e)
      : index.filter((e) => !e.u.includes("#"));
    active = 0;
    list.innerHTML = hits.length
      ? hits.map((e, i) => `<li role="option" aria-selected="${i === 0}"><a href="${e.u}">${mark(e.t, terms)}<span>${escapeHtml(e.g)}</span></a></li>`).join("")
      : `<li class="empty">No page or heading matches "${escapeHtml(input.value)}".</li>`;
  };
  const move = (delta) => {
    const items = [...list.querySelectorAll("li[role=option]")];
    if (!items.length) return;
    items[active].setAttribute("aria-selected", "false");
    active = (active + delta + items.length) % items.length;
    items[active].setAttribute("aria-selected", "true");
    items[active].scrollIntoView({ block: "nearest" });
  };
  const openSearch = () => {
    if (dialog.open) return;
    setMenu(false);
    dialog.showModal();
    input.select();
    render();
  };
  input.addEventListener("input", render);
  input.addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { e.preventDefault(); move(1); }
    else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
    else if (e.key === "Enter") {
      const link = list.querySelectorAll("li[role=option] a")[active];
      if (link) { e.preventDefault(); dialog.close(); location.href = link.href; }
    }
  });
  list.addEventListener("click", (e) => { if (e.target.closest("a")) dialog.close(); });
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });
  document.querySelector(".search-btn").addEventListener("click", openSearch);
  document.addEventListener("keydown", (e) => {
    const typing = /^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement?.tagName) || document.activeElement?.isContentEditable;
    if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !typing)) {
      e.preventDefault();
      openSearch();
    } else if (e.key === "Escape" && sidebar.classList.contains("open")) {
      setMenu(false);
    }
  });
})();
