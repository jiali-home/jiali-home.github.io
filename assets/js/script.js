'use strict';



// element toggle function
const elementToggleFunc = function (elem) { elem.classList.toggle("active"); }



// sidebar variables
const sidebar = document.querySelector("[data-sidebar]");
const sidebarBtn = document.querySelector("[data-sidebar-btn]");

// sidebar toggle functionality for mobile
sidebarBtn.addEventListener("click", function () { elementToggleFunc(sidebar); });



// testimonials variables
const testimonialsItem = document.querySelectorAll("[data-testimonials-item]");
const modalContainer = document.querySelector("[data-modal-container]");
const modalCloseBtn = document.querySelector("[data-modal-close-btn]");
const overlay = document.querySelector("[data-overlay]");

// modal variable
const modalImg = document.querySelector("[data-modal-img]");
const modalTitle = document.querySelector("[data-modal-title]");
const modalText = document.querySelector("[data-modal-text]");

// modal toggle function
const testimonialsModalFunc = function () {
  modalContainer.classList.toggle("active");
  overlay.classList.toggle("active");
}

// add click event to all modal items
for (let i = 0; i < testimonialsItem.length; i++) {

  testimonialsItem[i].addEventListener("click", function () {

    modalImg.src = this.querySelector("[data-testimonials-avatar]").src;
    modalImg.alt = this.querySelector("[data-testimonials-avatar]").alt;
    modalTitle.innerHTML = this.querySelector("[data-testimonials-title]").innerHTML;
    modalText.innerHTML = this.querySelector("[data-testimonials-text]").innerHTML;

    testimonialsModalFunc();

  });

}

// add click event to modal close button
modalCloseBtn.addEventListener("click", testimonialsModalFunc);
overlay.addEventListener("click", testimonialsModalFunc);



// custom select variables
const select = document.querySelector("[data-select]");
const selectItems = document.querySelectorAll("[data-select-item]");
const selectValue = document.querySelector("[data-selecct-value]");
const filterBtn = document.querySelectorAll("[data-filter-btn]");

select.addEventListener("click", function () { elementToggleFunc(this); });

// add event in all select items
for (let i = 0; i < selectItems.length; i++) {
  selectItems[i].addEventListener("click", function () {

    let selectedValue = this.innerText.toLowerCase();
    selectValue.innerText = this.innerText;
    elementToggleFunc(select);
    filterFunc(selectedValue);

  });
}

// filter variables
const filterItems = document.querySelectorAll("[data-filter-item]");

const filterFunc = function (selectedValue) {

  for (let i = 0; i < filterItems.length; i++) {

    if (selectedValue === "all") {
      filterItems[i].classList.add("active");
    } else if (selectedValue === filterItems[i].dataset.category) {
      filterItems[i].classList.add("active");
    } else {
      filterItems[i].classList.remove("active");
    }

  }

}

// add event in all filter button items for large screen
let lastClickedBtn = filterBtn[0];

for (let i = 0; i < filterBtn.length; i++) {

  filterBtn[i].addEventListener("click", function () {

    let selectedValue = this.innerText.toLowerCase();
    selectValue.innerText = this.innerText;
    filterFunc(selectedValue);

    lastClickedBtn.classList.remove("active");
    this.classList.add("active");
    lastClickedBtn = this;

  });

}



// contact form variables
const form = document.querySelector("[data-form]");
const formInputs = document.querySelectorAll("[data-form-input]");
const formBtn = document.querySelector("[data-form-btn]");

// add event to all form input field
for (let i = 0; i < formInputs.length; i++) {
  formInputs[i].addEventListener("input", function () {

    // check form validation
    if (form.checkValidity()) {
      formBtn.removeAttribute("disabled");
    } else {
      formBtn.setAttribute("disabled", "");
    }

  });
}



// page navigation variables
const navigationLinks = document.querySelectorAll("[data-nav-link]");
const pages = document.querySelectorAll("[data-page]");

// add event to all nav link
for (let i = 0; i < navigationLinks.length; i++) {
  navigationLinks[i].addEventListener("click", function () {
    const label = this.innerHTML.trim().toLowerCase();

    // If user clicks "Resume", open the PDF in a new tab
    if (label === "resume") {
      window.open("./assets/Resume_JiaLi_Oct2026.pdf", "_blank", "noopener");
      return;
    }

    // Switch visible page based on label
    for (let j = 0; j < pages.length; j++) {
      if (label === pages[j].dataset.page) {
        pages[j].classList.add("active");
        window.scrollTo(0, 0);
      } else {
        pages[j].classList.remove("active");
      }
    }

    // Update nav link active state (independent of pages list length)
    for (let k = 0; k < navigationLinks.length; k++) {
      if (navigationLinks[k] === this) {
        navigationLinks[k].classList.add("active");
      } else {
        navigationLinks[k].classList.remove("active");
      }
    }
  });
}


// --- Dynamic content: News & Publications ---

async function fetchJSON(path) {
  try {
    const res = await fetch(path, { cache: 'no-cache' });
    if (!res.ok) throw new Error(`Failed to fetch ${path}`);
    return await res.json();
  } catch (e) {
    console.error(e);
    return null;
  }
}

function formatNewsText(text) {
  const escaped = escapeHTML(text || '');
  return escaped.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
}

function createLinkHTML(text, links) {
  if (!links || !links.length) return text;
  const tail = links.map(l => `<a href="${escapeHTML(l.href)}">${escapeHTML(l.label)}</a>`).join(', ');
  return `${text} (${tail})`;
}

function renderNewsItemHTML(item) {
  const iconHTML = item.icon
    ? `<ion-icon class="news-row-icon" name="${escapeHTML(item.icon)}" aria-hidden="true"></ion-icon>`
    : '';
  const textHTML = createLinkHTML(formatNewsText(item.text), item.links);
  return `<p class="news-row">${iconHTML}<span class="news-date">${escapeHTML(item.date)}:</span> <span class="news-body">${textHTML}</span></p>`;
}

function renderNews(items) {
  const host = document.getElementById('news-list');
  if (!host || !Array.isArray(items)) return;

  // Sort by date desc (YYYY-MM)
  items.sort((a, b) => (b.date || '').localeCompare(a.date || ''));

  // Build scrollable container (manual user scroll)
  const ticker = document.createElement('div');
  ticker.className = 'news-ticker has-scrollbar';

  const track = document.createElement('div');
  track.className = 'news-track';

  const rows = items.map(renderNewsItemHTML).join('\n');
  track.innerHTML = rows;

  ticker.appendChild(track);
  host.innerHTML = '';
  host.appendChild(ticker);

  // If not overflowing, remove max-height so it displays fully
  requestAnimationFrame(() => {
    const overflows = track.scrollHeight > ticker.clientHeight + 1;
    if (!overflows) ticker.classList.add('no-scroll');
  });
}

function escapeHTML(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function publicationAuthorsHTML(authors) {
  return (authors || []).map(name => {
    return name === 'Jia Li' ? `<strong>${escapeHTML(name)}</strong>` : escapeHTML(name);
  }).join(', ');
}

function publicationLinkLabel(url) {
  if (!url || url === '#') return 'Link';
  if (/arxiv\.org/i.test(url)) return 'arXiv';
  if (/\.pdf($|\?)/i.test(url)) return 'PDF';
  return 'Link';
}

function isArxivUrl(url) {
  return /arxiv\.org/i.test(url || '');
}

function parseVenueAndAward(p) {
  const raw = p.venue || p.tag || '';
  let award = p.award || '';
  let venue = raw;

  if (raw.includes('|')) {
    const parts = raw.split('|').map(part => part.trim()).filter(Boolean);
    venue = parts[0] || '';
    if (!award) {
      const awardParts = parts.slice(1).filter(part => /award/i.test(part));
      if (awardParts.length) award = awardParts.join(' | ');
    }
  }

  if (!award && /best paper award/i.test(raw)) {
    award = 'Best Paper Award';
    venue = raw.replace(/\s*\|\s*Best Paper Award/i, '').trim();
  }

  return { venue, award };
}

function publicationResourceIcon(label, url) {
  const text = `${label} ${url}`.toLowerCase();
  if (text.includes('pdf')) return 'document-text-outline';
  if (text.includes('arxiv') || text.includes('paper')) return 'book-outline';
  if (text.includes('poster')) return 'image-outline';
  if (text.includes('cite')) return 'chatbox-ellipses-outline';
  if (text.includes('code') || text.includes('github')) return 'logo-github';
  if (text.includes('project') || text.includes('demo') || text.includes('site')) return 'rocket-outline';
  if (text.includes('video') || text.includes('talk')) return 'videocam-outline';
  return 'link-outline';
}

function publicationResourceHTML(label, url) {
  const safeLabel = escapeHTML(label);
  const safeUrl = url || '#';

  if (isArxivUrl(safeUrl)) {
    return `<button type="button" class="pub-resource pub-resource--external" data-external-url="${safeUrl}" title="${safeLabel}">
      <ion-icon name="book-outline" aria-hidden="true"></ion-icon>
      <span>arXiv</span>
    </button>`;
  }

  const externalAttrs = safeUrl.startsWith('http') || safeUrl.endsWith('.pdf')
    ? ' target="_blank" rel="noopener noreferrer"'
    : '';
  const icon = publicationResourceIcon(label, safeUrl);

  return `<a class="pub-resource" href="${safeUrl}"${externalAttrs} title="${safeLabel}">
    <ion-icon name="${icon}" aria-hidden="true"></ion-icon>
    <span>${safeLabel}</span>
  </a>`;
}

function initPubThumbFallbacks(host) {
  host.querySelectorAll('.pub-entry-thumb img[data-fallback-src]').forEach(img => {
    img.addEventListener('error', function handleThumbError() {
      const fallback = this.dataset.fallbackSrc;
      if (!fallback || this.dataset.fallbackApplied === 'true') return;
      this.dataset.fallbackApplied = 'true';
      this.src = fallback;
    }, { once: true });
  });
}

function initPubResourceLinks(host) {
  if (!host || host.dataset.pubResourcesBound === 'true') return;
  host.addEventListener('click', event => {
    const button = event.target.closest('[data-external-url]');
    if (!button) return;
    event.preventDefault();
    const url = button.dataset.externalUrl;
    if (url) window.open(url, '_blank', 'noopener');
  });
  host.dataset.pubResourcesBound = 'true';
}

function publicationSlug(title) {
  return (title || 'paper')
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80);
}

function publicationCiteButtonHTML(p) {
  const slug = publicationSlug(p.title);
  return `<button type="button" class="pub-resource pub-resource--cite" data-pub-cite="${slug}" title="Cite">
    <ion-icon name="chatbox-ellipses-outline" aria-hidden="true"></ion-icon>
    <span>Cite</span>
  </button>`;
}

function getCatalogThumb(p) {
  if (p.catalogThumb) return p.catalogThumb;
  if (p.image) return p.image;
  return '';
}

function publicationCatalogEntryHTML(p) {
  const title = escapeHTML(p.title || '');
  const link = p.link && p.link !== '#' ? p.link : '';
  const titleLink = link && !isArxivUrl(link) ? link : '';
  const titleHTML = titleLink
    ? `<a href="${titleLink}" target="_blank" rel="noopener noreferrer">${title}</a>`
    : title;
  const authors = publicationAuthorsHTML(p.authors);
  const { venue, award } = parseVenueAndAward(p);
  const venueHTML = venue ? `<em>${escapeHTML(venue)}</em>` : '';
  const image = getCatalogThumb(p);
  const fallbackImage = p.image || '';
  const thumbLink = titleLink || '';
  const extraLinks = Array.isArray(p.links) ? p.links : [];

  const actions = [];
  if (link) {
    actions.push(publicationResourceHTML(publicationLinkLabel(link), link));
  }
  extraLinks.forEach(item => {
    const url = item.url || item.href || '#';
    const label = item.label || 'Link';
    actions.push(publicationResourceHTML(label, url));
  });
  if (!p.comingSoon) {
    actions.push(publicationCiteButtonHTML(p));
  }

  const actionsHTML = p.comingSoon
    ? `<div class="pub-entry-actions"><span class="pub-entry-coming-soon">Paper and materials will be released soon.</span></div>`
    : actions.length
      ? `<div class="pub-entry-actions">${actions.join('')}</div>`
      : '';

  const awardBadgeHTML = award
    ? '<span class="pub-award-badge">Award</span>'
    : '';

  const thumbHTML = image
    ? (thumbLink
      ? `<a class="pub-entry-thumb" href="${thumbLink}" target="_blank" rel="noopener noreferrer">
          <div class="pub-entry-thumb-frame">
            <img src="${image}" alt="${title} thumbnail" loading="lazy"${fallbackImage && fallbackImage !== image ? ` data-fallback-src="${fallbackImage}"` : ''}>
          </div>
          ${awardBadgeHTML}
        </a>`
      : `<div class="pub-entry-thumb" role="img" aria-label="${title}">
          <div class="pub-entry-thumb-frame">
            <img src="${image}" alt="${title} thumbnail" loading="lazy"${fallbackImage && fallbackImage !== image ? ` data-fallback-src="${fallbackImage}"` : ''}>
          </div>
          ${awardBadgeHTML}
        </div>`)
    : `<div class="pub-entry-thumb pub-entry-thumb--empty" aria-hidden="true">
        <div class="pub-entry-thumb-frame"></div>
        ${awardBadgeHTML}
      </div>`;

  const metaBits = [];
  if (venueHTML) metaBits.push(`<span class="pub-entry-venue">${venueHTML}</span>`);
  if (award) {
    metaBits.push(`<span class="pub-entry-award"><ion-icon name="trophy-outline" aria-hidden="true"></ion-icon>${escapeHTML(award)}</span>`);
  }
  const metaHTML = metaBits.length
    ? `<div class="pub-entry-meta">${metaBits.join('')}</div>`
    : '';

  return `
    <div class="pub-entry">
      ${thumbHTML}
      <div class="pub-entry-body">
        <h3 class="pub-entry-title">${titleHTML}</h3>
        <p class="pub-entry-authors">${authors}</p>
        ${metaHTML}
        ${actionsHTML}
      </div>
    </div>`;
}

function initPubYearNav(host) {
  const nav = host.querySelector('[data-pub-year-nav]');
  const main = host.querySelector('[data-pub-catalog-main]');
  if (!nav || !main) return;

  const links = [...nav.querySelectorAll('[data-pub-year-link]')];
  const sections = [...main.querySelectorAll('[data-pub-year-section]')];
  if (!links.length || !sections.length) return;

  const setActiveYear = year => {
    links.forEach(link => {
      link.classList.toggle('active', link.dataset.pubYearLink === year);
    });
  };

  links.forEach(link => {
    link.addEventListener('click', event => {
      event.preventDefault();
      const year = link.dataset.pubYearLink;
      const section = main.querySelector(`#pub-year-${year}`);
      if (!section) return;
      section.scrollIntoView({ behavior: 'smooth', block: 'start' });
      setActiveYear(year);
    });
  });

  if ('IntersectionObserver' in window) {
    const observer = new IntersectionObserver(entries => {
      const visible = entries
        .filter(entry => entry.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (visible.length) {
        setActiveYear(visible[0].target.dataset.pubYearSection);
      }
    }, {
      root: null,
      rootMargin: '-20% 0px -55% 0px',
      threshold: [0, 0.2, 0.5, 1]
    });

    sections.forEach(section => observer.observe(section));
  }

  setActiveYear(links[0].dataset.pubYearLink);
}

let publicationBibtexMap = {};

function getPublicationBibtex(slug, publication) {
  if (publicationBibtexMap[slug]?.bibtex) {
    return publicationBibtexMap[slug];
  }
  if (publication?.bibtex) {
    return { source: 'manual', bibtex: publication.bibtex };
  }
  return null;
}

function openPubCiteModal(slug, publication) {
  const modal = document.getElementById('pub-cite-modal');
  const content = document.getElementById('pub-cite-content');
  const source = document.getElementById('pub-cite-source');
  const title = document.getElementById('pub-cite-title');
  if (!modal || !content || !source || !title) return;

  const record = getPublicationBibtex(slug, publication);
  const bibtex = record?.bibtex || `@misc{${slug},\n  title={${publication?.title || 'Untitled'}},\n}`;
  const sourceLabel = {
    doi: 'Source: DOI / Crossref',
    arxiv: 'Source: arXiv',
    manual: 'Source: manual entry',
    generated: 'Source: generated from metadata'
  }[record?.source || 'generated'] || 'Source: generated from metadata';

  title.textContent = publication?.title || 'BibTeX';
  source.textContent = sourceLabel;
  content.textContent = bibtex;
  modal.classList.add('active');
  modal.setAttribute('aria-hidden', 'false');
}

function closePubCiteModal() {
  const modal = document.getElementById('pub-cite-modal');
  if (!modal) return;
  modal.classList.remove('active');
  modal.setAttribute('aria-hidden', 'true');
}

function initPubCiteModal(host, publications) {
  const modal = document.getElementById('pub-cite-modal');
  if (!modal || host.dataset.pubCiteBound === 'true') return;

  const publicationBySlug = new Map(
    (publications || []).map(item => [publicationSlug(item.title), item])
  );

  host.addEventListener('click', event => {
    const button = event.target.closest('[data-pub-cite]');
    if (!button) return;
    event.preventDefault();
    const slug = button.dataset.pubCite;
    openPubCiteModal(slug, publicationBySlug.get(slug));
  });

  modal.querySelectorAll('[data-pub-cite-close]').forEach(node => {
    node.addEventListener('click', closePubCiteModal);
  });

  document.addEventListener('keydown', event => {
    if (event.key === 'Escape' && modal.classList.contains('active')) {
      closePubCiteModal();
    }
  });

  const copyButton = document.getElementById('pub-cite-copy');
  const content = document.getElementById('pub-cite-content');
  if (copyButton && content) {
    copyButton.addEventListener('click', async () => {
      const text = content.textContent || '';
      try {
        await navigator.clipboard.writeText(text);
        const label = copyButton.querySelector('span');
        if (label) {
          const original = label.textContent;
          label.textContent = 'Copied!';
          setTimeout(() => { label.textContent = original; }, 1500);
        }
      } catch (error) {
        window.prompt('Copy BibTeX:', text);
      }
    });
  }

  host.dataset.pubCiteBound = 'true';
}

function renderPublicationsCatalog(items) {
  const host = document.getElementById('pub-catalog');
  if (!host || !Array.isArray(items)) return;

  const sortedItems = [...items].sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  const byYear = new Map();

  sortedItems.forEach(item => {
    const year = (item.date || '').slice(0, 4) || 'Other';
    if (!byYear.has(year)) byYear.set(year, []);
    byYear.get(year).push(item);
  });

  const years = [...byYear.keys()].sort((a, b) => b.localeCompare(a));

  const navHTML = `
    <nav class="pub-year-nav" data-pub-year-nav aria-label="Publication years">
      <ul class="pub-year-nav-list">
        ${years.map(year => `<li><a class="pub-year-nav-link" href="#pub-year-${year}" data-pub-year-link="${year}">${year}</a></li>`).join('')}
      </ul>
    </nav>`;

  const mainHTML = `
    <div class="pub-catalog-main" data-pub-catalog-main>
      ${years.map(year => `
        <section class="pub-year-group" id="pub-year-${year}" data-pub-year-section="${year}">
          <h2 class="pub-year-heading">${year}</h2>
          <div class="pub-year-list">
            ${byYear.get(year).map(publicationCatalogEntryHTML).join('\n')}
          </div>
        </section>
      `).join('\n')}
    </div>`;

  host.innerHTML = navHTML + mainHTML;
  initPubYearNav(host);
  initPubResourceLinks(host);
  initPubThumbFallbacks(host);
  initPubCiteModal(host, items);
}

function publicationCardHTML(p) {
  const authors = publicationAuthorsHTML(p.authors);
  const title = escapeHTML(p.title || '');
  const tag = p.tag ? `[${escapeHTML(p.tag)}]` : '';
  const description = escapeHTML(p.description || '');
  const image = p.image || '';
  const link = p.link && p.link !== '#' ? p.link : '';
  const alt = title;
  const mediaHTML = image
    ? `<figure class="blog-banner-box"><img src="${image}" alt="${alt}" loading="lazy"></figure>`
    : '';
  const bodyHTML = `
          <div class="blog-content">
            <h3 class="h3 blog-item-title">${tag ? `${tag} ` : ''}${title}</h3>
            <div class="blog-meta">
              <p class="blog-category">${authors}</p>
            </div>
            <p class="blog-text">${description}</p>
          </div>`;

  if (link) {
    return `
      <li class="blog-post-item">
        <a href="${link}">
          ${mediaHTML}
          ${bodyHTML}
        </a>
      </li>`;
  }

  return `
      <li class="blog-post-item">
        <div class="blog-post-card-static">
          ${mediaHTML}
          ${bodyHTML}
        </div>
      </li>`;
}

function renderPublications(items) {
  const modules = document.querySelectorAll('[data-publications-module]');
  if (!modules.length || !Array.isArray(items)) return;

  const sortedItems = [...items].sort((a, b) => (b.date || '').localeCompare(a.date || ''));

  modules.forEach(module => {
    const themeItems = module.querySelectorAll('[data-theme-item]');
    if (!themeItems.length) return;

    themeItems.forEach(item => {
      const theme = item.dataset.theme || 'perception';
      const list = item.querySelector('[data-pubs-list]');
      if (!list) return;
      const filtered = sortedItems.filter(entry => (entry.theme || 'perception') === theme);
      list.innerHTML = filtered.map(publicationCardHTML).join('\n');
    });
  });
}

function renderProjects(items) {
  const list = document.getElementById('projects-list');
  if (!list || !Array.isArray(items)) return;

  const liHTML = items.map(project => {
    const title = escapeHTML(project.title || '');
    const tag = project.tag ? `[${escapeHTML(project.tag)}]` : '';
    const description = escapeHTML(project.description || '');
    const image = project.image || '';
    const link = project.link || '#';
    const alt = title;
    const extraLinks = Array.isArray(project.links) ? project.links : [];
    const linkRow = extraLinks.length ? `
              <div class="blog-link-row">
                ${extraLinks.map(item => {
                  const label = escapeHTML(item.label || 'Link');
                  const url = item.url || '#';
                  const externalAttrs = item.external ? ' target="_blank" rel="noopener"' : '';
                  return `<a href="${url}"${externalAttrs}>${label}</a>`;
                }).join('\n')}
              </div>` : '';

    return `
      <li class="blog-post-item">
        <div class="blog-post-card">
          <a href="${link}">
            <figure class="blog-banner-box">
              <img src="${image}" alt="${alt}" loading="lazy">
            </figure>
            <div class="blog-content">
              <h3 class="h3 blog-item-title">${tag ? `${tag} ` : ''}${title}</h3>
              <div class="blog-meta">
                <p class="blog-category"></p>
              </div>
              <p class="blog-text">${description}</p>
            </div>
          </a>${linkRow}
        </div>
      </li>`;
  }).join('\n');

  list.innerHTML = liHTML;
}

(async function initDynamicSections() {
  // News
  const news = await fetchJSON('./assets/data/news.json');
  if (news) renderNews(news);

  // Publications
  const pubs = await fetchJSON('./assets/data/publications.json');
  const pubBibtex = await fetchJSON('./assets/data/publications-bibtex.json');
  if (pubBibtex) publicationBibtexMap = pubBibtex;
  if (pubs) {
    renderPublications(pubs);
    renderPublicationsCatalog(pubs);
  }

  // Projects
  const projects = await fetchJSON('./assets/data/projects.json');
  if (projects) renderProjects(projects);
  
  // Blog
  const blog = await fetchJSON('./assets/data/blog.json');
  if (blog) renderBlog(blog);
})();

// --- Blog rendering ---
function renderBlog(items) {
  const list = document.getElementById('blog-list');
  if (!list || !Array.isArray(items)) return;
  // Sort by date desc (YYYY-MM or YYYY-MM-DD)
  items.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  const liHTML = items.map(p => {
    const title = escapeHTML(p.title || '');
    const date = escapeHTML(p.date || '');
    const summary = escapeHTML(p.summary || '');
    const image = p.image || '';
    const link = p.link || '#';
    const alt = title;
    return `
      <li class="blog-post-item">
        <a href="${link}">
          <figure class="blog-banner-box">
            ${image ? `<img src="${image}" alt="${alt}" loading="lazy">` : ''}
          </figure>
          <div class="blog-content">
            <div class="blog-meta">
              <p class="blog-category">${date}</p>
            </div>
            <h3 class="h3 blog-item-title">${title}</h3>
            <p class="blog-text">${summary}</p>
          </div>
        </a>
      </li>`;
  }).join('\n');
  list.innerHTML = liHTML;
}
