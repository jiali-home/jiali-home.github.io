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
      window.open("./assets/Resume_Jia_Li_Apr2026.pdf", "_blank", "noopener");
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

function createLinkHTML(text, links) {
  if (!links || !links.length) return text;
  let html = text;
  // Append links at the end in parentheses if not embedded
  const tail = links.map(l => `<a href="${l.href}">${l.label}</a>`).join(', ');
  if (tail) html += ` (${tail})`;
  return html;
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

  const rows = items.map(item => {
    const html = createLinkHTML(item.text, item.links);
    return `<p class="news-row"><span class="news-date">${item.date}:</span> ${html}</p>`;
  }).join('\n');
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

function publicationCardHTML(p) {
  const authors = (p.authors || []).map(name => {
    return name === 'Jia Li' ? `<strong>${escapeHTML(name)}</strong>` : escapeHTML(name);
  }).join(', ');
  const title = escapeHTML(p.title || '');
  const tag = p.tag ? `[${escapeHTML(p.tag)}]` : '';
  const description = escapeHTML(p.description || '');
  const image = p.image || '';
  const link = p.link || '#';
  const alt = title;
  return `
      <li class="blog-post-item">
        <a href="${link}">
          <figure class="blog-banner-box">
            <img src="${image}" alt="${alt}" loading="lazy">
          </figure>
          <div class="blog-content">
            <h3 class="h3 blog-item-title">${tag ? `${tag} ` : ''}${title}</h3>
            <div class="blog-meta">
              <p class="blog-category">${authors}</p>
            </div>
            <p class="blog-text">${description}</p>
          </div>
        </a>
      </li>`;
}

function renderPublications(items) {
  const modules = document.querySelectorAll('[data-publications-module]');
  if (!modules.length || !Array.isArray(items)) return;

  const sortedItems = [...items].sort((a, b) => (b.date || '').localeCompare(a.date || ''));

  modules.forEach(module => {
    const list = module.querySelector('[data-pubs-list]');
    const buttons = module.querySelectorAll('[data-pubs-theme-btn]');
    const panels = module.querySelectorAll('[data-theme-panel]');
    const banner = module.querySelector('[data-theme-banner]');
    const bannerKicker = module.querySelector('[data-theme-kicker]');
    const bannerTitle = module.querySelector('[data-theme-title]');
    if (!list || !buttons.length) return;

    let activeTheme = module.dataset.defaultTheme || buttons[0].dataset.theme;

    const paint = () => {
      buttons.forEach(button => {
        button.classList.toggle('active', button.dataset.theme === activeTheme);
      });

      panels.forEach(panel => {
        panel.classList.toggle('active', panel.dataset.themePanel === activeTheme);
      });

      const activeButton = module.querySelector(`[data-pubs-theme-btn][data-theme="${activeTheme}"]`);
      if (banner && activeButton) {
        banner.dataset.theme = activeTheme;
        if (bannerKicker) {
          const label = activeButton.querySelector('.research-theme-label');
          bannerKicker.textContent = label ? label.textContent : '';
        }
        if (bannerTitle) {
          const titleNode = activeButton.querySelector('.research-theme-title');
          bannerTitle.textContent = titleNode ? titleNode.textContent : '';
        }
      }

      const filtered = sortedItems.filter(item => (item.theme || 'perception') === activeTheme);
      list.innerHTML = filtered.map(publicationCardHTML).join('\n');
    };

    buttons.forEach(button => {
      button.addEventListener('click', function () {
        activeTheme = this.dataset.theme;
        paint();
      });
    });

    paint();
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
  if (pubs) renderPublications(pubs);

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
