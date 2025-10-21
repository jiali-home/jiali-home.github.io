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
      window.open("./assets/Resume_Jia_Li.pdf", "_blank", "noopener");
      return;
    }

    // If user clicks "Blog", go to Jekyll blog index
    if (label === "blog") {
      window.location.href = "./blog/";
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
  const container = document.getElementById('news-list');
  if (!container || !Array.isArray(items)) return;
  // Sort by date desc (YYYY-MM)
  items.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  container.innerHTML = items.map(item => {
    const html = createLinkHTML(item.text, item.links);
    // Each news in one line: date + text inline
    return `<p><span class="news-date">${item.date}:</span> ${html}</p>`;
  }).join('\n');
}

function escapeHTML(s) {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function renderPublications(items) {
  const list = document.getElementById('pubs-list');
  if (!list || !Array.isArray(items)) return;
  // Sort by date desc
  items.sort((a, b) => (b.date || '').localeCompare(a.date || ''));
  const liHTML = items.map(p => {
    const authors = (p.authors || []).map(name => {
      // Bold Jia Li in author list
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
