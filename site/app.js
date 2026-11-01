(() => {
  const nav = document.querySelector('#site-nav');
  const toggle = document.querySelector('.menu-toggle');
  if (!nav || !toggle) return;
  document.documentElement.classList.add('js');
  const setOpen = (open, focus = false) => {
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Đóng điều hướng' : 'Mở điều hướng');
    nav.classList.toggle('is-open', open);
    if (focus) toggle.focus();
  };
  toggle.addEventListener('click', () => setOpen(toggle.getAttribute('aria-expanded') !== 'true'));
  nav.addEventListener('click', e => { if (e.target.closest('a')) setOpen(false); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape' && toggle.getAttribute('aria-expanded') === 'true') setOpen(false, true); });
  document.addEventListener('click', e => { if (!nav.contains(e.target) && !toggle.contains(e.target)) setOpen(false); });
  matchMedia('(min-width: 801px)').addEventListener('change', () => setOpen(false));
})();
