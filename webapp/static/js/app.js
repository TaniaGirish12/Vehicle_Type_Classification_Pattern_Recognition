/* VehicleVision — shared JS (runs on every page) */

// Active nav link highlight is handled server-side via Jinja.
// This file handles any global UI behaviour.

document.addEventListener('DOMContentLoaded', () => {
  // Animate elements that enter viewport
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) {
          e.target.classList.add('visible');
          io.unobserve(e.target);
        }
      });
    }, { threshold: 0.1 });

    document.querySelectorAll('.card, .stat-card, .model-arch-card').forEach(el => {
      el.classList.add('fade-in-ready');
      io.observe(el);
    });
  }
});
