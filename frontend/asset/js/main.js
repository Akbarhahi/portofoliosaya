// Modern script for scroll-triggered fade-in animations
document.addEventListener('DOMContentLoaded', function() {
  const fadeInElements = document.querySelectorAll('.fade-in');

  if (!fadeInElements.length) return;

  const observer = new IntersectionObserver((entries, observer) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add('visible');
        observer.unobserve(entry.target); // Stop observing after it's visible
      }
    });
  }, {
    threshold: 0.1 // Trigger when 10% of the element is visible
  });

  fadeInElements.forEach(el => {
    observer.observe(el);
  });
});
