// Script to add 'visible' class to all elements with 'fade-in' after page load
window.addEventListener('DOMContentLoaded', function() {
  document.querySelectorAll('.fade-in').forEach(function(el) {
    el.classList.add('visible');
  });
});
