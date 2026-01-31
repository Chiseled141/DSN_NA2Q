/**
 * Professional Scroll Animations
 * Uses Intersection Observer to trigger fade-in and slide-up effects
 */

document.addEventListener('DOMContentLoaded', () => {
    // Options for the observer
    const options = {
        root: null, // Use the viewport
        rootMargin: '0px',
        threshold: 0.15 // Trigger when 15% of the element is visible
    };

    // Callback function when intersection changes
    const handleIntersect = (entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
                // Optional: Stop observing once visible to run animation only once
                observer.unobserve(entry.target);
            }
        });
    };

    // Create the observer
    const observer = new IntersectionObserver(handleIntersect, options);

    // Elements to animate
    const animatedElements = document.querySelectorAll('.animate-on-scroll');

    // Also target existing sections for backward compatibility/ease of use
    const sections = document.querySelectorAll('section, .overview-card, .algorithm-card');

    // Combine and observe
    [...animatedElements, ...sections].forEach(el => {
        el.classList.add('scroll-hidden'); // Ensure initial state is hidden
        observer.observe(el);
    });

    // Staggered animations for grids
    const grids = document.querySelectorAll('.overview-grid, .algorithms-grid');
    grids.forEach(grid => {
        const children = grid.children;
        Array.from(children).forEach((child, index) => {
            child.style.transitionDelay = `${index * 100}ms`; // 100ms stagger
        });
    });
});
