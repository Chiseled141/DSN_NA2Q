/**
 * Code Section Tab Controller
 * Handles switching between algorithm sections (NA²Q / HiT-MAC)
 * and code tabs within each section
 */

document.addEventListener('DOMContentLoaded', () => {
  initAlgorithmSelector();
  initCodeTabs();
});

/**
 * Initialize algorithm selector (NA²Q / HiT-MAC toggle)
 */
function initAlgorithmSelector() {
  const algoButtons = document.querySelectorAll('.algo-btn');
  const algoSections = document.querySelectorAll('.algo-code-section');

  if (algoButtons.length === 0) return;

  algoButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const algoName = btn.dataset.algo;

      // Remove active class from all algo buttons
      algoButtons.forEach(b => b.classList.remove('active'));

      // Hide all algo sections
      algoSections.forEach(s => s.classList.remove('active'));

      // Activate clicked button
      btn.classList.add('active');

      // Show corresponding algo section
      const targetSection = document.getElementById(`${algoName}-code`);
      if (targetSection) {
        targetSection.classList.add('active');
      }
    });
  });
}

/**
 * Initialize code tabs functionality within each algorithm section
 */
function initCodeTabs() {
  // Get all algo sections
  const algoSections = document.querySelectorAll('.algo-code-section');

  algoSections.forEach(section => {
    const tabButtons = section.querySelectorAll('.code-tab-btn');
    const codePanels = section.querySelectorAll('.code-panel');

    tabButtons.forEach(btn => {
      btn.addEventListener('click', () => {
        const tabName = btn.dataset.tab;

        // Remove active class from all buttons in this section
        tabButtons.forEach(b => b.classList.remove('active'));

        // Remove active class from all panels in this section
        codePanels.forEach(p => p.classList.remove('active'));

        // Add active class to clicked button
        btn.classList.add('active');

        // Show corresponding panel
        const targetPanel = document.getElementById(`${tabName}-panel`);
        if (targetPanel) {
          targetPanel.classList.add('active');
        }
      });
    });
  });
}
