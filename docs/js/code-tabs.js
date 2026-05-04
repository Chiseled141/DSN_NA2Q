/**
 * Code Section Tab Controller
 * Handles switching between algorithm sections (NA²Q / HiT-MAC)
 * and code tabs within each section
 */

document.addEventListener('DOMContentLoaded', () => {
  initScenarioTabs();
  initAlgorithmSelector();
  initCodeTabs();
});

/**
 * Initialize scenario tab switching (Scenario 1 / Scenario 2)
 */
function initScenarioTabs() {
  const tabButtons = document.querySelectorAll('.tab-btn[data-scenario]');
  const s1 = document.getElementById('scenario-1-content');
  const s2 = document.getElementById('scenario-2-content');

  if (!tabButtons.length || !s1 || !s2) return;

  tabButtons.forEach(btn => {
    btn.addEventListener('click', () => {
      const scenario = btn.dataset.scenario;

      tabButtons.forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      if (scenario === '1') {
        s1.hidden = false;
        s2.hidden = true;
      } else {
        s1.hidden = true;
        s2.hidden = false;
        if (window.initS2Charts) requestAnimationFrame(window.initS2Charts);
      }
    });
  });
}

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
