document.addEventListener('DOMContentLoaded', includeHTML);

function startCountdown() {
    const target = new Date('2026-04-12T00:00:00').getTime();

    function tick() {
        const now = Date.now();
        const diff = target - now;

        const days  = document.getElementById('cd-days');
        const hours = document.getElementById('cd-hours');
        const mins  = document.getElementById('cd-mins');
        const secs  = document.getElementById('cd-secs');
        if (!days) return;

        if (diff <= 0) {
            days.textContent = hours.textContent = mins.textContent = secs.textContent = '00';
            return;
        }

        days.textContent  = String(Math.floor(diff / 86400000)).padStart(2, '0');
        hours.textContent = String(Math.floor((diff % 86400000) / 3600000)).padStart(2, '0');
        mins.textContent  = String(Math.floor((diff % 3600000) / 60000)).padStart(2, '0');
        secs.textContent  = String(Math.floor((diff % 60000) / 1000)).padStart(2, '0');
    }

    tick();
    setInterval(tick, 1000);
}

function initializeApp() {
    console.log("[SYSTEM] Starting CyberTitans...");
    checkLoginState(); 
    
    if (typeof startCountdown === "function") startCountdown();
    buildRanking();
    buildTeam();
    buildProjects();
    buildPublications();
    buildFaqAndPolicies();
    buildAnnouncements();
    buildEvents();

    const activePage = document.querySelector('.page.active');
    if (activePage && activePage.id === 'page-my-profile') {
        loadOperativeData();
    }
}