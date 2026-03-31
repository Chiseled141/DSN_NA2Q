function checkLoginState() {
    const sessionData = sessionStorage.getItem('cyber_user');
    const localData = localStorage.getItem('cyber_user');
    const savedUser = sessionData || localData;
    if (savedUser) applyLoginState(JSON.parse(savedUser));
}

function applyLoginState(userData) {
    isLoggedIn = true;
    const navGuest = document.getElementById('nav-guest');
    const navUser = document.getElementById('nav-user');
    
    if (navGuest) { navGuest.classList.add('hidden'); navGuest.classList.remove('flex'); }
    if (navUser) {
        navUser.classList.remove('hidden'); navUser.classList.add('flex');
        
        const nameSpan = navUser.querySelector('span.truncate'); 
        if (nameSpan) nameSpan.textContent = userData.name;
        
        const coinDisplay = document.getElementById('nav-coin-display');
        if (coinDisplay) {
            const currentText = coinDisplay.textContent;
            const startCoin = parseInt(currentText.replace(/,/g, '')) || 0;
            const endCoin = userData.coin || 0;
            if (startCoin !== endCoin) animateCoinValue('nav-coin-display', startCoin, endCoin, 1500); 
            else coinDisplay.textContent = endCoin.toLocaleString(); 
        }

        const avatarDiv = navUser.querySelector('.w-7.h-7');
        if (avatarDiv) {
            const initials = userData.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
            avatarDiv.innerHTML = `<span class="text-black font-bold text-[10px] font-mono">${initials}</span>`;
        }

        // Show Mentor Hub link only for MENTOR role
        const mentorHubLink = document.getElementById('nav-mentor-hub-link');
        if (mentorHubLink) {
            const isMentor = (userData.role || '').toUpperCase() === 'MENTOR';
            mentorHubLink.classList.toggle('hidden', !isMentor);
        }
    }
}

async function doLogin() {
    const userEl = document.getElementById('login-user');
    const passEl = document.getElementById('login-pass');
    const keepPersistentEl = document.getElementById('keep-persistent');
    
    if (!userEl.value || !passEl.value) return showToast('Vui lòng nhập đủ thông tin!', 'error');

    try {
        const response = await fetch(`${API_BASE_URL}/auth/login`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: userEl.value, password: passEl.value })
        });

        if (response.ok) {
            const userData = await response.json();
            localStorage.removeItem('cyber_user'); localStorage.removeItem('cyber_token');
            sessionStorage.removeItem('cyber_user'); sessionStorage.removeItem('cyber_token');

            const storage = keepPersistentEl && keepPersistentEl.checked ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(userData));
            storage.setItem('cyber_token', userData.token); 
            
            closeModal('login-modal'); applyLoginState(userData);
            showToast(`ACCESS GRANTED. Chào mừng, ${userData.name}!`, 'success');
            userEl.value = ''; passEl.value = '';
        } else { showToast('ACCESS DENIED: Sai thông tin.', 'error'); }
    } catch (err) { showToast('SERVER ERROR: Lỗi kết nối.', 'error'); }
}

function logout() {
    localStorage.removeItem('cyber_user'); localStorage.removeItem('cyber_token');
    sessionStorage.removeItem('cyber_user'); sessionStorage.removeItem('cyber_token');
    isLoggedIn = false;

    const navGuest = document.getElementById('nav-guest');
    const navUser = document.getElementById('nav-user');
    if (navGuest) navGuest.classList.replace('hidden', 'flex');
    if (navUser) navUser.classList.replace('flex', 'hidden');

    showPage('home'); showToast('DISCONNECTED.', 'success');
}