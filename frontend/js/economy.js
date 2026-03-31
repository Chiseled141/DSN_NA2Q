async function purchaseTier(tierName, coinsAmount) {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token'); 

    if (!savedUserStr || !token) {
        showToast('Vui lòng Đăng nhập để giao dịch!', 'error');
        openModal('login-modal'); return;
    }

    const currentUser = JSON.parse(savedUserStr);
    if (coinsAmount === 0) return showToast(`Bạn đang ở cấp ${tierName}.`, 'success');

    showToast(`Đang thiết lập kết nối...`, 'success');
    const newCoinBalance = (currentUser.coin || 0) + coinsAmount;
    const payload = {
        name: currentUser.name, email: currentUser.email,
        phone: currentUser.phone, description: currentUser.description,
        role: tierName, coin: newCoinBalance
    };

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${currentUser.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            currentUser.coin = newCoinBalance; currentUser.role = tierName;
            const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(currentUser));
            showToast(`GIAO DỊCH THÀNH CÔNG!`, 'success');
            applyLoginState(currentUser); 
        } else { showToast('Connection denied.', 'error'); }
    } catch (error) { showToast('Connection failed.', 'error'); }
}

async function handleMentorRequest(mentorId, mentorName) {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token'); 

    if (!savedUserStr || !token) return showToast('Vui lòng Đăng nhập!', 'error');

    const currentUser = JSON.parse(savedUserStr);
    const MENTOR_COST = 500;

    if (!currentUser.coin || currentUser.coin < MENTOR_COST) return showToast(`Không đủ ${MENTOR_COST} Coins!`, 'error');

    try {
        showToast(`Đang kết nối tới ${mentorName}...`, 'success');
        const response = await fetch(`${API_BASE_URL}/team/members/${mentorId}/request-mentor`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify({ menteeId: currentUser.id })
        });

        if (response.ok) {
            const data = await response.json();
            currentUser.coin = data.remainingCoins;
            const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(currentUser));

            // Save request to localStorage so mentor sees it in their inbox
            const requests = JSON.parse(localStorage.getItem('mentorship_requests') || '[]');
            requests.push({
                id: 'req-' + Date.now(),
                mentorId: mentorId,
                mentorName: mentorName,
                menteeId: currentUser.id,
                menteeName: currentUser.name,
                status: 'PENDING',
                timestamp: new Date().toISOString()
            });
            localStorage.setItem('mentorship_requests', JSON.stringify(requests));

            applyLoginState(currentUser);
            showToast(`Request sent to ${mentorName}! −${MENTOR_COST} Coins.`, 'success');

            setTimeout(() => { showToast(`[SYSTEM] ${mentorName} has received your request!`, 'success'); }, 3000);
        } else {
            const errorData = await response.json();
            showToast(`LỖI: ${errorData.message}`, 'error');
        }
    } catch (error) { showToast('Connection failed.', 'error'); }
}

// UC008 — Cancel own pending mentorship request
function getMyPendingRequest(mentorId) {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    if (!savedUserStr) return null;
    const currentUser = JSON.parse(savedUserStr);
    const requests = JSON.parse(localStorage.getItem('mentorship_requests') || '[]');
    return requests.find(r => r.mentorId === mentorId && r.menteeId === currentUser.id && r.status === 'PENDING') || null;
}

function cancelMentorRequest(mentorId) {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    if (!savedUserStr) return;
    const currentUser = JSON.parse(savedUserStr);
    const requests = JSON.parse(localStorage.getItem('mentorship_requests') || '[]');
    const updated = requests.filter(r => !(r.mentorId === mentorId && r.menteeId === currentUser.id && r.status === 'PENDING'));
    localStorage.setItem('mentorship_requests', JSON.stringify(updated));
    showToast('Mentorship request cancelled.', 'success');
    openProfileModal(mentorId); // Re-render the modal to reflect new state
}