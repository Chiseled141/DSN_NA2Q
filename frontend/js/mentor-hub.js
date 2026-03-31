// ─── MENTOR HUB ─────────────────────────────────────────────────────────────
// UC014 Approve Mentorship Request
// UC015 Decline Mentorship Request
// UC018 Guide Project Members
// UC019 View Assigned Mentees

function _getRequests() {
    try { return JSON.parse(localStorage.getItem('mentorship_requests') || '[]'); } catch { return []; }
}
function _saveRequests(arr) {
    localStorage.setItem('mentorship_requests', JSON.stringify(arr));
}
function _getGuidance() {
    try { return JSON.parse(localStorage.getItem('mentor_guidance') || '[]'); } catch { return []; }
}

function buildMentorHub() {
    _updateInboxBadge();
    showMentorHubTab('inbox');
}

function showMentorHubTab(tab) {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    if (!savedUserStr) return;
    const currentUser = JSON.parse(savedUserStr);

    ['inbox', 'assigned', 'guide'].forEach(t => {
        const btn = document.getElementById('tab-' + t);
        if (!btn) return;
        if (t === tab) {
            btn.classList.add('border-primary', 'text-primary');
            btn.classList.remove('border-transparent', 'text-gray-500');
        } else {
            btn.classList.remove('border-primary', 'text-primary');
            btn.classList.add('border-transparent', 'text-gray-500');
        }
    });

    const content = document.getElementById('mentor-hub-content');
    if (!content) return;

    const requests = _getRequests();
    const mine = requests.filter(r => r.mentorId === currentUser.id);

    if (tab === 'inbox') {
        const pending = mine.filter(r => r.status === 'PENDING');
        if (!pending.length) {
            content.innerHTML = `<div class="text-center py-20 font-mono text-gray-500 text-xs uppercase tracking-widest">[ No pending requests ]</div>`;
            return;
        }
        content.innerHTML = pending.map(r => `
            <div class="bg-[#0a0a0a] border border-white/5 hover:border-primary/20 p-6 transition-all">
                <div class="flex items-start justify-between gap-4 flex-wrap">
                    <div>
                        <div class="flex items-center gap-2 mb-2">
                            <span class="font-mono text-[9px] uppercase tracking-widest text-yellow-400 border border-yellow-400/30 px-2 py-0.5 bg-yellow-400/5">PENDING</span>
                            <span class="font-mono text-[10px] text-gray-500">${new Date(r.timestamp).toLocaleDateString('en-GB', {day:'2-digit',month:'short',year:'numeric'})}</span>
                        </div>
                        <p class="text-white font-bold text-lg font-headline">${r.menteeName}</p>
                        <p class="font-mono text-xs text-gray-500 mt-0.5">Mentee ID: #${String(r.menteeId).padStart(4, '0')}</p>
                    </div>
                    <div class="flex gap-2 flex-shrink-0">
                        <button onclick="approveRequest('${r.id}')"
                            class="px-5 py-2.5 bg-primary text-black text-[10px] font-bold uppercase tracking-widest hover:brightness-110 transition-all flex items-center gap-1.5">
                            <span class="material-symbols-outlined text-[14px]">check</span> ACCEPT
                        </button>
                        <button onclick="declineRequest('${r.id}')"
                            class="px-5 py-2.5 bg-red-600/20 border border-red-500/50 text-red-400 text-[10px] font-bold uppercase tracking-widest hover:bg-red-600 hover:text-white transition-all flex items-center gap-1.5">
                            <span class="material-symbols-outlined text-[14px]">close</span> DECLINE
                        </button>
                    </div>
                </div>
            </div>`).join('');

    } else if (tab === 'assigned') {
        const accepted = mine.filter(r => r.status === 'ACCEPTED');
        if (!accepted.length) {
            content.innerHTML = `<div class="text-center py-20 font-mono text-gray-500 text-xs uppercase tracking-widest">[ No assigned mentees yet ]</div>`;
            return;
        }
        content.innerHTML = accepted.map(r => `
            <div class="bg-[#0a0a0a] border border-white/5 hover:border-primary/20 p-6 transition-all">
                <div class="flex items-center justify-between gap-4 flex-wrap">
                    <div>
                        <span class="font-mono text-[9px] uppercase tracking-widest text-primary border border-primary/30 px-2 py-0.5 bg-primary/5">ACTIVE</span>
                        <p class="text-white font-bold text-lg font-headline mt-2">${r.menteeName}</p>
                        <p class="font-mono text-xs text-gray-500 mt-0.5">Since ${new Date(r.timestamp).toLocaleDateString('en-GB', {day:'2-digit',month:'short',year:'numeric'})}</p>
                    </div>
                    <button onclick="openGuideForm('${r.menteeId}', '${r.menteeName}')"
                        class="px-5 py-2.5 border border-primary/30 text-primary text-[10px] font-mono uppercase tracking-widest hover:bg-primary hover:text-black transition-all flex items-center gap-1.5">
                        <span class="material-symbols-outlined text-[14px]">edit_note</span> GUIDE
                    </button>
                </div>
            </div>`).join('');

    } else if (tab === 'guide') {
        const accepted = mine.filter(r => r.status === 'ACCEPTED');
        const guidance = _getGuidance().filter(g => g.mentorId === currentUser.id);

        const menteeOptions = accepted.length
            ? accepted.map(r => `<option value="${r.menteeId}" data-name="${r.menteeName}">${r.menteeName}</option>`).join('')
            : '<option disabled>No assigned mentees yet</option>';

        const guidanceLog = guidance.length
            ? guidance.map(g => `
                <div class="border border-white/5 bg-[#0a0a0a] p-4 hover:border-primary/10 transition-all">
                    <div class="flex items-center justify-between mb-2">
                        <span class="font-mono text-[10px] text-primary uppercase">${g.menteeName}</span>
                        <span class="font-mono text-[9px] text-gray-500">${new Date(g.timestamp).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})}</span>
                    </div>
                    <p class="font-mono text-xs text-gray-400 leading-relaxed">${g.content}</p>
                </div>`).join('')
            : '<p class="font-mono text-xs text-gray-500 text-center py-6">No guidance on record yet.</p>';

        content.innerHTML = `
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div>
                    <h3 class="font-headline text-sm font-bold uppercase tracking-widest mb-5 text-white flex items-center gap-3">
                        <span class="w-2 h-2 bg-primary"></span>Send Guidance
                    </h3>
                    <div class="space-y-4">
                        <div>
                            <label class="font-mono text-[10px] uppercase tracking-widest text-gray-500 block mb-2">Select Mentee</label>
                            <select id="guide-mentee-select" class="w-full bg-[#0a0a0a] border border-white/10 text-white font-mono text-sm p-3 focus:border-primary outline-none">
                                <option value="">— Select operative —</option>
                                ${menteeOptions}
                            </select>
                        </div>
                        <div>
                            <label class="font-mono text-[10px] uppercase tracking-widest text-gray-500 block mb-2">Guidance Message</label>
                            <textarea id="guide-content" rows="6"
                                placeholder="Write guidance, feedback, or tasks for your mentee..."
                                class="w-full bg-[#0a0a0a] border border-white/10 text-white font-mono text-sm p-3 focus:border-primary outline-none resize-none placeholder:text-gray-600 leading-relaxed"></textarea>
                        </div>
                        <button onclick="submitGuidance()"
                            class="w-full py-3.5 bg-primary text-black font-bold font-mono text-[10px] uppercase tracking-widest hover:brightness-110 transition-all flex items-center justify-center gap-2">
                            <span class="material-symbols-outlined text-[14px]">send</span> TRANSMIT GUIDANCE
                        </button>
                    </div>
                </div>
                <div>
                    <h3 class="font-headline text-sm font-bold uppercase tracking-widest mb-5 text-white flex items-center gap-3">
                        <span class="w-2 h-2 bg-secondary"></span>Guidance Log
                    </h3>
                    <div class="space-y-3 max-h-[500px] overflow-y-auto hide-scrollbar">${guidanceLog}</div>
                </div>
            </div>`;
    }
}

function approveRequest(reqId) {
    const requests = _getRequests();
    const req = requests.find(r => r.id === reqId);
    if (!req) return;
    req.status = 'ACCEPTED';
    _saveRequests(requests);
    showToast(`${req.menteeName} accepted as mentee!`, 'success');
    _updateInboxBadge();
    showMentorHubTab('inbox');
}

function declineRequest(reqId) {
    const requests = _getRequests();
    const req = requests.find(r => r.id === reqId);
    if (!req) return;
    req.status = 'DECLINED';
    _saveRequests(requests);
    showToast(`Request from ${req.menteeName} declined.`, 'success');
    _updateInboxBadge();
    showMentorHubTab('inbox');
}

function openGuideForm(menteeId, menteeName) {
    showMentorHubTab('guide');
    setTimeout(() => {
        const sel = document.getElementById('guide-mentee-select');
        if (sel) sel.value = menteeId;
    }, 50);
}

function submitGuidance() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    if (!savedUserStr) return;
    const currentUser = JSON.parse(savedUserStr);

    const sel = document.getElementById('guide-mentee-select');
    const textarea = document.getElementById('guide-content');

    if (!sel || !sel.value) return showToast('Select a mentee first.', 'error');
    if (!textarea || !textarea.value.trim()) return showToast('Guidance message cannot be empty.', 'error');

    const menteeName = sel.options[sel.selectedIndex].dataset.name || sel.options[sel.selectedIndex].text;

    const guidance = _getGuidance();
    guidance.unshift({
        mentorId: currentUser.id,
        menteeId: parseInt(sel.value),
        menteeName,
        content: textarea.value.trim(),
        timestamp: new Date().toISOString()
    });
    localStorage.setItem('mentor_guidance', JSON.stringify(guidance));

    textarea.value = '';
    showToast('Guidance transmitted!', 'success');
    showMentorHubTab('guide');
}

function _updateInboxBadge() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    if (!savedUserStr) return;
    const currentUser = JSON.parse(savedUserStr);
    const count = _getRequests().filter(r => r.mentorId === currentUser.id && r.status === 'PENDING').length;
    const badge = document.getElementById('mentor-inbox-badge');
    if (!badge) return;
    badge.textContent = count;
    badge.classList.toggle('hidden', count === 0);
}
