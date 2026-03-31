async function loadOperativeData() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token'); 
    if (!savedUserStr || !token) return;
    
    const currentUser = JSON.parse(savedUserStr);
    const nameInput = document.getElementById('input-name');
    if (!nameInput) return; 

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${currentUser.id}`, { headers: { 'Authorization': `Bearer ${token}` } });
        if (response.ok) {
            const user = await response.json();
            nameInput.value = user.name || '';
            if (document.getElementById('input-email')) document.getElementById('input-email').value = user.email || '';
            if (document.getElementById('input-phone')) document.getElementById('input-phone').value = user.phone || '';
            if (document.getElementById('input-bio')) document.getElementById('input-bio').value = user.description || '';
            if (document.getElementById('input-username')) document.getElementById('input-username').value = currentUser.username || "@unknown";
            const roleEl = document.getElementById('profile-role');
            if (roleEl) roleEl.innerText = user.role || 'MEMBER';

            // Load tags from experiences
            const tags = (user.experiences || []).map(e => e.name).filter(Boolean);
            _renderTags(tags);
        } else if (response.status === 403 || response.status === 401) { logout(); }
    } catch (error) { console.error("[SYSTEM] Error loading user data:", error); }
}

// ── Tag system ─────────────────────────────────────────────────────────────

const _tagColors = ['tag-primary', 'tag-secondary', 'tag-tertiary', 'tag-gold'];

function _renderTags(tags) {
    const list = document.getElementById('tag-list');
    if (!list) return;
    list.innerHTML = tags.map((tag, i) => `
        <span class="inline-flex items-center gap-1.5 px-3 py-1 ${_tagColors[i % _tagColors.length]} text-[11px] font-mono">
            ${tag}
            <button onclick="removeTag('${tag.replace(/'/g, "\\'")}')" class="opacity-50 hover:opacity-100 transition-opacity leading-none">&times;</button>
        </span>`).join('');
}

function _getTags() {
    const list = document.getElementById('tag-list');
    if (!list) return [];
    return [...list.querySelectorAll('span')].map(s => s.childNodes[0].textContent.trim()).filter(Boolean);
}

function addTag() {
    const input = document.getElementById('tag-input');
    if (!input) return;
    const val = input.value.trim();
    if (!val) return;
    const current = _getTags();
    if (current.includes(val)) { showToast('Tag already exists', 'error'); return; }
    if (current.length >= 10) { showToast('Max 10 tags allowed', 'error'); return; }
    _renderTags([...current, val]);
    input.value = '';
    input.focus();
}

function removeTag(tag) {
    _renderTags(_getTags().filter(t => t !== tag));
}

function handleTagKeydown(event) {
    if (event.key === 'Enter') { event.preventDefault(); addTag(); }
}

async function handleAvatarUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate type & size (max 2MB)
    if (!file.type.startsWith('image/')) return showToast('Invalid file type. Images only.', 'error');
    if (file.size > 2 * 1024 * 1024)    return showToast('File too large. Max 2MB.', 'error');

    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');
    if (!savedUserStr || !token) return showToast('Session expired. Please log in again.', 'error');

    const currentUser = JSON.parse(savedUserStr);

    // Preview immediately (local blob URL)
    const previewUrl = URL.createObjectURL(file);
    const avatarImg = document.getElementById('profile-avatar');
    if (avatarImg) avatarImg.src = previewUrl;

    // Show uploading overlay
    const overlay = document.getElementById('avatar-upload-overlay');
    if (overlay) overlay.classList.remove('hidden');

    try {
        const formData = new FormData();
        formData.append('avatar', file);

        const response = await fetch(`${API_BASE_URL}/team/members/${currentUser.id}/avatar`, {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${token}` },
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            const newAvatarUrl = data.avatarUrl || previewUrl;

            // Persist new avatar URL in storage
            currentUser.avatar = newAvatarUrl;
            const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(currentUser));

            if (avatarImg) avatarImg.src = newAvatarUrl;
            showToast('Avatar updated successfully!', 'success');
        } else {
            // Backend not ready yet — keep local preview, save as base64 fallback
            const reader = new FileReader();
            reader.onload = e => {
                const base64 = e.target.result;
                currentUser.avatar = base64;
                const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
                storage.setItem('cyber_user', JSON.stringify(currentUser));
                if (avatarImg) avatarImg.src = base64;
            };
            reader.readAsDataURL(file);
            showToast('Avatar saved locally (server sync pending).', 'success');
        }
    } catch {
        // Offline fallback — base64 in storage
        const reader = new FileReader();
        reader.onload = e => {
            const base64 = e.target.result;
            currentUser.avatar = base64;
            const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(currentUser));
            if (avatarImg) avatarImg.src = base64;
        };
        reader.readAsDataURL(file);
        showToast('Avatar saved locally.', 'success');
    } finally {
        if (overlay) overlay.classList.add('hidden');
        // Reset input so same file can be re-selected
        event.target.value = '';
    }
}

async function saveAccountProfile() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');
    if (!savedUserStr || !token) return showToast('Lỗi: Phiên đăng nhập hết hạn!', 'error');

    const currentUser = JSON.parse(savedUserStr);
    const tags = _getTags();
    const payload = {
        name: document.getElementById('input-name').value,
        email: document.getElementById('input-email').value,
        phone: document.getElementById('input-phone').value,
        description: document.getElementById('input-bio').value,
        experiences: tags.map(t => ({ name: t }))
    };

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${currentUser.id}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json', 'Authorization': `Bearer ${token}` },
            body: JSON.stringify(payload)
        });
        if (response.ok) {
            showToast('Profile saved successfully.', 'success');
            currentUser.name = payload.name;
            const storage = localStorage.getItem('cyber_user') ? localStorage : sessionStorage;
            storage.setItem('cyber_user', JSON.stringify(currentUser));
            applyLoginState(currentUser);
        } else { showToast('Lỗi: Không có quyền sửa.', 'error'); }
    } catch (error) { showToast('Lỗi Server.', 'error'); }
}

async function openProfileModal(id) {
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');
    const modal = document.getElementById('profile-modal');
    const modalContent = document.getElementById('profile-modal-content');
    const modalBody = document.getElementById('modal-body');
    
    // 1. Kịch bản GUEST: Nếu không có token, yêu cầu đăng nhập
    if (!token) return showToast("Please log in to view profiles!", "error");

    modal.classList.remove('hidden');
    setTimeout(() => modalContent.classList.remove('translate-x-full'), 10);
    modalBody.innerHTML = '<p class="text-primary font-mono animate-pulse text-center mt-20">Establishing secure connection... Fetching operative data...</p>';

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${id}`, { 
            headers: { 'Authorization': `Bearer ${token}` } 
        });
        
        if (!response.ok) throw new Error("Data access denied.");
        const user = await response.json(); 
        
        // --- LOGIC PHÂN QUYỀN HIỂN THỊ NÚT BẤM ---
        const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
        const currentUser = JSON.parse(savedUserStr);
        const viewerRole = currentUser.role;

        let actionButtonsHTML = '';

        // KỊCH BẢN 1: Tự xem Profile của chính mình
        if (currentUser.id === user.id) {
            actionButtonsHTML = `
                <button onclick="showPage('my-profile'); closeProfileModal();" class="w-full bg-secondary text-black font-bold font-mono tracking-widest py-3.5 hover:bg-white transition-all text-[11px]">
                    EDIT MY PROFILE
                </button>`;
        }
        // KỊCH BẢN 2: Xem Profile của người khác
        else {
            const pendingReq = getMyPendingRequest(user.id);
            if (pendingReq) {
                actionButtonsHTML = `
                    <div class="w-full bg-yellow-500/10 border border-yellow-500/30 p-3 mb-3 text-center font-mono text-[10px] text-yellow-400 uppercase tracking-widest">
                        ● Request pending since ${new Date(pendingReq.timestamp).toLocaleDateString('en-GB',{day:'2-digit',month:'short',year:'numeric'})}
                    </div>
                    <button onclick="cancelMentorRequest(${user.id})" class="w-full bg-red-600/20 border border-red-500/50 text-red-400 font-bold font-mono tracking-widest py-3.5 hover:bg-red-600 hover:text-white transition-all text-[11px] mb-2">
                        CANCEL REQUEST
                    </button>`;
            } else {
                actionButtonsHTML = `
                    <button onclick="handleMentorRequest(${user.id}, '${user.name}')" class="w-full bg-primary text-black font-bold font-mono tracking-widest py-3.5 hover:bg-white transition-all text-[11px] mb-2">
                        MENTOR REQUEST (500 COINS)
                    </button>`;
            }
            actionButtonsHTML += `
                <button class="w-full bg-[#111] border border-white/10 text-white font-bold font-mono tracking-widest py-3.5 hover:border-primary transition-all text-[11px]">
                    MESSAGE
                </button>`;

            // ĐẶC QUYỀN ADMIN: Hiện thêm nút Xóa đặc vụ
            if (viewerRole === 'ADMIN' || viewerRole === 'SUPER ADMIN') {
                actionButtonsHTML += `
                    <button onclick="adminDeleteUser(${user.id})" class="w-full mt-4 bg-red-600/20 border border-red-500/50 text-red-500 font-bold font-mono tracking-widest py-3.5 hover:bg-red-600 hover:text-white transition-all text-[11px]">
                        [ADMIN] TERMINATE OPERATIVE
                    </button>`;
            }
        }

        // --- XỬ LÝ TIMELINE KINH NGHIỆM ---
        const defaultAvt = "https://ui-avatars.com/api/?background=222&color=fff&name=";
        const avatarUrl = user.avatar || (defaultAvt + user.name);

        let experiencesHTML = '<p class="text-gray-500 font-mono text-sm">No classified records found.</p>';
        if (user.experiences && user.experiences.length > 0) {
            experiencesHTML = user.experiences.map((exp, index) => {
                const isLast = index === user.experiences.length - 1;
                const isActive = !exp.endDate || exp.endDate.toUpperCase() === 'PRESENT';
                
                let coursesHTML = '';
                if (exp.courseInfo) {
                    let courses = exp.courseInfo.includes(';') ? exp.courseInfo.split(';') : exp.courseInfo.split(',');
                    let tags = courses.map(c => {
                        let text = c.includes('@#') ? c.split('@#')[0].trim() : c.trim();
                        return isActive ? `<span class="bg-[#222] text-white text-[10px] px-2 py-1 border border-white/10">${text}</span>` : `<span class="text-gray-500 text-[11px] mr-3">${text}</span>`;
                    }).join('');
                    coursesHTML = `<div class="mt-3 flex flex-wrap gap-2">${tags}</div>`;
                }

                return `
                    <div class="relative pl-8 mb-10">
                        ${!isLast ? `<div class="absolute left-[3px] top-4 bottom-[-40px] w-[1px] ${isActive ? 'bg-primary/30' : 'bg-white/10'}"></div>` : ''}
                        <div class="absolute left-0 top-1.5 w-2 h-2 ${isActive ? 'bg-primary' : 'bg-gray-500'}"></div>
                        <h3 class="text-lg font-bold text-white leading-tight">${exp.organizationName}</h3>
                        <p class="${isActive ? 'text-primary' : 'text-gray-400'} font-mono text-[10px] tracking-widest uppercase mt-1">
                            ${exp.positionTitle} <span class="text-gray-600 mx-1">|</span> ${exp.startDate} - ${exp.endDate || 'PRESENT'}
                        </p>
                        ${coursesHTML}
                    </div>`;
            }).join('');
        }

        // --- RENDER  ---
        modalBody.innerHTML = `
            <div class="grid grid-cols-1 lg:grid-cols-[1fr_2fr] gap-12 mt-4">
                <div class="space-y-6">
                    <div class="p-1 border border-white/10 bg-[#111] shadow-2xl">
                        <img src="${avatarUrl}" class="w-full aspect-square object-cover grayscale" />
                    </div>
                    
                    <div id="modal-actions-container">
                        ${actionButtonsHTML}
                    </div>

                    <div class="bg-[#111] border-y border-r border-white/5 p-6 mt-6 relative overflow-hidden">
                        <div class="absolute left-0 top-0 bottom-0 w-[3px] bg-primary"></div>
                        <h4 class="text-primary font-mono text-[10px] tracking-[0.2em] uppercase mb-6 font-bold">TACTICAL DATA</h4>
                        <div class="space-y-5">
                            <div><p class="text-[9px] font-bold text-gray-500 uppercase mb-1">PHONE</p><p class="text-white font-mono text-xs tracking-widest">${user.phone || 'CLASSIFIED'}</p></div>
                            <div><p class="text-[9px] font-bold text-gray-500 uppercase mb-1">ADDRESS</p><p class="text-white font-mono text-xs tracking-wide">${user.address || 'UNKNOWN'}</p></div>
                            <div><p class="text-[9px] font-bold text-gray-500 uppercase mb-1">EMAIL</p><p class="text-white font-mono text-xs break-all">${user.email || 'ENCRYPTED'}</p></div>
                        </div>
                    </div>
                </div>
                <div>
                    <h2 class="text-3xl font-bold text-white mb-8 pb-4 border-b border-white/10">Experience</h2>
                    <div class="space-y-2">${experiencesHTML}</div>
                    <div class="mt-10 pt-6 border-t border-white/5">
                         <h4 class="text-gray-500 font-mono text-[10px] uppercase tracking-widest mb-4">Briefing Notes</h4>
                         <p class="text-gray-400 text-sm leading-relaxed font-mono">${user.description || 'No additional logs found.'}</p>
                    </div>
                </div>
            </div>`;

    } catch (error) { 
        modalBody.innerHTML = `<p class="text-red-500 font-mono text-center mt-20">CONNECTION TERMINATED. ${error.message}</p>`; 
    }
}

function closeProfileModal() {
    const modalContent = document.getElementById('profile-modal-content');
    if (modalContent) modalContent.classList.add('translate-x-full');
    setTimeout(() => document.getElementById('profile-modal').classList.add('hidden'), 300);
}
// ================================================================
//  PORTFOLIO / VERIFIED CV — Proof of Work Center
//  Sections: §1 Hero  §2 Stats  §3 Skills  §4 Timeline
// ================================================================

async function loadPortfolioData() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');
    if (!savedUserStr || !token) return;

    const currentUser = JSON.parse(savedUserStr);
    if (!document.getElementById('portfolio-name')) return; // guard: page not in DOM yet

    try {
        const [userRes, rankRes] = await Promise.all([
            fetch(`${API_BASE_URL}/team/members/${currentUser.id}`, {
                headers: { 'Authorization': `Bearer ${token}` }
            }),
            fetch(`${API_BASE_URL}/ranking`).catch(() => null)
        ]);

        if (!userRes.ok) {
            if (userRes.status === 401 || userRes.status === 403) logout();
            return;
        }

        const user = await userRes.json();

        // Match current user in public ranking list (by name)
        let userRank = 0, userScore = 0;
        if (rankRes && rankRes.ok) {
            const rankList = await rankRes.json();
            const idx = Array.isArray(rankList)
                ? rankList.findIndex(r => r.name === user.name)
                : -1;
            if (idx >= 0) { userRank = idx + 1; userScore = rankList[idx].point || 0; }
        }

        _pfRenderHero(user);
        _pfRenderStats(user, userRank, userScore);
        _pfRenderSkills(user.experiences || []);
        _pfRenderTimeline(user.experiences || []);
        _pfRenderContributions(user);

    } catch (err) {
        console.error("[PORTFOLIO]", err);
        ['pf-skills-certified','pf-skills-declared'].forEach(id => {
            const el = document.getElementById(id);
            if (el && el.querySelector('.italic')?.textContent === 'Loading...')
                el.innerHTML = '<span class="font-mono text-[10px] text-on-surface-variant italic">—</span>';
        });
    }
}

/* §1 — populate hero: avatar, name, role tag, UID, clearance, bio */
function _pfRenderHero(user) {
    const get = id => document.getElementById(id);
    const set = (id, v) => { const el = get(id); if (el) el.textContent = v; };

    const defaultAvt = `https://ui-avatars.com/api/?background=222&color=fff&name=${encodeURIComponent(user.name || 'Operative')}`;
    const img = get('portfolio-avatar');
    if (img) img.src = user.avatar || defaultAvt;

    set('portfolio-name',          user.name || 'UNKNOWN OPERATIVE');
    set('portfolio-uid',           String(user.id || 0).padStart(6, '0'));
    set('portfolio-bio',           user.description || 'No briefing notes on file.');

    // Role tag colour: admins get .tag-tertiary (purple), everyone else .tag-primary (green)
    const roleTag = get('portfolio-role-tag');
    if (roleTag) {
        const isAdmin = user.role === 'ADMIN' || user.role === 'SUPER ADMIN';
        roleTag.textContent = user.role || 'MEMBER';
        roleTag.className   = (isAdmin ? 'tag-tertiary' : 'tag-primary') +
                              ' px-2.5 py-0.5 text-[10px] font-mono uppercase tracking-widest shrink-0';
    }

    set('portfolio-clearance-line', `ROLE: ${user.role || 'MEMBER'}`);
}

/* §2 — populate stat cards */
function _pfRenderStats(user, rank, score) {
    const set = (id, v) => { const el = document.getElementById(id); if (el) el.textContent = v; };

    set('pf-xp',       score > 0 ? score.toLocaleString() : '0');
    set('pf-stat-rank', rank > 0 ? `#${rank}` : '#--');
    set('pf-projects',  (user.experiences || []).length);
    // Workshops = experiences that recorded a courseInfo (training/course data present)
    set('pf-workshops', (user.experiences || []).filter(e => e.courseInfo && e.courseInfo.trim()).length);
}

/* §3 — build verified skill tag pills */
function _pfRenderSkills(experiences) {
    const certEl = document.getElementById('pf-skills-certified');
    const declEl = document.getElementById('pf-skills-declared');
    if (!certEl || !declEl) return;

    const certified = new Set();
    const declared  = new Set();

    experiences.forEach(exp => {
        if (!exp.courseInfo) return;
        const parts = exp.courseInfo.includes(';')
            ? exp.courseInfo.split(';')
            : exp.courseInfo.split(',');
        parts.forEach(p => {
            p = p.trim();
            if (!p) return;
            p.includes('@#')
                ? certified.add(p.split('@#')[0].trim())
                : declared.add(p);
        });
    });

    // Verified icon used inside every pill
    const checkIcon = `<span class="material-symbols-outlined" ` +
        `style="font-size:10px;font-variation-settings:'FILL' 1;line-height:1">check_circle</span>`;

    certEl.innerHTML = certified.size
        ? [...certified].map(n =>
            `<span class="pf-skill-tag tag-primary">${checkIcon}${n}</span>`
          ).join('')
        : '<span class="font-mono text-[10px] text-on-surface-variant italic">No system-certified skills on record.</span>';

    const altClass = ['tag-secondary', 'tag-tertiary'];
    declEl.innerHTML = declared.size
        ? [...declared].map((n, i) =>
            `<span class="pf-skill-tag ${altClass[i % 2]}">${checkIcon}${n}</span>`
          ).join('')
        : '<span class="font-mono text-[10px] text-on-surface-variant italic">No declared skills on record.</span>';
}

/* §4 — build proof-of-work vertical timeline */
function _pfRenderTimeline(experiences) {
    const container = document.getElementById('pf-timeline-container');
    if (!container) return;

    if (!experiences.length) {
        container.innerHTML = `<div class="pf-timeline-line"></div>
            <p class="font-mono text-[10px] text-on-surface-variant italic pl-2">No proof-of-work records on file.</p>`;
        return;
    }

    // Active deployments first, then newest by startDate
    const sorted = [...experiences].sort((a, b) => {
        const aLive = !a.endDate || a.endDate.toUpperCase() === 'PRESENT';
        const bLive = !b.endDate || b.endDate.toUpperCase() === 'PRESENT';
        if (aLive && !bLive) return -1;
        if (!aLive && bLive) return 1;
        return (b.startDate || '').localeCompare(a.startDate || '');
    });

    const nodes = sorted.map(exp => {
        const isLive = !exp.endDate || exp.endDate.toUpperCase() === 'PRESENT';

        const statusTag = isLive
            ? `<span class="tag-primary px-2 py-0.5 text-[8px] font-mono">ACTIVE</span>`
            : `<span class="tag-secondary px-2 py-0.5 text-[8px] font-mono">COMPLETED</span>`;

        // Skill chips (max 5) — certified chips show a verified icon
        let skillsHTML = '';
        if (exp.courseInfo) {
            const parts = (exp.courseInfo.includes(';')
                ? exp.courseInfo.split(';')
                : exp.courseInfo.split(','))
                .filter(p => p.trim()).slice(0, 5);

            const chips = parts.map(p => {
                const isCert = p.includes('@#');
                const name   = (isCert ? p.split('@#')[0] : p).trim();
                const icon   = isCert
                    ? `<span class="material-symbols-outlined" style="font-size:9px;font-variation-settings:'FILL' 1;line-height:1">verified</span>`
                    : '';
                return name
                    ? `<span class="${isCert ? 'tag-tertiary' : 'tag-secondary'} px-2 py-0.5 text-[8px] font-mono inline-flex items-center gap-1">${icon}${name}</span>`
                    : '';
            }).join('');
            if (chips) skillsHTML = `<div class="flex flex-wrap gap-1.5 mt-3">${chips}</div>`;
        }

        return `
        <div class="pf-timeline-node">
          <div class="pf-timeline-dot ${isLive ? '' : 'inactive'}"></div>
          <div class="glass-panel pf-timeline-card">
            <div class="flex items-start justify-between gap-3 flex-wrap">
              <div class="space-y-1 min-w-0">
                <div class="flex items-center gap-2 flex-wrap">
                  ${statusTag}
                  <span class="font-mono text-[9px] text-on-surface-variant tracking-widest">
                    ${exp.startDate || '???'} — ${exp.endDate || 'PRESENT'}
                  </span>
                </div>
                <h3 class="font-headline text-base font-bold text-white leading-tight">
                  ${exp.organizationName || 'Unnamed Organisation'}
                </h3>
                <p class="font-mono text-[10px] ${isLive ? 'text-primary' : 'text-on-surface-variant'} uppercase tracking-widest">
                  ${exp.positionTitle || 'OPERATIVE'}
                </p>
              </div>
              <button onclick="showToast('Project details coming soon!', 'success')"
                class="shrink-0 text-[9px] font-mono text-on-surface-variant border border-outline-variant/30 px-3 py-1.5 hover:border-primary hover:text-primary transition-all uppercase tracking-widest flex items-center gap-1.5">
                <span class="material-symbols-outlined" style="font-size:12px;line-height:1">open_in_new</span>
                VIEW
              </button>
            </div>
            ${skillsHTML}
          </div>
        </div>`;
    }).join('');

    container.innerHTML = `<div class="pf-timeline-line"></div>${nodes}`;
}

/* §5 — Project contribution analytics (UC005) */
async function _pfRenderContributions(user) {
    const container = document.getElementById('pf-contributions-container');
    if (!container) return;
    try {
        const response = await fetch(`${API_BASE_URL}/projects`);
        if (!response.ok) throw new Error();
        const projects = await response.json();
        const mine = projects.filter(p =>
            (p.members || []).some(m =>
                m.memberName === user.name || String(m.memberId) === String(user.id)
            )
        );
        if (!mine.length) {
            container.innerHTML = `<span class="font-mono text-[10px] text-on-surface-variant italic">No project contributions on record.</span>`;
            return;
        }
        container.innerHTML = mine.map(p => {
            const myMember = (p.members || []).find(m => m.memberName === user.name);
            const total = p.totalTasks || 0;
            const done = p.completedTasks || 0;
            const pct = total ? Math.round((done / total) * 100) : 0;
            return `
                <div class="bg-surface-container-low border border-outline-variant/20 p-4 hover:border-primary/30 transition-all">
                    <div class="flex items-center justify-between mb-3">
                        <div>
                            <h4 class="text-white font-bold text-sm font-headline">${p.name}</h4>
                            <p class="font-mono text-[10px] text-on-surface-variant uppercase mt-0.5">${myMember ? myMember.role : 'Contributor'}</p>
                        </div>
                        <span class="font-mono text-xs text-primary font-bold">${pct}%</span>
                    </div>
                    <div class="skill-bar">
                        <div class="skill-bar-fill verified" style="width:${pct}%" data-w="${pct}"></div>
                    </div>
                    <p class="font-mono text-[9px] text-on-surface-variant mt-2">${done} / ${total} tasks completed</p>
                </div>`;
        }).join('');
    } catch {
        container.innerHTML = `<span class="font-mono text-[10px] text-on-surface-variant italic">Contribution data unavailable.</span>`;
    }
}

function generateVerifiedCV() {
    const portfolio = document.getElementById('page-portfolio');
    if (!portfolio) { showToast('Navigate to your portfolio first', 'error'); return; }

    const styles = Array.from(document.querySelectorAll('link[rel="stylesheet"]'))
        .map(l => `<link rel="stylesheet" href="${l.href}"/>`)
        .join('');

    const tailwindConfig = document.querySelector('script[src*="tailwindcss"]')?.outerHTML || '';
    const configScript = document.querySelector('script:not([src])')?.outerHTML || '';
    const gFonts = Array.from(document.querySelectorAll('link[href*="fonts.googleapis"]'))
        .map(l => l.outerHTML).join('');

    const html = `<!DOCTYPE html>
<html class="dark">
<head>
<meta charset="utf-8"/>
<title>CyberTitans — Verified CV</title>
${tailwindConfig}
${configScript}
${gFonts}
${styles}
</head>
<body class="bg-[#0e0e0e] text-white font-body p-8">
${portfolio.innerHTML}
<script>
window.onload = function() {
    document.querySelectorAll('.skill-bar-fill[data-w]').forEach(function(el) {
        el.style.width = el.dataset.w + '%';
    });
    setTimeout(function() { window.print(); }, 600);
};
<\/script>
</body>
</html>`;

    const blob = new Blob([html], { type: 'text/html;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    window.open(url, '_blank');
    showToast('CV export opened — print or save as PDF', 'success');
}

async function deleteSelfAccount() {
    const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');
    if (!savedUserStr || !token) return showToast('Session expired.', 'error');

    const currentUser = JSON.parse(savedUserStr);
    const input = document.getElementById('delete-confirm-input');
    const typed = input ? input.value.trim() : '';

    if (typed !== currentUser.username) {
        showToast('Username does not match. Deletion cancelled.', 'error');
        return;
    }

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${currentUser.id}`, {
            method: 'DELETE',
            headers: { 'Authorization': `Bearer ${token}`, 'Content-Type': 'application/json' }
        });

        if (response.ok) {
            closeModal('delete-account-modal');
            showToast('Account deleted. Logging out...', 'success');
            setTimeout(() => logout(), 1500);
        } else {
            const err = await response.json().catch(() => ({}));
            showToast(`Error: ${err.message || 'Could not delete account.'}`, 'error');
        }
    } catch {
        showToast('Server error. Please try again.', 'error');
    }
}

async function adminDeleteUser(userId) {
    const confirmation = confirm("⚠️ WARNING: You are sure you want to expel this member from the system?");
    if (!confirmation) return;

    const token = sessionStorage.getItem('cyber_token') || localStorage.getItem('cyber_token');

    try {
        const response = await fetch(`${API_BASE_URL}/team/members/${userId}`, {
            method: 'DELETE',
            headers: { 
                'Authorization': `Bearer ${token}`,
                'Content-Type': 'application/json'
            }
        });

        if (response.ok) {
            showToast("User deleted successfully!", "success");
            closeProfileModal(); 
            buildTeam(); 
        } else {
            const errorData = await response.json();
            showToast(`ERROR: ${errorData.message || 'Access denied.'}`, "error");
        }
    } catch (error) {
        showToast("SERVER ERROR: Cannot connect to the server.", "error");
    }
}