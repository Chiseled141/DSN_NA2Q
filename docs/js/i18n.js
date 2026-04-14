/**
 * i18n — English / Vietnamese language switcher
 */

const translations = {
  en: {
    /* Nav */
    'nav.dashboard'       : 'Training Dashboard',
    'nav.code'            : 'Code',
    /* Hero */
    'hero.badge'          : 'Live Training Data',
    'hero.h1'             : 'Training <span class="gradient-text">Dashboard</span>',
    'hero.subtitle'       : '8,395-episode comparison of NA²Q and HiT-MAC on directional sensor network coverage tasks.',
    /* Tabs */
    'tab.s1'              : 'Scenario 1 (5 Sensors)',
    'tab.s2'              : 'Scenario 2 (50 Sensors)',
    /* Stat cards */
    'stat.episodes'       : 'Total Episodes',
    'stat.sensors'        : 'Sensors',
    'stat.targets'        : 'Targets',
    'stat.fov'            : 'FoV',
    /* Stats comparison */
    'stats.section'       : '/STATS/COMPARISON',
    'stats.title'         : 'Performance Summary',
    'stats.badge'         : '8k Episodes · Scenario 1',
    'stats.mean'          : 'Mean Coverage',
    'stats.best'          : 'Best Episode',
    'stats.std'           : 'Std Deviation',
    'stats.conv'          : 'Ep. to 50% Cov.',
    'stats.stable'        : 'Final Stability',
    'badge.well'          : 'Well Optimized',
    'badge.warn'          : 'Params Not Tuned',
    'badge.warn2'         : 'Hyperparams Not Tuned',
    /* Algo compare */
    'na2q.desc'           : 'Compared over <strong>8,395 episodes</strong> with tuned hyperparameters. The attention mechanism enables efficient credit assignment via a GRU-based per-agent Q-network and GAM mixer.',
    'hitmac.desc'         : 'Compared over <strong>8,395 episodes</strong> of Phase 1 (executor) training using 6 A3C workers. Default hyperparameters — Phase 2 coordinator training is pending. Raw worker-average coverage shown here.',
    /* Coverage chart */
    'coverage.title'      : 'Coverage Rate',
    'coverage.badge'      : '100-ep avg',
    'coverage.note'       : '<strong>HiT-MAC (orange)</strong> Phase 1 executor trains across 16 parallel A3C workers. Coverage shown is a raw worker average — Phase 2 coordinator (which assigns targets) is not yet trained, so coverage reflects individual tracking behaviour only. <strong>NA²Q (blue)</strong> uses epsilon-greedy exploration with curriculum learning — noisier early on but capable of reaching 100% coverage per episode. Toggle <em>Smooth</em> above for a 50-episode moving average view.',
    /* Data note */
    'data.note'           : '<strong>Note:</strong> Coverage chart shows 100-episode moving average. HiT-MAC values are raw per-worker-episode from 16 parallel A3C workers; NA²Q values are raw per-episode.',
    /* Benchmark */
    'bench.title'         : '<span class="gradient-text">Performance</span> Benchmark',
    'bench.subtitle'      : 'Side-by-side replay comparison of trained agent behaviors',
    'bench.na2q'          : 'NA²Q Agent',
    'bench.hitmac'        : 'HiT-MAC Agent',
    'bench.play'          : 'Play',
    'bench.step'          : 'Step',
    'bench.reset'         : 'Reset',
    /* Footer */
    'footer.links'        : 'Quick Links',
    'footer.algos'        : 'Algorithms',
    'footer.resources'    : 'Resources',
    'footer.desc'         : 'Multi-Agent Reinforcement Learning for Directional Sensor Networks. Advancing cooperative AI for real-world surveillance and monitoring applications.',
    'footer.copy'         : '© 2026 Special Subject 2. Hanoi University. Multi-Agent Reinforcement Learning for Directional Sensor Networks.',
  },

  vi: {
    /* Nav */
    'nav.dashboard'       : 'Bảng Huấn Luyện',
    'nav.code'            : 'Mã Nguồn',
    /* Hero */
    'hero.badge'          : 'Dữ Liệu Huấn Luyện Trực Tiếp',
    'hero.h1'             : 'Bảng Điều Khiển <span class="gradient-text">Huấn Luyện</span>',
    'hero.subtitle'       : 'So sánh 8.395 tập của NA²Q và HiT-MAC trên các bài toán bao phủ mạng cảm biến định hướng.',
    /* Tabs */
    'tab.s1'              : 'Kịch bản 1 (5 Cảm biến)',
    'tab.s2'              : 'Kịch bản 2 (50 Cảm biến)',
    /* Stat cards */
    'stat.episodes'       : 'Tổng số tập',
    'stat.sensors'        : 'Cảm biến',
    'stat.targets'        : 'Mục tiêu',
    'stat.fov'            : 'Góc nhìn',
    /* Stats comparison */
    'stats.section'       : '/THỐNG KÊ/SO SÁNH',
    'stats.title'         : 'Tóm tắt Hiệu suất',
    'stats.badge'         : '8k Tập · Kịch bản 1',
    'stats.mean'          : 'Bao phủ trung bình',
    'stats.best'          : 'Tập tốt nhất',
    'stats.std'           : 'Độ lệch chuẩn',
    'stats.conv'          : 'Tập đạt 50% bao phủ',
    'stats.stable'        : 'Độ ổn định cuối',
    'badge.well'          : 'Tối ưu tốt',
    'badge.warn'          : 'Chưa tinh chỉnh tham số',
    'badge.warn2'         : 'Chưa tinh chỉnh tham số',
    /* Algo compare */
    'na2q.desc'           : 'So sánh qua <strong>8.395 tập</strong> với siêu tham số được tinh chỉnh. Cơ chế chú ý cho phép gán tín hiệu thưởng hiệu quả qua mạng Q theo từng agent (GRU) và bộ trộn GAM.',
    'hitmac.desc'         : 'So sánh qua <strong>8.395 tập</strong> huấn luyện Giai đoạn 1 (executor) với 6 worker A3C. Dùng tham số mặc định — huấn luyện Giai đoạn 2 (coordinator) đang chờ. Dữ liệu hiển thị là trung bình thô của các worker.',
    /* Coverage chart */
    'coverage.title'      : 'Tỷ lệ bao phủ',
    'coverage.badge'      : 'Trung bình 100 tập',
    'coverage.note'       : '<strong>HiT-MAC (cam)</strong> — Giai đoạn 1 huấn luyện executor qua 16 worker A3C song song. Dữ liệu là trung bình thô — Giai đoạn 2 (coordinator gán mục tiêu) chưa huấn luyện, nên bao phủ phản ánh hành vi theo dõi cá nhân. <strong>NA²Q (xanh)</strong> dùng epsilon-greedy và học theo giáo trình — biến động ban đầu nhưng có thể đạt 100% mỗi tập. Bật <em>Smooth</em> để xem trung bình trượt 50 tập.',
    /* Data note */
    'data.note'           : '<strong>Lưu ý:</strong> Biểu đồ bao phủ hiển thị trung bình trượt 100 tập. Giá trị HiT-MAC là dữ liệu thô từ 16 worker A3C song song; NA²Q là giá trị thô theo từng tập.',
    /* Benchmark */
    'bench.title'         : '<span class="gradient-text">So sánh</span> Hiệu suất',
    'bench.subtitle'      : 'Phát lại song song hành vi của các agent đã huấn luyện',
    'bench.na2q'          : 'Agent NA²Q',
    'bench.hitmac'        : 'Agent HiT-MAC',
    'bench.play'          : 'Phát',
    'bench.step'          : 'Bước',
    'bench.reset'         : 'Đặt lại',
    /* Footer */
    'footer.links'        : 'Liên kết nhanh',
    'footer.algos'        : 'Thuật toán',
    'footer.resources'    : 'Tài nguyên',
    'footer.desc'         : 'Học tăng cường đa tác nhân cho mạng cảm biến định hướng. Thúc đẩy AI hợp tác cho các ứng dụng giám sát thực tế.',
    'footer.copy'         : '© 2026 Chuyên đề 2. Đại học Hà Nội. Học tăng cường đa tác nhân cho mạng cảm biến định hướng.',
  }
};

let currentLang = localStorage.getItem('lang') || 'en';

function applyLang(lang) {
  currentLang = lang;
  localStorage.setItem('lang', lang);

  const t = translations[lang];

  // Update all data-i18n elements (textContent)
  document.querySelectorAll('[data-i18n]').forEach(el => {
    const key = el.dataset.i18n;
    if (t[key] !== undefined) el.textContent = t[key];
  });

  // Update all data-i18n-html elements (innerHTML)
  document.querySelectorAll('[data-i18n-html]').forEach(el => {
    const key = el.dataset.i18nHtml;
    if (t[key] !== undefined) el.innerHTML = t[key];
  });

  // Update toggle button label
  const btn = document.getElementById('lang-toggle');
  if (btn) btn.textContent = lang === 'en' ? 'VI' : 'EN';
}

function toggleLang() {
  applyLang(currentLang === 'en' ? 'vi' : 'en');
}

document.addEventListener('DOMContentLoaded', () => applyLang(currentLang));
