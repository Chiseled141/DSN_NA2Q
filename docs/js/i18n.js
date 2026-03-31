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
    'hero.subtitle'       : '50,000-episode comparison of NA²Q and HiT-MAC on directional sensor network coverage tasks.',
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
    'stats.badge'         : '50k Episodes · Scenario 1',
    'stats.mean'          : 'Mean Coverage',
    'stats.best'          : 'Best Episode',
    'stats.std'           : 'Std Deviation',
    'stats.conv'          : 'Ep. to 50% Cov.',
    'stats.stable'        : 'Final Stability',
    'badge.well'          : 'Well Optimized',
    'badge.warn'          : 'Params Not Tuned',
    'badge.warn2'         : 'Hyperparams Not Tuned',
    /* Algo compare */
    'na2q.desc'           : 'Trained for <strong>50,000 episodes</strong> with tuned hyperparameters. The attention mechanism enables efficient credit assignment and stable convergence in cooperative coverage tasks.',
    'hitmac.desc'         : 'Also trained for <strong>50,000 episodes</strong>, but default hyperparameters were used throughout, leading to suboptimal performance. Poor results reflect configuration, not the algorithm.',
    /* Coverage chart */
    'coverage.title'      : 'Coverage Rate',
    'coverage.badge'      : '100-ep avg',
    'coverage.note'       : '<strong>NA²Q (xanh)</strong> ramps from ~30% to a stable <strong>55–65%</strong> within 5,000 episodes and sustains it — demonstrating effective cooperative coordination. <strong>HiT-MAC (đỏ)</strong> plateaus at ~20–25% with no meaningful improvement, a direct consequence of untuned hyperparameters. The ~35-point gap highlights the value of NA²Q\'s neighbourhood attention mechanism.',
    /* Data note */
    'data.note'           : '<strong>Note:</strong> Coverage chart shows 100-episode moving average. HiT-MAC data converted from steps to episodes (100 steps = 1 episode).',
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
    'hero.subtitle'       : 'So sánh 50.000 tập của NA²Q và HiT-MAC trên các bài toán bao phủ mạng cảm biến định hướng.',
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
    'stats.badge'         : '50k Tập · Kịch bản 1',
    'stats.mean'          : 'Bao phủ trung bình',
    'stats.best'          : 'Tập tốt nhất',
    'stats.std'           : 'Độ lệch chuẩn',
    'stats.conv'          : 'Tập đạt 50% bao phủ',
    'stats.stable'        : 'Độ ổn định cuối',
    'badge.well'          : 'Tối ưu tốt',
    'badge.warn'          : 'Chưa tinh chỉnh tham số',
    'badge.warn2'         : 'Chưa tinh chỉnh tham số',
    /* Algo compare */
    'na2q.desc'           : 'Được huấn luyện <strong>50.000 tập</strong> với các siêu tham số được tinh chỉnh. Cơ chế chú ý cho phép gán tín hiệu thưởng hiệu quả và hội tụ ổn định trong các bài toán phối hợp bao phủ.',
    'hitmac.desc'         : 'Cũng được huấn luyện <strong>50.000 tập</strong>, nhưng dùng tham số mặc định xuyên suốt, dẫn đến hiệu suất kém. Kết quả yếu phản ánh cấu hình, không phải thuật toán.',
    /* Coverage chart */
    'coverage.title'      : 'Tỷ lệ bao phủ',
    'coverage.badge'      : 'Trung bình 100 tập',
    'coverage.note'       : '<strong>NA²Q (xanh)</strong> tăng từ ~30% lên mức ổn định <strong>55–65%</strong> trong 5.000 tập đầu và duy trì — thể hiện khả năng phối hợp hiệu quả. <strong>HiT-MAC (đỏ)</strong> dừng ở ~20–25% mà không cải thiện, hậu quả trực tiếp của tham số chưa được tinh chỉnh. Khoảng cách ~35 điểm phần trăm cho thấy giá trị của cơ chế chú ý lân cận trong NA²Q.',
    /* Data note */
    'data.note'           : '<strong>Lưu ý:</strong> Biểu đồ bao phủ hiển thị trung bình trượt 100 tập. Dữ liệu HiT-MAC được chuyển đổi từ bước sang tập (100 bước = 1 tập).',
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
