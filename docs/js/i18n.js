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
    'hero.subtitle'       : '4,028-episode comparison of NA²Q and HiT-MAC on directional sensor network coverage tasks.',
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
    'stats.badge'         : '4k Episodes · Scenario 1',
    'stats.mean'          : 'Mean Coverage',
    'stats.best'          : 'Best Episode',
    'stats.std'           : 'Std Deviation',
    'stats.conv'          : 'Ep. to 50% Cov.',
    'stats.stable'        : 'Final Stability',
    'badge.well'          : 'Well Optimized',
    'badge.warn'          : 'Params Not Tuned',
    'badge.warn2'         : 'Hyperparams Not Tuned',
    /* Algo compare */
    'na2q.desc'           : 'Compared over the first <strong>4,028 episodes</strong> with tuned hyperparameters. The attention mechanism enables efficient credit assignment, though convergence is noisy (±21%) across this window.',
    'hitmac.desc'         : 'Compared over <strong>4,028 equivalent episodes</strong> (each averaged from 100 A3C sub-episodes). Default hyperparameters were used, yet HiT-MAC converges faster and more stably than NA²Q over this range.',
    /* Coverage chart */
    'coverage.title'      : 'Coverage Rate',
    'coverage.badge'      : '100-ep avg',
    'coverage.note'       : '<strong>HiT-MAC (orange)</strong> reaches 50% coverage by episode 676 and stabilises around <strong>56–60%</strong> with very low variance (±2%) — a result of A3C\'s parallel exploration. <strong>NA²Q (blue)</strong> is slower to converge (50% by episode 2,994) and more volatile (±21%) over this 4,028-episode window, though its per-episode peak reaches 100%. Both algorithms are compared on equal footing: 4,028 episodes of 100 steps each.',
    /* Data note */
    'data.note'           : '<strong>Note:</strong> Coverage chart shows 100-episode moving average. HiT-MAC values are averaged over 100 A3C sub-episodes per point; NA²Q values are raw per-episode.',
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
    'hero.subtitle'       : 'So sánh 4.028 tập của NA²Q và HiT-MAC trên các bài toán bao phủ mạng cảm biến định hướng.',
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
    'stats.badge'         : '4k Tập · Kịch bản 1',
    'stats.mean'          : 'Bao phủ trung bình',
    'stats.best'          : 'Tập tốt nhất',
    'stats.std'           : 'Độ lệch chuẩn',
    'stats.conv'          : 'Tập đạt 50% bao phủ',
    'stats.stable'        : 'Độ ổn định cuối',
    'badge.well'          : 'Tối ưu tốt',
    'badge.warn'          : 'Chưa tinh chỉnh tham số',
    'badge.warn2'         : 'Chưa tinh chỉnh tham số',
    /* Algo compare */
    'na2q.desc'           : 'So sánh trong <strong>4.028 tập đầu tiên</strong> với siêu tham số được tinh chỉnh. Cơ chế chú ý cho phép gán tín hiệu thưởng hiệu quả, dù hội tụ còn biến động (±21%) trong khoảng này.',
    'hitmac.desc'         : 'So sánh qua <strong>4.028 tập tương đương</strong> (mỗi tập là trung bình của 100 tập con A3C). Dùng tham số mặc định, nhưng HiT-MAC hội tụ nhanh hơn và ổn định hơn NA²Q trong khoảng này.',
    /* Coverage chart */
    'coverage.title'      : 'Tỷ lệ bao phủ',
    'coverage.badge'      : 'Trung bình 100 tập',
    'coverage.note'       : '<strong>HiT-MAC (cam)</strong> đạt 50% bao phủ từ tập 676 và ổn định ở mức <strong>56–60%</strong> với phương sai rất thấp (±2%) — nhờ khám phá song song của A3C. <strong>NA²Q (xanh)</strong> hội tụ chậm hơn (đạt 50% ở tập 2.994) và biến động hơn (±21%) trong cửa sổ 4.028 tập này, dù đỉnh cao nhất đạt 100% theo tập. Cả hai thuật toán được so sánh trên cùng điều kiện: 4.028 tập, mỗi tập 100 bước.',
    /* Data note */
    'data.note'           : '<strong>Lưu ý:</strong> Biểu đồ bao phủ hiển thị trung bình trượt 100 tập. Giá trị HiT-MAC là trung bình của 100 tập con A3C mỗi điểm; NA²Q là giá trị thô theo từng tập.',
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
