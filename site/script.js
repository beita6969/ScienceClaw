(function () {
  'use strict';
  var reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  var $ = function (s, r) { return (r || document).querySelector(s); };
  var $$ = function (s, r) { return Array.prototype.slice.call((r || document).querySelectorAll(s)); };

  /* ---------- navbar ---------- */
  var nav = $('#navbar'), toggle = $('#navToggle'), links = $('#navLinks');
  function onScroll() { nav.classList.toggle('scrolled', window.scrollY > 40); }
  window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
  toggle.addEventListener('click', function () {
    var open = links.classList.toggle('open');
    toggle.setAttribute('aria-expanded', open);
  });
  $$('a', links).forEach(function (a) { a.addEventListener('click', function () { links.classList.remove('open'); }); });

  /* ---------- hero network ---------- */
  (function () {
    var cv = $('#heroCanvas'); if (!cv) return;
    var ctx = cv.getContext('2d'), W = 0, H = 0, pts = [], raf = 0, visible = true;
    var colors = ['rgba(95,208,193,', 'rgba(233,196,106,', 'rgba(160,190,230,'];
    function size() {
      var r = cv.getBoundingClientRect(), dpr = Math.min(window.devicePixelRatio || 1, 2);
      W = r.width; H = r.height; cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      var n = Math.round(Math.min(90, Math.max(28, W * H / 16000)));
      pts = [];
      for (var i = 0; i < n; i++) pts.push({ x: Math.random() * W, y: Math.random() * H, vx: (Math.random() - .5) * .22, vy: (Math.random() - .5) * .22, r: Math.random() * 1.6 + .8, c: colors[i % 3] });
    }
    function draw() {
      ctx.clearRect(0, 0, W, H);
      var i, j, a, b, d, max = 130;
      for (i = 0; i < pts.length; i++) {
        a = pts[i];
        if (!reduce) { a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > W) a.vx *= -1; if (a.y < 0 || a.y > H) a.vy *= -1; }
        for (j = i + 1; j < pts.length; j++) {
          b = pts[j]; d = Math.hypot(a.x - b.x, a.y - b.y);
          if (d < max) { ctx.strokeStyle = 'rgba(95,208,193,' + (0.16 * (1 - d / max)) + ')'; ctx.lineWidth = .8; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
        }
      }
      for (i = 0; i < pts.length; i++) { a = pts[i]; ctx.fillStyle = a.c + '0.75)'; ctx.beginPath(); ctx.arc(a.x, a.y, a.r, 0, 6.2832); ctx.fill(); }
      if (!reduce && visible) raf = requestAnimationFrame(draw);
    }
    size(); draw();
    var t; window.addEventListener('resize', function () { clearTimeout(t); t = setTimeout(function () { size(); if (reduce) draw(); }, 150); });
    if ('IntersectionObserver' in window) new IntersectionObserver(function (e) {
      visible = e[0].isIntersecting; if (visible && !reduce) { cancelAnimationFrame(raf); draw(); }
    }).observe(cv);
  })();

  /* ---------- count-up ---------- */
  $$('[data-count]').forEach(function (el) {
    if (reduce) return;
    var to = parseFloat(el.dataset.count), dec = +el.dataset.dec || 0, pre = el.dataset.prefix || '', suf = el.dataset.suffix || '', t0 = null;
    el.textContent = pre + (0).toFixed(dec) + suf;
    function step(t) {
      if (t0 === null) t0 = t;
      var p = Math.min(1, (t - t0) / 1400), e = 1 - Math.pow(1 - p, 3);
      el.textContent = pre + (to * e).toFixed(dec) + suf;
      if (p < 1) requestAnimationFrame(step);
    }
    setTimeout(function () { requestAnimationFrame(step); }, 350);
  });

  /* ---------- reveal ---------- */
  var revealed = function (el) { el.classList.add('in'); };
  if ('IntersectionObserver' in window && !reduce) {
    var io = new IntersectionObserver(function (es) {
      es.forEach(function (e) { if (e.isIntersecting) { revealed(e.target); io.unobserve(e.target); } });
    }, { threshold: .08, rootMargin: '0px 0px -40px 0px' });
    $$('.reveal').forEach(function (el) { io.observe(el); });
  } else { $$('.reveal').forEach(revealed); }

  /* ---------- lightbox ---------- */
  var lb = $('#lightbox'), lbImg = $('img', lb);
  $$('.zoom').forEach(function (b) {
    b.addEventListener('click', function () { lbImg.src = b.dataset.src; lbImg.alt = $('img', b).alt; lb.hidden = false; });
  });
  lb.addEventListener('click', function () { lb.hidden = true; });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.hidden = true; });

  /* ---------- copy buttons ---------- */
  $$('.code.copy').forEach(function (box) {
    var btn = $('.copybtn', box), pre = $('pre', box);
    btn.addEventListener('click', function () {
      var text = pre.innerText;
      var done = function () { btn.textContent = 'Copied'; setTimeout(function () { btn.textContent = 'Copy'; }, 1400); };
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(text).then(done, function () {});
      else { var ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); try { document.execCommand('copy'); done(); } catch (e) {} ta.remove(); }
    });
  });

  /* ---------- ablation retention bars ---------- */
  (function () {
    var host = $('#retainChart'); if (!host) return;
    var rows = [
      ['Full system', 100, true], ['Committed separately (unlinked)', 60], ['w/o independent validator', 59], ['Operator only', 57],
      ['w/o source replay', 55], ['w/o scientific constraints', 46], ['Skill only', 40], ['w/o independent IID selection', 15]
    ];
    host.innerHTML = rows.map(function (r) {
      return '<div class="bar-row' + (r[2] ? ' full' : '') + '"><label>' + r[0] + '</label><div class="bar-track"><div class="bar-fill" data-w="' + r[1] + '"></div></div><b>' + r[1] + '%</b></div>';
    }).join('');
    var fill = function () { $$('.bar-fill', host).forEach(function (f) { f.style.width = f.dataset.w + '%'; }); };
    if ('IntersectionObserver' in window && !reduce) {
      var o = new IntersectionObserver(function (e) { if (e[0].isIntersecting) { fill(); o.disconnect(); } }, { threshold: .3 }); o.observe(host);
    } else fill();
  })();

  /* ---------- per-discipline gains ---------- */
  var SHORT = { 30: 'Agriculture & food', 31: 'Biology', 32: 'Biomedicine', 33: 'Built environment', 34: 'Chemistry', 35: 'Commerce & tourism', 36: 'Creative arts', 37: 'Earth sciences', 38: 'Economics', 39: 'Education', 40: 'Engineering', 41: 'Environment', 42: 'Health', 43: 'History & heritage', 44: 'Human society', 45: 'Indigenous studies', 46: 'Computing', 47: 'Language & culture', 48: 'Law', 49: 'Mathematics', 50: 'Philosophy', 51: 'Physics', 52: 'Psychology' };
  (function () {
    var host = $('#discChart'), D = window.DISC; if (!host || !D) return;
    var split = 'ood', sort = 'code', MAX = 25;
    function render() {
      var rows = D.slice().sort(function (a, b) { return sort === 'code' ? a.code - b.code : b[split] - a[split]; });
      host.innerHTML = rows.map(function (d) {
        return '<div class="drow" title="' + d.name + ' — ' + d.task + ' (' + d.metric + ')"><span class="dcode">' + d.code + '</span><div class="nm"><span>' + SHORT[d.code] + '</span><div class="tk"><i data-w="' + (d[split] / MAX * 100).toFixed(1) + '"></i></div></div><b>+' + d[split].toFixed(2) + '%</b></div>';
      }).join('');
      requestAnimationFrame(function () { requestAnimationFrame(function () { $$('.tk i', host).forEach(function (i) { i.style.width = i.dataset.w + '%'; }); }); });
    }
    function seg(id, set) {
      var g = $(id); $$('button', g).forEach(function (b) {
        b.addEventListener('click', function () { $$('button', g).forEach(function (x) { x.classList.toggle('on', x === b); }); set(b.dataset.v); render(); });
      });
    }
    seg('#splitSeg', function (v) { split = v; }); seg('#sortSeg', function (v) { sort = v; });
    render();
  })();

  /* ---------- discipline chips ---------- */
  (function () {
    var host = $('#chips'), D = window.DISC; if (!host || !D) return;
    host.innerHTML = D.map(function (d) {
      return '<span class="chip" title="' + d.task + ' · ' + d.metric + '"><i>FoR' + d.code + '</i>' + SHORT[d.code] + '</span>';
    }).join('');
  })();
})();
