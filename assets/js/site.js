// Azul Labs site script: the mobile nav, reveals on scroll, and motion in the look of
// Brushtail's terminal UI. Pages are complete without it; it only adds classes that play
// CSS animations (assets/css/main.css, "Motion") from a visible state, and does nothing
// for readers who prefer reduced motion.
(() => {
    const nav = document.getElementById('site-nav');
    const toggle = nav.querySelector('.nav-toggle');

    toggle.addEventListener('click', () => {
        const open = nav.classList.toggle('open');
        toggle.setAttribute('aria-expanded', open);
    });

    nav.querySelectorAll('.nav-links a').forEach(link => {
        link.addEventListener('click', () => {
            nav.classList.remove('open');
            toggle.setAttribute('aria-expanded', 'false');
        });
    });

    const onScroll = () => nav.classList.toggle('scrolled', window.scrollY > 8);
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    if (!('IntersectionObserver' in window)) return;

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('in');
                observer.unobserve(entry.target);
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.service-card, .own-point, .hosting-point, .step, .about-content').forEach((el, i) => {
        el.classList.add('reveal');
        el.style.transitionDelay = `${(i % 3) * 80}ms`;
        observer.observe(el);
    });

    if (window.matchMedia && matchMedia('(prefers-reduced-motion: reduce)').matches) return;

    // Run fn once on each element as it comes into view.
    const whenSeen = (selector, fn, options) => {
        const io = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (!entry.isIntersecting) return;
                io.unobserve(entry.target);
                fn(entry.target);
            });
        }, options);
        document.querySelectorAll(selector).forEach(el => io.observe(el));
    };

    // The squares: section labels, the terminal's tiles and the about stats are laid in turn.
    whenSeen('.section-label, .eval-card:not([data-cli]) .eval-dots, .about-stats',
        el => el.classList.add('in'), { rootMargin: '0px 0px -12% 0px' });

    // One band of light across each heading's accent word, then it rests.
    const shine = el => {
        el.classList.add('shine');
        el.addEventListener('animationend', () => el.classList.remove('shine'), { once: true });
    };
    whenSeen('h1 em, h2 em', shine, { threshold: 1, rootMargin: '0px 0px -8% 0px' });

    // Terminal mocks marked data-cli play like the Brushtail CLI.
    const SPINNER = ['·', '✢', '✳︎', '✶', '✻', '✽', '✻', '✶', '✳︎', '✢'];
    const SPINNER_COLOURS = ['#5BA3D9', '#3B82C4', '#5BA3D9', '#D4A853'];
    const STEP_SECONDS = [0.75, 0.95, 0.6, 0.85];
    const CLI_CLASSES = ['cli-wait', 'cli-show', 'cli-active', 'cli-done'];

    const play = card => {
        const items = [];
        for (const el of card.querySelector('.eval-card-body').children) {
            if (el.matches('.eval-rows')) items.push(...el.querySelectorAll(':scope > li'));
            else items.push(el);
        }

        // When each line prints, and how long a step runs before the next line.
        let clock = 0.3;
        let steps = 0;
        const plan = items.map(el => {
            const step = el.tagName === 'LI';
            const icon = step && el.querySelector('.eval-ok');
            const score = step && el.querySelector('.eval-score');
            const item = {
                el, step, icon, score, at: clock,
                icon0: icon && icon.textContent, score0: score && score.textContent,
            };
            clock += step ? STEP_SECONDS[steps++ % STEP_SECONDS.length]
                : el.matches('.eval-cmd') ? 0.5 : 0.4;
            return item;
        });

        const settle = item => {
            if (!item.step) return;
            item.icon.textContent = item.icon0;
            item.icon.style.color = '';
            item.score.textContent = item.score0;
            item.el.classList.replace('cli-active', 'cli-done');
        };

        // Back to the page as written, whatever state the run is in.
        const finish = () => {
            plan.forEach(item => {
                if (item.step) {
                    item.icon.textContent = item.icon0;
                    item.icon.style.color = '';
                    item.score.textContent = item.score0;
                }
                item.el.classList.remove(...CLI_CLASSES);
            });
            card.removeAttribute('aria-busy');
        };

        try {
            card.setAttribute('aria-busy', 'true');
            plan.forEach(item => item.el.classList.add('cli-wait'));
            const dots = card.querySelector('.eval-dots');
            if (dots) dots.classList.add('in');
            const title = card.querySelector('.eval-card-title');
            if (title) shine(title);
        } catch (err) {
            finish();
            return;
        }

        const start = performance.now();
        let next = 0;
        let active = null;

        // A timer rather than animation frames: 20 ticks a second is enough for a spinner
        // at ten frames a second, and a timer keeps going (throttled) in a background tab.
        const timer = setInterval(() => {
            try {
                const t = (performance.now() - start) / 1000;
                while (next < plan.length && t >= plan[next].at) {
                    if (active) settle(active);
                    active = plan[next++];
                    active.el.classList.replace('cli-wait', 'cli-show');
                    if (active.step) active.el.classList.add('cli-active');
                }
                if (active && active.step) {
                    if (next === plan.length && t >= clock) {
                        settle(active);
                    } else {
                        active.icon.textContent = SPINNER[Math.floor(t * 10) % SPINNER.length];
                        active.icon.style.color = SPINNER_COLOURS[Math.floor(t * 2) % SPINNER_COLOURS.length];
                        active.score.textContent = `${(t - active.at).toFixed(1)}s`;
                    }
                }
                if (next === plan.length && t >= clock) {
                    clearInterval(timer);
                    if (active) settle(active);
                    setTimeout(finish, 1200);
                }
            } catch (err) {
                clearInterval(timer);
                finish();
            }
        }, 50);
    };

    whenSeen('.eval-card[data-cli]', play, { threshold: 0.35 });
})();
