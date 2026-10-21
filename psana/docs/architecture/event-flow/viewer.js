/* Original model-driven psana viewer. Content comes only from workflow-model.json. */
(() => {
  'use strict';
  const offlineDocument = '<!doctype html>\n' + document.documentElement.outerHTML;
  const model = JSON.parse(document.getElementById('workflow-model').textContent);
  const nodes = new Map(model.nodes.map(n => [n.id, n]));
  const $ = id => document.getElementById(id);
  const NS = 'http://www.w3.org/2000/svg';
  const colors = {
    call: '#1761ad',
    mpi: '#5860b4',
    data: '#367966',
    io: '#955700',
    cuda: '#aa4075',
    ownership: '#6e7780'
  };
  let view = 'overview',
    selected = 'setup',
    tourStep = 0,
    serial = false;

  function el(tag, text, attrs = {}) {
    const e = document.createElement(tag);
    if (text !== null) e.textContent = text;
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    return e;
  }

  function svg(tag, attrs = {}, text = null) {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== null) e.textContent = text;
    return e;
  }

  function text(id, value) {
    $(id).textContent = value;
  }

  function list(id, items) {
    $(id).replaceChildren(...items.map(x => el('li', x)));
  }

  function url(v, id) {
    return '#/' + v + '/' + id + (serial ? '?serial=1' : '');
  }

  function navigate(v, id, tour = false) {
    location.hash = url(v, tour ? 'tour/' + id : id);
  }

  function sourceLink(s) {
    const li = el('li', null);
    li.append(el('a', s.symbol, {
      href: s.url,
      target: '_blank',
      rel: 'noopener'
    }), el('span', s.path + ':' + s.line, {
      'class': 'source-path'
    }));
    return li;
  }
  text('title', model.title);
  text('subtitle', model.subtitle);
  text('scope', model.scope);
  $('baseline').replaceChildren(el('span', 'Baseline '), el('a', model.baseline.commit.slice(0, 12), {
    href: model.baseline.repo + '/commit/' + model.baseline.commit
  }), el('span', ' · ' + model.baseline.branch + ' · Reviewed ' + model.baseline.inspected +
    ' · ' + model.baseline.status + ' · Source links are pinned; your checkout may differ.'));
  $('docs-home').href = model.documentation_url;
  for (const id of Object.keys(model.views)) {
    const b = el('button', id === 'cpu' ? 'CPU path' : id === 'gpu' ? 'GPU path' : 'Overview', {
      'type': 'button',
      'data-view': id,
      'aria-pressed': id === view
    });
    b.addEventListener('click', () => navigate(id, selected in model.views[id].positions ?
      selected : 'setup'));
    $('view-buttons').append(b);
  }
  for (const [kind, label] of Object.entries(model.kinds)) {
    const item = el('span', null);
    const swatch = el('i', null, {
      'class': kind,
      'aria-hidden': 'true'
    });
    swatch.style.borderColor = colors[kind];
    item.append(swatch, el('span', label));
    $('legend').append(item);
  }
  text('serial-note', model.serial_note);
  $('serial').addEventListener('change', () => {
    serial = $('serial').checked;
    navigate(view, selected);
  });
  $('toggle-connections').addEventListener('click', () => {
    const open = $('connections').hidden;
    $('connections').hidden = !open;
    $('toggle-connections').setAttribute('aria-expanded', open);
    text('toggle-connections', open ? 'Hide connections' : 'Show connections');
  });
  $('download').addEventListener('click', () => {
    const blob = new Blob([offlineDocument], {
      type: 'text/html;charset=utf-8'
    });
    const a = el('a', null, {
      href: URL.createObjectURL(blob),
      download: 'psana-event-flow.html'
    });
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });

  function boundary(point, toward) {
    const [x, y] = point, cx = x + 100, cy = y + 42, dx = toward[0] - cx, dy = toward[1] - cy;
    const scale = 1 / Math.max(Math.abs(dx) / 100, Math.abs(dy) / 42);
    return [cx + dx * scale, cy + dy * scale];
  }

  function center(p) {
    return [p[0] + 100, p[1] + 42];
  }

  function points(edge, layout) {
    const a = layout.positions[edge.source],
      b = layout.positions[edge.target];
    const mid = layout.routes[edge.id] || [];
    if (mid.length) return [boundary(a, mid[0]), ...mid, boundary(b, mid[mid.length - 1])];
    const ac = center(a),
      bc = center(b);
    if (ac[0] === bc[0] || ac[1] === bc[1]) return [boundary(a, bc), boundary(b, ac)];
    const my = (ac[1] + bc[1]) / 2;
    const bends = [
      [ac[0], my],
      [bc[0], my]
    ];
    return [boundary(a, bends[0]), ...bends, boundary(b, bends[1])];
  }

  function drawMap() {
    const layout = model.views[view],
      root = svg('svg', {
        viewBox: '0 0 ' + layout.size.join(' '),
        'role': 'img',
        'aria-labelledby': 'map-title map-note'
      });
    const defs = svg('defs');
    for (const [k, color] of Object.entries(colors)) {
      const marker = svg('marker', {
        id: 'arrow-' + k,
        viewBox: '0 0 10 10',
        refX: 9,
        refY: 5,
        markerWidth: 7,
        markerHeight: 7,
        orient: 'auto-start-reverse'
      });
      marker.append(svg('path', {
        d: 'M 0 0 L 10 5 L 0 10 z',
        fill: color
      }));
      defs.append(marker);
    }
    root.append(defs);
    const active = model.edges.filter(e => e.views.includes(view));
    active.forEach((edge, index) => {
      const ps = points(edge, layout),
        path = svg('path', {
          d: ps.map((p, i) => (i ? 'L' : 'M') + p.join(',')).join(' '),
          class: 'edge ' + edge.kind,
          stroke: colors[edge.kind],
          'marker-end': 'url(#arrow-' + edge.kind + ')',
          'data-edge': edge.id
        });
      path.append(svg('title', {}, (index + 1) + '. ' + edge.label));
      root.append(path);
      const a = ps[Math.floor((ps.length - 1) / 2)],
        b = ps[Math.ceil((ps.length - 1) / 2)];
      let x = (a[0] + b[0]) / 2,
        y = (a[1] + b[1]) / 2;
      if (ps.length === 2) {
        x = (ps[0][0] + ps[1][0]) / 2;
        y = (ps[0][1] + ps[1][1]) / 2;
      }
      root.append(svg('text', {
        x: x + 5,
        y: y - 6,
        class: 'edge-number'
      }, String(index + 1)));
    });
    Object.entries(layout.positions).forEach(([id, [x, y]]) => {
      const n = nodes.get(id),
        link = svg('a', {
          href: url(view, id),
          class: 'node' + (id === selected ? ' active' : ''),
          'data-node': id,
          'aria-label': n.title + ' — ' + (serial ? n.serial : n.role),
          'aria-current': id === selected ? 'true' : 'false'
        });
      link.append(svg('rect', {
        x,
        y,
        width: 200,
        height: 84,
        rx: 10
      }));
      link.append(svg('text', {
        x: x + 14,
        y: y + 21,
        class: 'node-index'
      }, String(model.nodes.indexOf(n) + 1).padStart(2, '0')));
      link.append(svg('text', {
        x: x + 14,
        y: y + 45,
        class: 'node-title'
      }, n.map_title));
      link.append(svg('text', {
        x: x + 14,
        y: y + 67,
        class: 'node-tag'
      }, n.tag));
      root.append(link);
    });
    $('map').replaceChildren(root);
    text('map-title', layout.title);
    text('map-note', layout.note);
    const table = el('table', null, {
        class: 'connections-table'
      }),
      head = el('tr', null);
    ['# / kind', 'Components', 'Meaning'].forEach(x => head.append(el('th', x, {
      scope: 'col'
    })));
    const thead = el('thead', null);
    thead.append(head);
    table.append(thead);
    const tbody = el('tbody', null);
    active.forEach((e, i) => {
      const row = el('tr', null);
      row.append(el('td', (i + 1) + ' · ' + model.kinds[e.kind]), el('td', nodes.get(e.source)
        .title + ' → ' + nodes.get(e.target).title), el('td', e.label));
      tbody.append(row);
    });
    table.append(tbody);
    $('connections').replaceChildren(table);
    $('component-list').replaceChildren(...Object.keys(layout.positions).map(id => el('a', nodes
      .get(id).title, {
        href: url(view, id),
        'aria-current': id === selected ? 'true' : 'false'
      })));
  }

  function callTree(items) {
    const tree = el('ul', null, {class: 'call-tree'});
    for (const item of items) {
      const li = el('li', null), entry = el('div', null, {class: 'call-entry'});
      const code = el('code', item.label);
      if (item.source) {
        const link = el('a', null, {
          href: item.source.url, target: '_blank', rel: 'noopener',
          title: item.source.path + ':' + item.source.line
        });
        link.append(code);
        entry.append(link);
      } else entry.append(code);
      entry.append(el('span', item.note, {class: 'call-note'}));
      li.append(entry);
      if (item.children && item.children.length) li.append(callTree(item.children));
      tree.append(li);
    }
    return tree;
  }

  function detail() {
    const n = nodes.get(selected);
    text('detail-title', n.title);
    text('detail-tag', n.tag);
    text('detail-summary', n.summary);
    text('detail-role', (serial ? 'Serial: ' : 'MPI: ') + (serial ? n.serial : n.role));
    list('detail-inputs', n.inputs);
    list('detail-outputs', n.outputs);
    list('detail-flow', n.flow);
    $('detail-call-paths').replaceChildren(...n.call_paths.map(path => {
      const section = el('details', null, {class: 'call-path', open: ''});
      section.append(el('summary', path.title), el('p', path.when), callTree(path.roots));
      return section;
    }));
    text('detail-owner', n.ownership);
    $('detail-sources').replaceChildren(...n.sources.map(sourceLink));
    $('detail-notes').replaceChildren(...n.notes.map(note => el('p', note, {
      class: 'notice'
    })));
    $('permalink').href = url(view, selected);
    $('detail').dataset.node = selected;
  }

  function showTour() {
    const steps = view === 'cpu' ? model.tour.filter(t => t.node in model.views.cpu.positions) :
      model.tour;
    tourStep = Math.max(0, Math.min(tourStep, steps.length - 1));
    const step = steps[tourStep];
    text('tour-title', step.title);
    text('tour-text', step.text);
    text('tour-count', (tourStep + 1) + ' / ' + steps.length);
    $('tour-prev').disabled = tourStep === 0;
    $('tour-next').disabled = tourStep === steps.length - 1;
    return steps;
  }
  $('tour-prev').addEventListener('click', () => navigate(view, tourStep - 1, true));
  $('tour-next').addEventListener('click', () => navigate(view, tourStep + 1, true));

  function route() {
    const hash = location.hash;
    if (hash && !hash.startsWith('#/')) return;
    const [path, query] = hash.slice(2).split('?'), parts = (path || 'overview/setup').split('/');
    view = model.views[parts[0]] ? parts[0] : 'overview';
    serial = new URLSearchParams(query || '').get('serial') === '1';
    $('serial').checked = serial;
    $('serial-note').hidden = !serial;
    const requested = parts[1];
    if (requested === 'tour') {
      tourStep = Number(parts[2]) || 0;
      const steps = showTour();
      selected = steps[tourStep].node;
      if (!(selected in model.views[view].positions)) view = 'gpu';
    } else {
      selected = nodes.has(requested) ? requested : 'setup';
      if (!(selected in model.views[view].positions)) view = 'gpu';
      const steps = view === 'cpu' ? model.tour.filter(t => t.node in model.views.cpu.positions) :
        model.tour;
      tourStep = Math.max(0, steps.findIndex(t => t.node === selected));
    }
    drawMap();
    detail();
    showTour();
    document.querySelectorAll('[data-view]').forEach(b => b.setAttribute('aria-pressed', b.dataset
      .view === view));
    document.body.dataset.view = view;
    document.body.dataset.ready = 'true';
  }
  for (const [term, description] of model.granularities) $('granularities').append(el('dt', term),
    el('dd', description));
  for (const [title, chain, release] of model.lifetimes) {
    const item = el('div', null, {
      class: 'lifetime'
    });
    item.append(el('strong', title), el('p', chain, {
      class: 'chain'
    }), el('p', release));
    $('lifetimes').append(item);
  }
  text('transition-title', model.transitions.title);
  text('transition-intro', model.transitions.intro);
  for (const type of ['cpu', 'gpu'])
    for (const step of model.transitions[type]) {
      const item = el('li', null);
      item.append(el('strong', step.label), el('p', step.text));
      $('transition-' + type).append(item);
    }
  list('transition-notes', model.transitions.notes);
  $('transition-issue').href = model.baseline.repo + '/issues/153';
  for (const issue of model.issues) {
    const item = el('article', null, {
      class: 'issue'
    });
    item.append(el('div', '#' + issue.number + ' · ' + issue.state, {
      class: 'eyebrow'
    }), el('a', issue.title, {
      href: model.baseline.repo + '/issues/' + issue.number
    }), el('p', issue.note));
    $('issues').append(item);
  }
  $('attribution').append(el('span', 'Reference approach: '), el('a', model.attribution.name, {
    href: model.attribution.url
  }), el('span', ' · '), el('a', 'Inspected source', {
    href: model.attribution.source
  }));
  text('attribution-note', model.attribution.note);
  $('interactive').hidden = false;
  window.addEventListener('hashchange', route);
  route();
  if (!document.body.dataset.ready) {
    location.hash = '#/overview/setup';
    route();
  }
})();
