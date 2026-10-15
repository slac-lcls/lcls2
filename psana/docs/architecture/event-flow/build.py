#!/usr/bin/env python3
"""Validate source-backed workflow model and build a portable offline HTML file."""
import argparse
import html
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]

def checked_model():
    model = json.loads((HERE / 'workflow-model.json').read_text(encoding='utf-8'))
    assert model['schema_version'] == 2, 'unsupported model schema'
    sha = model['baseline']['commit']
    assert re.fullmatch(r'[0-9a-f]{40}', sha), 'baseline must be a full commit'
    nodes = {n['id']: n for n in model['nodes']}
    assert len(nodes) == len(model['nodes']), 'duplicate node IDs'
    cache = {}
    def resolve_source(src):
        path = src['path']
        assert not path.startswith('/') and '..' not in Path(path).parts
        if path not in cache:
            cache[path] = subprocess.check_output(['git', 'show', sha + ':' + path], cwd=ROOT).decode().splitlines()
        lines = cache[path]
        start, end = 0, len(lines)
        if 'scope' in src:
            scopes = [i for i, line in enumerate(lines) if line == src['scope']]
            assert len(scopes) == 1, 'ambiguous source scope: ' + str(src)
            start = scopes[0] + 1
            end = next((i for i in range(start, len(lines))
                        if re.match(r'^(?:class |cdef class |def )', lines[i])), len(lines))
        hits = [i + 1 for i in range(start, end) if src['needle'] in lines[i]]
        assert hits, 'source anchor not found: ' + str(src)
        if 'scope' in src:
            assert len(hits) == 1, 'ambiguous scoped source anchor: ' + str(src)
        src['line'] = hits[0]
        src['url'] = '{}/blob/{}/{}#L{}'.format(model['baseline']['repo'], sha, path, hits[0])

    def check_calls(items):
        assert items, 'empty call tree'
        for item in items:
            assert item['label'] and item['note'], 'calls need names and timing/context'
            if 'source' in item:
                resolve_source(item['source'])
            else:
                assert item.get('kind') in ('user', 'library', 'handoff'), 'unanchored repository call'
            if item.get('children'):
                check_calls(item['children'])

    for n in model['nodes']:
        assert re.fullmatch(r'[a-z][a-z0-9-]*', n['id'])
        for key in ('title','tag','role','serial','summary','inputs','outputs','flow','ownership','sources'):
            assert n[key], (n['id'], key)
        for src in n['sources']:
            resolve_source(src)
        assert n['call_paths'], 'component needs a call path: ' + n['id']
        for tree in n['call_paths']:
            assert tree['title'] and tree['when']
            check_calls(tree['roots'])
    ids = set()
    for edge in model['edges']:
        assert edge['id'] not in ids, 'duplicate edge'
        ids.add(edge['id'])
        assert edge['source'] in nodes and edge['target'] in nodes
        assert edge['kind'] in model['kinds']
        assert edge['views'] and all(v in model['views'] for v in edge['views'])
        for view in edge['views']:
            positions = model['views'][view]['positions']
            assert edge['source'] in positions and edge['target'] in positions
    for view in model['views'].values():
        for id, point in view['positions'].items():
            assert id in nodes and len(point) == 2
            assert 0 <= point[0] <= view['size'][0]-200
            assert 0 <= point[1] <= view['size'][1]-84
        assert all(id in ids for id in view['routes'])
    assert all(step['node'] in nodes for step in model['tour'])
    assert {s['node'] for s in model['tour']} == set(nodes), 'tour must cover every primary component'
    return model

def fallback_calls(items):
    out = ['<ul class="call-tree">']
    for item in items:
        label = '<code>' + html.escape(item['label']) + '</code>'
        if 'source' in item:
            label = '<a href="{}">{}</a>'.format(html.escape(item['source']['url']), label)
        out.append('<li><div class="call-entry">{}<span class="call-note">{}</span></div>'.format(label, html.escape(item['note'])))
        if item.get('children'):
            out.append(fallback_calls(item['children']))
        out.append('</li>')
    return ''.join(out) + '</ul>'

def fallback(model):
    out = ['<h2>Component reference</h2>']
    for n in model['nodes']:
        out += ['<section id="reference-{}"><h3>{}</h3><p>{}</p><p><strong>Role:</strong> {}</p>'.format(n['id'],html.escape(n['title']),html.escape(n['summary']),html.escape(n['role']))]
        for key in ('inputs','outputs','flow'):
            out.append('<h4>{}</h4><ul>{}</ul>'.format(key.title(),''.join('<li>'+html.escape(x)+'</li>' for x in n[key])))
        for tree in n['call_paths']:
            out.append('<h4>{}</h4><p>{}</p>{}'.format(html.escape(tree['title']), html.escape(tree['when']), fallback_calls(tree['roots'])))
        out.append('<p><strong>Ownership:</strong> '+html.escape(n['ownership'])+'</p>')
        out.append('<ul>'+''.join('<li><a href="{}">{} · {}:{}</a></li>'.format(html.escape(s['url']),html.escape(s['symbol']),html.escape(s['path']),s['line']) for s in n['sources'])+'</ul></section>')
    return ''.join(out)

def build():
    model = checked_model()
    template = (HERE/'template.html').read_text(encoding='utf-8')
    replacements = {'{{TITLE}}':html.escape(model['title']), '{{STYLE}}':(HERE/'viewer.css').read_text(), '{{SCRIPT}}':(HERE/'viewer.js').read_text(), '{{MODEL}}':json.dumps(model,ensure_ascii=False).replace('<','\\u003c'), '{{FALLBACK}}':fallback(model), '{{LICENSE}}':html.escape((ROOT/'LICENSE.md').read_text())}
    for key,value in replacements.items():
        assert key in template
        template=template.replace(key,value)
    assert not re.search(r'\{\{[A-Z]+\}\}',template)
    assert not re.search(r'<(?:script|link|img)[^>]+(?:src|href)=["\']https?://',template), 'external asset'
    return template

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check',action='store_true',help='validate model/source and require committed HTML to match')
    args=parser.parse_args()
    result=build()
    dest=HERE/'index.html'
    if args.check:
        assert dest.read_text(encoding='utf-8') == result, 'HTML is stale; run build.py'
        print('PASS: model, source anchors, routes, offline assets and deterministic generated HTML')
    else:
        dest.write_text(result,encoding='utf-8')
        print(dest)
