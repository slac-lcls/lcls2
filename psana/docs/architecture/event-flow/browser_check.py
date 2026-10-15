#!/usr/bin/env python3
"""Offline file:// browser checks; needs Playwright + Chromium, not psana."""
import argparse
import json
from pathlib import Path
from playwright.sync_api import sync_playwright, expect

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--artifacts',type=Path,default=Path('/tmp/psana-workflow-browser-check'))
args=parser.parse_args();args.artifacts.mkdir(parents=True,exist_ok=True)
model=json.loads((HERE/'workflow-model.json').read_text())
errors=[];network=[]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True)
    context=browser.new_context(viewport={'width':1440,'height':1050},offline=True,accept_downloads=True)
    context.on('request',lambda request:network.append(request.url) if request.url.startswith(('http:','https:')) else None)
    page=context.new_page();page.on('pageerror',lambda error:errors.append(str(error)))
    page.goto((HERE/'index.html').as_uri())
    page.wait_for_selector('body[data-ready="true"]')
    for view in model['views']:
        page.locator('button[data-view="'+view+'"]').click()
        expect(page.locator('body')).to_have_attribute('data-view',view)
        for id in model['views'][view]['positions']:
            page.locator('#component-list a[href="#/'+view+'/'+id+'"]').click()
            expect(page.locator('#detail')).to_have_attribute('data-node',id)
            node=next(n for n in model['nodes'] if n['id']==id)
            expect(page.locator('#detail-call-paths > details')).to_have_count(len(node['call_paths']))
            assert page.locator('#detail-call-paths .call-tree code').count()>0
            assert page.locator('#detail-sources a').count()>0
            for link in page.locator('#detail-sources a, #detail-call-paths a').all():
                assert '/blob/'+model['baseline']['commit']+'/' in link.get_attribute('href')
        page.screenshot(path=str(args.artifacts/(view+'.png')),full_page=True)
        assert not page.locator('.node').evaluate_all('nodes=>nodes.flatMap(n=>{const r=n.querySelector("rect").getBBox();return Array.from(n.querySelectorAll("text")).filter(t=>t.getBBox().x+t.getBBox().width>r.x+r.width-5).map(t=>t.textContent)})'), 'map label overflows component'
    page.goto((HERE/'index.html').as_uri()+'#/gpu/input')
    expect(page.locator('#detail')).to_have_attribute('data-node','input')
    calls=page.locator('#detail-call-paths')
    for name in ('KvikioGpuReader.__init__()', 'GpuEventManager._process_batch()',
                 'KvikioGpuReader._file_for_identity()', 'GpuXtcBatchPool.parse_groups()',
                 'GpuXtcBatchPool.parse_window()'):
        assert name in calls.inner_text()
    # A call opens its pinned source in another tab. Stub that external page so
    # the click itself can be tested without network access.
    context.route('https://github.com/**',lambda route:route.fulfill(status=200,body='Pinned source preview'))
    link=calls.locator('a').filter(has_text='KvikioGpuReader.__init__()')
    with page.expect_popup() as opened: link.click()
    popup=opened.value;popup.wait_for_load_state()
    assert popup.url==link.get_attribute('href')
    popup.close()
    # This one intercepted navigation is intentional; offline runtime must still
    # issue no external asset requests.
    assert network==[link.get_attribute('href').split('#')[0]],network
    network.clear()
    first_path=calls.locator('details').first
    first_path.locator('summary').focus();page.keyboard.press('Enter')
    expect(first_path.locator('.call-tree').first).not_to_be_visible()
    page.keyboard.press('Enter')
    expect(first_path.locator('.call-tree').first).to_be_visible()
    calls.screenshot(path=str(args.artifacts/'input-call-paths.png'))
    page.locator('#serial').check()
    assert 'serial=1' in page.url
    expect(page.locator('#serial-note')).to_be_visible()
    assert page.locator('#detail-role').inner_text().startswith('Serial:')
    page.reload();expect(page.locator('#serial')).to_be_checked()
    page.goto((HERE/'index.html').as_uri()+'#/gpu/tour/0')
    for i,step in enumerate(model['tour']):
        expect(page.locator('#detail')).to_have_attribute('data-node',step['node'])
        if i<len(model['tour'])-1:page.locator('#tour-next').click()
    expect(page.locator('#tour-next')).to_be_disabled()
    page.locator('#tour-prev').click()
    page.locator('#toggle-connections').click();expect(page.locator('#connections')).to_be_visible()
    assert page.locator('#connections tbody tr').count()==len([e for e in model['edges'] if 'gpu' in e['views']])
    # Keyboard activation of component anchors, independently of pointer clicks.
    page.locator('#component-list a').first.focus();page.keyboard.press('Enter')
    expect(page.locator('#detail')).to_have_attribute('data-node','setup')
    with page.expect_download() as pending:page.locator('#download').click()
    saved=args.artifacts/'downloaded.html';pending.value.save_as(str(saved))
    downloaded=context.new_page();downloaded.on('pageerror',lambda error:errors.append(str(error)))
    downloaded.goto(saved.as_uri()+'#/gpu/execute')
    downloaded.wait_for_selector('body[data-ready="true"]')
    expect(downloaded.locator('#detail')).to_have_attribute('data-node','execute')
    downloaded.locator('#tour-next').click()
    expect(downloaded.locator('#detail')).to_have_attribute('data-node','publish')
    downloaded.set_viewport_size({'width':390,'height':844})
    downloaded.goto(saved.as_uri()+'#/gpu/input')
    expect(downloaded.locator('#detail')).to_have_attribute('data-node','input')
    assert downloaded.evaluate('document.documentElement.scrollWidth <= innerWidth'), 'whole-page mobile horizontal overflow'
    downloaded.screenshot(path=str(args.artifacts/'mobile.png'),full_page=True)
    # The map deliberately scrolls horizontally on narrow screens; surrounding content must fit.
    assert downloaded.locator('.map-shell').evaluate('(e)=>e.scrollWidth>e.clientWidth')
    # Verify static references still exist if scripting is disabled.
    noscript=browser.new_context(java_script_enabled=False,offline=True)
    plain=noscript.new_page();plain.goto(saved.as_uri())
    assert plain.locator('noscript section').count()==len(model['nodes'])
    assert plain.locator('#reference-input .call-tree a').count()>0
    assert not errors,errors
    assert not network,network
    result={'status':'PASS','chromium':browser.version,'views':len(model['views']),'components':len(model['nodes']),'tour_steps':len(model['tour']),'external_requests':len(network),'javascript_errors':errors,'checks':['all component links','all component call trees','commit-pinned source links','call-source click (intercepted locally)','call-tree keyboard collapse/expand','direct hash navigation','serial toggle persists on reload','full event tour','connection table','keyboard activation','downloaded file remains interactive offline','mobile layout with deepest call trees','no-JavaScript call-tree fallback']}
    (args.artifacts/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    browser.close()
