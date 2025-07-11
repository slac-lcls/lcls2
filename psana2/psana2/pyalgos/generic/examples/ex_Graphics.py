#!/usr/bin/env python

#import psana.pyalgos.generic.Graphics as gr
from psana2.pyalgos.generic.Graphics import *

logger = logging.getLogger('ex_Graphics')

def test01():
    """imshow"""
    img = random_standard(shape=(40,60), mu=200, sigma=25)
    fig, axim = fig_img_axes()
    move_fig(fig, x0=50, y0=20)
    imsh = imshow(axim, img, amp_range=None, extent=None,\
           interpolation='nearest', aspect='auto', origin='upper',\
           orientation='horizontal', cmap='jet')


def test02():
    """ hist
    """
    mu, sigma = 200, 25
    arr = random_standard((500,), mu, sigma)
    #fig = figure(figsize=(6,5), title='Test hist', dpi=80, facecolor='w', edgecolor='w', frameon=True, move=(100,10))
    #axhi = add_axes(fig, axwin=(0.10, 0.08, 0.85, 0.88))
    fig, axhi = fig_img_axes()
    move_fig(fig, x0=50, y0=20)
    his = hist(axhi, arr, bins=100, amp_range=(mu-6*sigma,mu+6*sigma), weights=None, color=None, log=False)


def test03():
    """ Update image in the event loop
    """
    #fig = figure(figsize=(6,5), title='Test hist', dpi=80, facecolor='w', edgecolor='w', frameon=True, move=(100,10))
    #axim = add_axes(fig, axwin=(0.10, 0.08, 0.85, 0.88))
    fig, axim = fig_img_axes()
    move_fig(fig, x0=50, y0=20)
    imsh = None
    for i in range(10):
       print('Event %3d' % i)
       img = random_standard((1000,1000), mu=200, sigma=25)
       #axim.cla()
       set_win_title(fig, 'Event %d' % i)

       if imsh is None:
           imsh = imshow(axim, img, amp_range=None, extent=None,\
                  interpolation='nearest', aspect='auto', origin='upper',\
                  orientation='horizontal', cmap='jet')
       else:
           imsh.set_data(img)
       show(mode=1)  # !!!!!!!!!!
       #draw_fig(fig) # !!!!!!!!!!


def test04():
    """ Update histogram in the event loop
    """
    mu, sigma = 200, 25
    #fig = figure(figsize=(6,5), title='Test hist', dpi=80, facecolor='w', edgecolor='w', frameon=True, move=(100,10))
    #axhi = add_axes(fig, axwin=(0.10, 0.08, 0.85, 0.88))
    fig, axhi = fig_img_axes()

    for i in range(10):
       print('Event %3d' % i)
       arr = random_standard((500,), mu, sigma, dtype=np.float32)
       axhi.cla()
       set_win_title(fig, 'Event %d' % i)
       his = hist(axhi, arr, bins=100, amp_range=(mu-6*sigma,mu+6*sigma), weights=None, color=None, log=False)

       show(mode=1) # !!!!!!!!!!
       #draw(fig)    # !!!!!!!!!!


def test05():
    """ Update image with color bar in the event loop
    """
    fig, axim, axcb = fig_img_cbar_axes()
    move_fig(fig, x0=200, y0=0)
    imsh = None
    for i in range(20):
       print('Event %3d' % i)
       img = random_standard((1000,1000), mu=i, sigma=10)
       #axim.cla()
       set_win_title(fig, 'Event %d' % i)
       if imsh is None:
           imsh, cbar = imshow_cbar(fig, axim, axcb, img, amin=None, amax=None, extent=None,\
                                    interpolation='nearest', aspect='auto', origin='upper',\
                                    orientation='vertical', cmap='inferno')
       else:
           imsh.set_data(img)
           ave, rms = img.mean(), img.std()
           imsh.set_clim(ave-1*rms, ave+3*rms)
       show(mode=1)  # !!!!!!!!!!
       #draw_fig(fig) # !!!!!!!!!!


def test06():
    """ fig_img_cbar
    """
    img = random_standard((1000,1000), mu=100, sigma=10)
    fig, axim, axcb, imsh, cbar = fig_img_cbar(img)#, **kwa)
    move_fig(fig, x0=200, y0=0)


def test07():
    """ r-phi fig_img_proj_cbar
    """
    img = random_standard((200,200), mu=100, sigma=10)
    fig, axim, axcb, imsh, cbar = fig_img_proj_cbar(img)
    move_fig(fig, x0=200, y0=0)


def usage():
    msg = 'Usage: python psalgos/examples/ex-02-localextrema.py <test-number>'\
          '\n  where <test-number> ='\
          '\n  1 - single 2d random image'\
          '\n  2 - single random histgram'\
          '\n  3 - in loop 2d random images'\
          '\n  4 - in loop random histgrams'\
          '\n  5 - in loop 2d large random images'\
          '\n  6 - fig_img_cbar'\
          '\n  7 - r-phi projection fig_img_proj_cbar'
    print(msg)


def do_test():
    from time import time
    from psana2.pyalgos.generic.NDArrGenerators import random_standard; global random_standard

    if len(sys.argv)==1:
        print('Use command > python %s <test-number [1-7]>' % sys.argv[0])
        sys.exit ('Add <test-number> in command line...')

    tname = sys.argv[1] if len(sys.argv) > 1 else '1'
    print(50*'_', '\nTest %s' % tname)
    t0_sec=time()
    if   tname == '1': test01()
    elif tname == '2': test02()
    elif tname == '3': test03()
    elif tname == '4': test04()
    elif tname == '5': test05()
    elif tname == '6': test06()
    elif tname == '7': test07()
    else: usage(); sys.exit('Test %s is not implemented' % tname)
    msg = 'Test %s consumed time %.3f' % (tname, time()-t0_sec)
    show()
    sys.exit(msg)


if __name__ == "__main__":

    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.INFO)

    import sys; global sys
    do_test()
    sys.exit('End of test')

# EOF
