"""
:py:class:`test_NDArrUtils` - tests of NDArrUtils
=================================================

"""

import sys

from psana2.pyalgos.generic.NDArrUtils import *

if __name__ == "__main__":

  def test_01():
    from psana2.pyalgos.generic.NDArrGenerators import random_standard

    print('%s\n%s\n' % (80*'_','Test method subtract_bkgd(...):'))
    shape1 = (32,185,388)
    winds = [(s, 10, 155, 20, 358) for s in (0,1)]
    data = random_standard(shape=shape1, mu=300, sigma=50)
    bkgd = random_standard(shape=shape1, mu=100, sigma=10)
    cdata = subtract_bkgd(data, bkgd, mask=None, winds=winds, pbits=0o377)


  def test_02():
    from psana2.pyalgos.generic.NDArrGenerators import random_standard
    shape1 = (32,185,388)
    data = random_standard(shape=shape1, mu=300, sigma=50)
    print(info_ndarr(data, 'test_02: info_ndarr', first=0, last=3))
    print(info_ndarr(shape1, 'test_02: info_ndarr'))


  def test_08():
import psana2.pyalgos.generic.Graphics as gg
    from psana2.pyalgos.generic.NDArrGenerators import random_standard
    from psana2.pyalgos.generic.NDArrUtils import reshape_to_2d

    print('%s\n%s\n' % (80*'_','Test method locxymax(nda, order, mode):'))
    #data = random_standard(shape=(32,185,388), mu=0, sigma=10)
    data = random_standard(shape=(2,185,388), mu=0, sigma=10)
    t0_sec = time()
    mask = locxymax(data, order=1, mode='clip')
    print('Consumed t = %10.6f sec' % (time()-t0_sec))

    if True:
      img = data if len(data.shape)==2 else reshape_to_2d(data)
      msk = mask if len(mask.shape)==2 else reshape_to_2d(mask)

      ave, rms = img.mean(), img.std()
      amin, amax = ave-2*rms, ave+2*rms
      gg.plotImageLarge(img, amp_range=(amin, amax), title='random')
      gg.plotImageLarge(msk, amp_range=(0, 1), title='mask loc max')
      gg.show()


  def test_mask_neighbors_2d(allnbrs=True):

    randexp = random_exponential(shape=(40,60), a0=1)
    fig  = gr.figure(figsize=(16,7), title='Random 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)
    img1 = mask # mask # randexp
    img2 = mask_nbrs # mask # randexp

    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2,  amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)


  def test_mask_neighbors_3d(allnbrs=True):

    #randexp = random_exponential(shape=(2,2,30,80), a0=1)
    randexp = random_exponential(shape=(2,30,80), a0=1)

    fig  = gr.figure(figsize=(16,7), title='Random > 2-d mask')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.40, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.452, 0.05, 0.01, 0.91))

    axim2 = gr.add_axes(fig, axwin=(0.55,  0.05, 0.40, 0.91))
    axcb2 = gr.add_axes(fig, axwin=(0.952, 0.05, 0.01, 0.91))

    mask = np.select((randexp>6,), (0,), default=1)
    mask_nbrs = mask_neighbors(mask, allnbrs)

    img1 = reshape_to_2d(mask)
    img2 = reshape_to_2d(mask_nbrs)

    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    imsh2, cbar2 = gr.imshow_cbar(fig, axim2, axcb2, img2, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)


  def test_mask_edges_2d(mrows=1, mcols=1):

    fig  = gr.figure(figsize=(8,7), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    mask = np.ones((20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = mask_out
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)


  def test_mask_edges_3d(mrows=1, mcols=1):

    fig  = gr.figure(figsize=(8,7), title='Mask edges 2-d')
    axim1 = gr.add_axes(fig, axwin=(0.05,  0.05, 0.87, 0.91))
    axcb1 = gr.add_axes(fig, axwin=(0.922, 0.05, 0.01, 0.91))

    #mask = np.ones((2,2,20,30))
    mask = np.ones((2,20,30))
    mask_out = mask_edges(mask, mrows, mcols)

    img1 = reshape_to_2d(mask_out)
    imsh1, cbar1 = gr.imshow_cbar(fig, axim1, axcb1, img1, amin=0, amax=10, orientation='vertical', cmap='jet')
    gr.show(mode=None)


  def do_test():

    from psana2.pyalgos.generic.NDArrGenerators import random_exponential; global random_exponential 
import psana2.pyalgos.generic.Graphics as gr; global gr

    print(80*'_')
    tname = sys.argv[1] if len(sys.argv)>1 else '1'
    if   tname == '1': test_mask_neighbors_2d(allnbrs = False)
    elif tname == '2': test_mask_neighbors_2d(allnbrs = True)
    elif tname == '3': test_mask_neighbors_3d(allnbrs = False)
    elif tname == '4': test_mask_neighbors_3d(allnbrs = True)
    elif tname == '5': test_mask_edges_2d(mrows=5, mcols=1)
    elif tname == '6': test_mask_edges_2d(mrows=0, mcols=5)
    elif tname == '7': test_mask_edges_3d(mrows=1, mcols=2)
    elif tname == '8': test_mask_edges_3d(mrows=5, mcols=0)
    elif tname =='12': test_02()
    else: sys.exit ('Not recognized test name: "%s"    Try tests 1-8' % tname)


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s: %(message)s',\
                        datefmt='%m-%d-%Y %H:%M:%S',\
                        level=logging.DEBUG)
                        #filename='example.log', filemode='w'
    do_test()
    sys.exit('\nEnd of test')

# EOF
