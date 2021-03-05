

import psana.pyalgos.generic.Graphics as gr

from psana.detector.UtilsGraphics import arr_median_limits#(arr, nneg=1, npos=3)

def test_pedestas_difference_between_dark_runs(run1, run2):
    from psana.pyalgos.generic.NDArrUtils import info_ndarr
    from psana.pscalib.calib.MDBWebUtils import calib_constants_all_types
    #d = calib_constants_all_types("epixquad", exp="ueddaq02", run=108)
    dcc1 = calib_constants_all_types("epix_000001", exp="ueddaq02", run=93)
    dcc1 = calib_constants_all_types("epix10ka_000001", exp="ueddaq02", run=run1)
    dcc2 = calib_constants_all_types("epix10ka_000001", exp="ueddaq02", run=run2)

    print(50*'-')
    print('d.keys():',dcc1.keys())
    print(50*'-')

    peds1 = dcc1['pedestals'][0]
    meta1 = dcc1['pedestals'][1]
    print('meta1', meta1)
    print(info_ndarr(peds1, 'peds1: '))
    print(50*'-')

    peds2 = dcc2['pedestals'][0]
    meta2 = dcc2['pedestals'][1]
    print('meta2', meta2)
    print(info_ndarr(peds2, 'peds2: '))


    peds_diff = peds2 - peds1

    print(info_ndarr(peds_diff, 'pedestal difference between runs (%d - %d): ' % (run2, run1)))


    for ig in range(5):
      arr = peds_diff[ig,:]
      amplimits = arr_median_limits(arr, nneg=50, npos=50)

      fig, axhi, hi = gr.hist1d(arr, bins=None, amp_range=amplimits, weights=None, color=None, show_stat=True,
                              log=False, figsize=(6,5), axwin=(0.15, 0.12, 0.78, 0.80), title=None, 
                              xlabel=None, ylabel=None, titwin=None)
      #gr.move_fig(fig, x0=ig*100, y0=10)

      title = 'peds-diff-runs-%04d-%04d-igain-%d' % (run2, run1, ig)
      gr.set_win_title(fig, titwin=title)
      gr.add_title_labels_to_axes(axhi, title=title)#, xlabel=None, ylabel=None, fslab=14, fstit=20, color='k')
      gr.show()#mode='non-hold')
      gr.save_fig(fig, fname=title+'.png', verb=True)
      fig.clf()

    #gr.show()

#test_pedestas_difference_between_dark_runs(93, 95) # [3.2158203 2.8850098 3.348877  3.0458984 2.9389648...]
#test_pedestas_difference_between_dark_runs(83, 95) # [-173.06616 -166.23608 -175.99512 -163.26514 -176.58813...]
#test_pedestas_difference_between_dark_runs(95, 97) # ~0
#test_pedestas_difference_between_dark_runs(97, 99) # ~0
#test_pedestas_difference_between_dark_runs(99, 101) # ~0
#test_pedestas_difference_between_dark_runs(101, 106) # ~0
test_pedestas_difference_between_dark_runs(106, 108) # [-61.351807 -69.78003  -59.70508  -73.783936 -56.28296 ...]
#test_pedestas_difference_between_dark_runs(95, 1000) # # ~0 - the same run 95 in 1000 processed with different versions of alg
