
from psana.graphqt.QWUtils import *

def test(tname):

    app = QApplication(sys.argv)

    if tname == '0':
        instrs = ['SXR', 'AMO', 'XPP', 'CXI', 'MEC']
        resp = select_item_from_popup_menu(instrs, title='Select INS', default='AMO')
        logger.debug('Selected: %s' % resp)

    elif tname == '1':
        list_of_cbox = [['VAR1', True], ['VAR2', False], ['VAR3', False], ['VAR4', False], ['VAR5', False]]
        resp = change_check_box_list_in_popup_menu(list_of_cbox, win_title='Select vars(s)')
        for (var,stat) in list_of_cbox: logger.debug('%s: %s' % (var, stat))
        logger.debug('resp: %s' % resp)

    elif tname == '2':
        dict_of_pars = {'checked':'radio1', 'list':['radio0', 'radio1', 'radio2']}
        resp = select_radio_button_in_popup_menu(dict_of_pars, win_title='Select vars(s)', do_confirm=True)
        for (k,v) in dict_of_pars.items(): logger.debug('%s: %s' % (k, v))
        logger.debug('resp: %s' % resp)

    elif tname == '3':
        parent=None; path0='./'; title='get_save_fname_through_dialog_box'
        resp = get_save_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        logger.debug('resp: %s' % resp)

    elif tname == '4':
        parent=None; path0='./'; title='get_open_fname_through_dialog_box'
        resp = get_open_fname_through_dialog_box(parent, path0, title, filter='*.txt')
        logger.debug('resp: %s' % resp)

    elif tname == '5':
        resp = confirm_dialog_box(parent=None, text='Confirm that you aware!', title='Acknowledge')
        logger.debug('resp: %s' % resp)

    elif tname == '6':
        resp = confirm_or_cancel_dialog_box(parent=None, text='Confirm or cancel', title='Confirm or cancel')
        logger.debug('resp: %s' % resp)

    elif tname == '7':
        from time import sleep
        resp = help_dialog_box(parent=None, text='Help message goes here', title='Help')
        logger.debug('resp: %s' % resp)
        sleep(3)
        del resp

    elif tname == '8':
        resp = select_color(colini=Qt.blue, parent=None)

    elif tname == '9':
        dict_of_cbox = {'VAR1':True, 'VAR2':False, 'VAR3':False, 'VAR4':False, 'VAR5':False}
        resp = change_check_box_dict_in_popup_menu(dict_of_cbox, win_title='Select vars(s)')
        for (var,stat) in dict_of_cbox.items(): logger.debug('%s: %s' % (var, stat))
        logger.debug('resp: %s' % resp)

    elif tname == '10':
        resp=edit_and_confirm_or_cancel_dialog_box(parent=None, text='Text confirm or cancel', title='Edit and confirm or cancel')
        logger.debug('resp=%s' % resp)

    else:
        logger.debug('Sorry, not-implemented test "%s"' % tname)

    del app


if __name__ == "__main__":
    import sys; global sys

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    logger.debug('%s\nTest %s' % (50*'_', tname))
    test(tname)
    sys.exit('End of test %s' % tname)

# EOF
