####!/usr/bin/env python
#----------------------------
"""
:py:class:`DCEmail` - class for Detector Calibration Store (DCS) project
========================================================================

Usage::

    # Import
    from PSCalib.DCEmail import send_text_email

    # Send message via e-mail
    send_text_email(msg='Text message',\
               subject='Text subject',\
               email_from='do-not-reply@slac.stanford.edu',\
               email_to='do-not-send@slac.stanford.edu')

See:
    * :py:class:`DCStore`
    * :py:class:`DCType`
    * :py:class:`DCRange`
    * :py:class:`DCVersion`
    * :py:class:`DCBase`
    * :py:class:`DCInterface`
    * :py:class:`DCUtils`
    * :py:class:`DCDetectorId`
    * :py:class:`DCConfigParameters`
    * :py:class:`DCFileName`
    * :py:class:`DCLogger`
    * :py:class:`DCMethods`
    * :py:class:`DCEmail`

This software was developed for the SIT project.
If you use all or part of it, please give an appropriate acknowledgment.

Author: Mikhail Dubrovin
"""
#------------------------------

import smtplib
from email.mime.text import MIMEText
#from PSCalib.DCLogger import log

#------------------------------

def send_text_email(msg=None, subject=None, email_from=None, email_to=None) :
    """Sends e-mail.
    """
    dmsg = MIMEText(msg)
    dmsg['Subject'] = str(subject)
    dmsg['From']    = email_from
    dmsg['To']      = email_to
    s = smtplib.SMTP('localhost')
    s.sendmail(email_from, [email_to], dmsg.as_string())
    s.quit()
    #log.debug('Message submitted:\n%s' % dmsg.as_string(), 'send_text_email')
    
#------------------------------

#class DCEmail() :
#    """
#    """
#    def __init__(self, env, src, calibdir=None) :
#        self._name = self.__class__.__name__
#        log.debug('c-tor', self._name)
#
#    def __del__(self) :
#        log.debug('d-tor', self._name)

#------------------------------

def test_send_text_email() :
    print 20*'_', '\n%s:' % sys._getframe().f_code.co_name
    log.setPrintBits(0377) 
    send_text_email(msg='Test message',\
               subject='Test subject',\
               email_from='no-reply@slac.stanford.edu',\
               email_to='dubrovin@slac.stanford.edu')

#------------------------------

def do_test() :
    tname = sys.argv[1] if len(sys.argv) > 1 else '0'
    print 50*'_', '\nTest %s:' % tname
    if   tname == '0' : test_send_text_email() # ; test_DCEmail()
    elif tname == '1' : test_send_text_email()
    else : print 'Not-recognized test: %s' % tname
    sys.exit('End of test %s' % tname)

#------------------------------

if __name__ == "__main__" :
    import sys; global sys
    do_test()

#------------------------------
