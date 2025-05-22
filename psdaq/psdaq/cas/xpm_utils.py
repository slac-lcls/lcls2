import socket
from psdaq.cas.pvedit import *

def atcaIp(v):
    return '10.0.{:}.{:}'.format((v>>12)&0xf,100+((v>>8)&0xf))

def hostName(v):
    ip = '172.21.{:d}.{:d}'.format((v>>8)&0xff,(v>>0)&0xff)
    try:
        name = socket.gethostbyaddr(ip)[0].split('.')[0].split('-')[-1]
    except:
        name = '{:x}'.format(v)
    return name

def nameLinkXpm(v):
    return ('XPM:{:}'.format((v>>16)&0xff), atcaIp(v))

def nameLinkDti(v):
    return ('DTI', atcaIp(v))

def nameLinkDrp(v):
    return ('TDetSim', hostName(v))

def nameLinkHsd(v):
    return ('HSD', '{:}.{:02X}'.format(hostName(v),(v>>16)&0xff))

def nameLinkTDet(v):
    return ('TDetSim', hostName(v))

def nameLinkWave8(v):
    return ('Wave8', hostName(v))

def nameLinkOpal(v):
    return ('Opal', hostName(v))

def nameLinkTimeTool(v):
    return ('TimeTool', hostName(v))

def nameLinkEpixQuad(v):
    return ('EpixQuad', hostName(v))

def nameLinkEpixHR2x2(v):
    return ('EpixHR2x2', hostName(v))

def nameLinkEpix100(v):
    return ('Epix100', hostName(v))

def nameLinkPiranha4(v):
    return ('Piranha4', hostName(v))

def nameLinkEpixM320(v):
    return ('ePixM320', hostName(v))

def nameLinkEpixUHR(v):
    return ('epixUHR', hostName(v))

def nameLinkHREncoder(v):
    return ('HREncoder', hostName(v))

timDevType = {}
timDevType['xpm']       = 0xff
timDevType['dti']       = 0xfe
timDevType['drp']       = 0xfd
timDevType['hsd']       = 0xfc
timDevType['tdet']      = 0xfb
timDevType['wave8']     = 0xfa
timDevType['opal']      = 0xf9
timDevType['timetool']  = 0xf8
timDevType['epixquad']  = 0xf7
timDevType['epixhr2x2'] = 0xf6
timDevType['epix100']   = 0xf5
timDevType['piranha4']  = 0xf4
timDevType['epixm320']  = 0xf3
timDevType['epixUHR']   = 0xf2
timDevType['hrencoder'] = 0xf1


linkType = {}
linkType[0xff] = nameLinkXpm
linkType[0xfe] = nameLinkDti
linkType[0xfd] = nameLinkDrp
linkType[0xfc] = nameLinkHsd
linkType[0xfb] = nameLinkTDet
linkType[0xfa] = nameLinkWave8
linkType[0xf9] = nameLinkOpal
linkType[0xf8] = nameLinkTimeTool
linkType[0xf7] = nameLinkEpixQuad
linkType[0xf6] = nameLinkEpixHR2x2
linkType[0xf5] = nameLinkEpix100
linkType[0xf4] = nameLinkPiranha4
linkType[0xf3] = nameLinkEpixM320
linkType[0xf2] = nameLinkEpixUHR
linkType[0xf1] = nameLinkHREncoder

def xpmLinkId(value):
    itype = (value>>24)&0xff
    names = None
    if itype in linkType:
        names = linkType[itype](value)
    elif value==0x10080:
        names = ('Fanout/','Loopback')
    else:
        names = ('undef','{:x}'.format(value))
    return names

def timTxId(timDevTypeStr):
    tdt = timDevType[timDevTypeStr]
    if tdt<timDevType['hsd']:
        info = socket.getaddrinfo(socket.gethostname(),None,0,0,socket.IPPROTO_UDP)
        for a in info:
            ip = a[4][0].split('.')
            if ip[0]=='172' and ip[1]=='21':
                break
        return (tdt<<24) | (int(ip[2])<<8) | int(ip[3])
    return 0
