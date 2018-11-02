from psana.psexp.node import PacketFooter

view = bytearray()
pf = PacketFooter(3)
for i, msg in enumerate([b'hello', b'how are you', b'my name is mars']):
    view.extend(msg)
    pf.set_size(i, memoryview(msg).shape[0])

view.extend(pf.footer)

pf2 = PacketFooter(view=view)
print(pf2.n_packets)
views = pf2.split_packets()

for v in views:
    print(bytearray(v))


