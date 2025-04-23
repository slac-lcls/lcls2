from psdaq.seq.seq import *

def main():
    instrset = [Wait('1H',1),  
                ControlRequest([0]), 
                Wait('910kH',20000), 
                ControlRequest([1]), 
                Wait('910kH',40000), 
                Branch.conditional(3,0,5),
                Branch.unconditional(1)]
    print(f'instrset')
    for line,instr in enumerate(instrset):
        print(f'{line}: {instr}')
    print(f'---')

    newinstr = preproc(instrset)
    print(f'newinstr')
    for line,instr in enumerate(newinstr):
        print(f'{line}: {instr}')
    print(f'---')

if __name__ == "__main__":
    main()
