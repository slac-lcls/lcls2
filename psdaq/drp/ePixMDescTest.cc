#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>


void dump(uint16_t* frame, unsigned elemRows, unsigned elemRowSize)
{
    for (unsigned row = 0; row < elemRows; ++row) {
        for (unsigned col = 0; col < elemRowSize; ++col) {
          printf("%4u ", frame[row * elemRowSize + col]);
        }
        printf("\n");
    }
}

int main(int argc, char* argv[])
{
    // Descramble the data

    // Reorder banks from
    // 18    19    20    21    22    23
    // 12    13    14    15    16    17
    //  6     7     8     9    10    11
    //  0     1     2     3     4     5
    //
    //                To
    //  3     7    11    15    19    23
    //  2     6    10    14    18    22
    //  1     5     9    13    17    21
    //  0     4     8    12    16    20

    const unsigned numAsics    = 1; /*4;*/                        printf("numAsics    = %u\n", numAsics   );
    const unsigned elemRows    = 12; /*192;*/              printf("elemRows    = %u\n", elemRows   );
    const unsigned elemRowSize = 24; /*384;*/              printf("elemRowSize = %u\n", elemRowSize);
    const unsigned asicSize    = elemRows * elemRowSize;   printf("asicSize    = %u\n", asicSize   );
    const unsigned numBanks    = 24;                       printf("numBanks    = %u\n", numBanks   );
    const unsigned bankRows    = 4;                        printf("bankRows    = %u\n", bankRows   );
    const unsigned bankCols    = 6;                        printf("bankCols    = %u\n", bankCols   );
    const unsigned bankHeight  = elemRows / bankRows;      printf("bankHeight  = %u\n", bankHeight );
    const unsigned bankWidth   = elemRowSize / bankCols;   printf("bankWidth   = %u\n", bankWidth  );
    const unsigned hw          = bankWidth / 2;            printf("hw          = %u\n", hw         );
    const unsigned hb          = bankHeight * hw;          printf("hb          = %u\n", hb         );

    std::vector< std::vector<uint16_t> > subframes(2 + numAsics);
    for (unsigned i = 0; i < numAsics; ++i) {
        subframes[2+i].resize(asicSize);
        for (unsigned r = 0; r < elemRows; ++r) {
            for (unsigned c = 0; c < elemRowSize; ++c) {
                unsigned j = elemRowSize * r + c;
                subframes[2+i][j] = elemRows * c + r;
            }
        }
    }
    printf("raw:\n");
    dump(subframes[2+0].data(), elemRows, elemRowSize);

    std::vector<uint16_t> aframe(numAsics * asicSize);
    uint16_t* frame = aframe.data();
    for (unsigned asic = 0; asic < numAsics; ++asic) {
      uint16_t* src = (uint16_t*)subframes[2 + asic].data();
      uint16_t* dst = &frame[asic * asicSize];
      for (unsigned bankRow = 0; bankRow < bankRows; ++bankRow) {
        for (unsigned row = 0; row < bankHeight; ++row) {
          // ASIC firmware bug: Rows are shifted up by one in ring buffer fashion
          unsigned rowFix = row == 0 ? bankHeight - 1 : row - 1; // Shift rows up by one for ASIC f/w bug
          for (unsigned bankCol = 0; bankCol < bankCols; ++bankCol) {
            unsigned bank = bankRows * bankCol + bankRow;        // Given (column, row), reorder banks
            for (unsigned col = 0; col < bankWidth; ++col) {
              //              (even cols w/ offset + row offset  + inc every 2 cols) * fill one pixel / bank + bank inc
              unsigned idx = (((col+1) % 2) * hb   + hw * rowFix + int(col / 2))     * numBanks              + bank;
              //printf("asic %u, bank %2u (%u, %u), row %u, col %u, idx %3u, src %3u\n", asic, bank, bankRow, bankCol, rowFix, col, idx, src[idx]);
              *dst++ = src[idx];
            }
          }
        }
      }
    }
    printf("\ndescrambled:\n");
    dump(frame, elemRows, elemRowSize);

  return 0;
}
