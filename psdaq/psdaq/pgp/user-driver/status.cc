#include "pgpdriver.h"

int main()
{
    AxisG2Device dev("0000:af:00.0");
    dev.status();
}

