// https://stackoverflow.com/questions/63446295/parse-comma-separated-ints-int-ranges-in-c

#include "range.hh"

#include <string>
#include <vector>
#include <regex>

using namespace Pds;

void Pds::getRange(const std::string& input, std::vector<int>& data)
{
    const std::regex re{ R"(((\d+)-(\d+))|(\d+))" };
    std::smatch sm{};

    // Search all occureences of integers OR ranges
    for (std::string s{ input }; std::regex_search(s, sm, re); s = sm.suffix()) {

        // We found something. Was it a range?
        if (sm[1].str().length())

            // Yes, range, add all values within to the vector
            for (int i{ std::stoi(sm[2]) }; i <= std::stoi(sm[3]); ++i)
                data.push_back(i);
        else
            // No, no range, just a plain integer value. Add it to the vector
            data.push_back(std::stoi(sm[0]));
    }
}
