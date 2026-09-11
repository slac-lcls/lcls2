#include "CubeResult.hh"

#include "psdaq/eb/src/CubeResultDgram.hh"
#include "psdaq/eb/src/WindowResultDgram.hh"

#include <stdio.h>

using namespace Pds::Eb;

std::vector<unsigned> Drp::CubeResult::add_bins(const ResultDgram& result) const 
{
    std::vector<unsigned> bins(0);
    if (m_resultType == Cube)
        bins.push_back(reinterpret_cast<const CubeResultDgram&>(result).binIndex());
    else {
        unsigned mask = reinterpret_cast<const WindowResultDgram&>(result).updateAdd();
        for(unsigned ib = __builtin_ffsl(mask); ib>0; ib = __builtin_ffsl(mask)) {
            bins.push_back(ib-1);
            mask &= ~(1<<(ib-1));
        }
    }
    return bins;
}

std::vector<unsigned> Drp::CubeResult::monitor_bins(const ResultDgram& result) const 
{
    std::vector<unsigned> bins(0);
    if (m_resultType == Cube)
        bins.push_back(reinterpret_cast<const CubeResultDgram&>(result).binIndex());
    else {
        unsigned mask = reinterpret_cast<const WindowResultDgram&>(result).updateMonitor();
        for(unsigned ib = __builtin_ffsl(mask); ib>0; ib = __builtin_ffsl(mask)) {
            bins.push_back(ib-1);
            mask &= ~(1<<(ib-1));
        }
    }
    return bins;
}

std::vector<unsigned> Drp::CubeResult::record_bins(const ResultDgram& result) const 
{
    std::vector<unsigned> bins(0);
    if (m_resultType == Cube)
        bins.push_back(reinterpret_cast<const CubeResultDgram&>(result).binIndex());
    else {
        unsigned mask = reinterpret_cast<const WindowResultDgram&>(result).updateRecord();
        for(unsigned ib = __builtin_ffsl(mask); ib>0; ib = __builtin_ffsl(mask)) {
            bins.push_back(ib-1);
            mask &= ~(1<<(ib-1));
        }
    }
    return bins;
}

std::vector<unsigned> Drp::CubeResult::flush_bins(const ResultDgram& result) const 
{
    std::vector<unsigned> bins(0);
    if (m_resultType == Cube)
        bins.push_back(reinterpret_cast<const CubeResultDgram&>(result).binIndex());
    else {
        unsigned mask = reinterpret_cast<const WindowResultDgram&>(result).flush();
        for(unsigned ib = __builtin_ffsl(mask); ib>0; ib = __builtin_ffsl(mask)) {
            bins.push_back(ib-1);
            mask &= ~(1<<(ib-1));
        }
    }
    return bins;
}

bool Drp::CubeResult::update_monitor(const ResultDgram& result) const
{
    if (m_resultType == Cube)
        return reinterpret_cast<const CubeResultDgram&>(result).updateMonitor();
    else
        return reinterpret_cast<const WindowResultDgram&>(result).updateMonitor()!=0;
    return false;
}

bool Drp::CubeResult::update_record(const ResultDgram& result) const
{
    if (m_resultType == Cube)
        return reinterpret_cast<const CubeResultDgram&>(result).updateRecord();
    else
        return reinterpret_cast<const WindowResultDgram&>(result).updateRecord()!=0;
    return false;
}

bool Drp::CubeResult::flush       (const ResultDgram& result) const
{
    if (m_resultType == Cube)
        return reinterpret_cast<const CubeResultDgram&>(result).flush();
    else
        return reinterpret_cast<const WindowResultDgram&>(result).flush()!=0;
    return false;
}
