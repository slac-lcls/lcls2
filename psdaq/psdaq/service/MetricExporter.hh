#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <prometheus/exposer.h>

namespace Pds
{

std::unique_ptr<prometheus::Exposer>
    createExposer(const std::string& prometheusDir,
                  const std::string& hostname);

struct Previous
{
    uint64_t value;
    std::chrono::steady_clock::time_point time;
};

class PromHistogram
{
public:
    using BucketBoundaries = std::vector<double>;

    PromHistogram(unsigned numBins, double binWidth, double binMin);

    void observe(double value);
    void collect(prometheus::MetricFamily& family);
    void clear();

private:
    BucketBoundaries      _boundaries;
    std::vector<uint64_t> _counts;
    double                _sum;
};

enum class MetricType
{
    Gauge,
    Counter,
    Rate,
    Histogram
};

class MetricExporter : public prometheus::Collectable
{
public:
    void add(const std::string& name,
             const std::map<std::string, std::string>& labels,
             MetricType type, std::function<uint64_t()> value);
    std::shared_ptr<PromHistogram>
         add(const std::string& name,
             const std::map<std::string, std::string>& labels,
             unsigned numBins, double binWidth=1.0, double binMin=0.0);
    std::vector<prometheus::MetricFamily> Collect() const override;
private:
    mutable std::vector<prometheus::MetricFamily> m_families;
    std::vector<std::function<uint64_t()> > m_values;
    mutable std::vector<std::shared_ptr<PromHistogram> > m_histos;
    std::vector<MetricType> m_type;
    mutable std::vector<Previous> m_previous;
};

template<typename T>
void setValue(prometheus::MetricFamily& family, const T& value)
{
    switch (family.type) {
        case prometheus::MetricType::Counter: {
            family.metric[0].counter.value = static_cast<double>(value);
            break;
        }
        case prometheus::MetricType::Gauge: {
            family.metric[0].gauge.value = static_cast<double>(value);
            break;
        }
        default:
            std::cout<<"wrong type\n";
    }
}

};
