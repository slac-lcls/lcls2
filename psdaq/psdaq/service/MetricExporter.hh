#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <functional>
#include <prometheus/exposer.h>
#include <prometheus/metric_type.h>
#include <prometheus/metric_family.h>

namespace Pds
{

std::unique_ptr<prometheus::Exposer>
    createExposer(const std::string& prometheusDir,
                  const std::string& hostname);

// Entry for allowing app to attempt to pick the same port on each invocation
std::unique_ptr<prometheus::Exposer>
    createExposer(const std::string& prometheusDir,
                  const std::string& hostname,
                  unsigned           portOffset);

struct Previous
{
    int64_t value;
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
    BucketBoundaries      m_boundaries;
    std::vector<uint64_t> m_counts;
    double                m_sum;
};

enum class MetricType
{
    Gauge,
    Counter,
    Rate,
    Constant,                           // To be used only by addConst()
    Histogram,
    Float
};

class MetricExporter : public prometheus::Collectable
{
public:
    int find(const std::string& name) const;
    void add(const std::string& name,
             const std::map<std::string, std::string>& labels,
             MetricType type, std::function<int64_t()> value);
    void addFloat(const std::string& name,
                  const std::map<std::string, std::string>& labels,
                  std::function<bool(double&)> value);
    void constant(const std::string& name,
                  const std::map<std::string, std::string>& labels,
                  int64_t value);
    std::shared_ptr<PromHistogram>
         histogram(const std::string& name,
                   const std::map<std::string, std::string>& labels,
                   unsigned numBins, double binWidth=1.0, double binMin=0.0);
    std::vector<prometheus::MetricFamily> Collect() const override;
private:
    void _erase(unsigned index);
private:
    mutable std::mutex m_mutex;
    mutable std::vector<prometheus::MetricFamily> m_families;
    std::vector<std::function<int64_t()> > m_values;
    std::vector<std::function<bool(double&)> > m_floats;
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

}
