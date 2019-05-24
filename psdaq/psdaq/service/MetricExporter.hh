#pragma once

#include <iostream>
#include <vector>
#include <chrono>
#include <prometheus/exposer.h>

struct Previous
{
    uint64_t value;
    std::chrono::steady_clock::time_point time;
};

enum class MetricType
{
    Gauge,
    Counter,
    Rate
};

class MetricExporter : public prometheus::Collectable
{
public:
    void add(const std::string& name,
             const std::map<std::string, std::string>& labels,
             MetricType type, std::function<uint64_t()> value);
    std::vector<prometheus::MetricFamily> Collect() override;
private:
    std::vector<prometheus::MetricFamily> m_families;
    std::vector<std::function<uint64_t()> > m_values;
    std::vector<MetricType> m_type;
    std::vector<Previous> m_previous;
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
