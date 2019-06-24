#include <iostream>
#include "MetricExporter.hh"


prometheus::MetricFamily makeMetric(const std::string& name,
                                    const std::map<std::string, std::string>& labels,
                                    prometheus::MetricType type)
{
    prometheus::MetricFamily family;
    family.name = name;
    family.type = type;

    prometheus::ClientMetric metric;
    for (const auto& labelPair : labels) {
        auto label = prometheus::ClientMetric::Label{};
        label.name = labelPair.first;
        label.value = labelPair.second;
        metric.label.emplace_back(label);

    }
    family.metric.emplace_back(metric);
    return family;
}

void MetricExporter::add(const std::string& name,
         const std::map<std::string, std::string>& labels,
         MetricType type, std::function<uint64_t()> value)
{
    prometheus::MetricType prometheusType;
    switch (type) {
        case MetricType::Gauge:
            prometheusType = prometheus::MetricType::Gauge;
            break;
        case MetricType::Counter:
            prometheusType = prometheus::MetricType::Counter;
            break;
        case MetricType::Rate:
            prometheusType = prometheus::MetricType::Gauge;
            break;
    }
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_values.push_back(value);
    m_type.push_back(type);
    Previous previous;
    if (type == MetricType::Rate) {
        previous.value = value();
        previous.time = std::chrono::steady_clock::now();
    }
    m_previous.push_back(previous);

}

std::vector<prometheus::MetricFamily> MetricExporter::Collect()
{
    // std::cout<<"Collect()\n";
    for (size_t i=0; i<m_values.size(); i++) {
        //std::cout<<"Collector  "<<m_families[i].name<<'\n';
        if (m_type[i] == MetricType::Rate) {
            uint64_t newValue = m_values[i]();
            auto now = std::chrono::steady_clock::now();
            uint64_t previousValue = m_previous[i].value;
            uint64_t difference = 0UL;
            if (newValue > previousValue) {
                difference = newValue - previousValue;
            }
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - m_previous[i].time).count();
            double rate = static_cast<double>(difference) / static_cast<double>(duration) * 1.0e6;
            setValue(m_families[i], rate);
            m_previous[i].value = newValue;
            m_previous[i].time = now;
        }
        else {
            setValue(m_families[i], m_values[i]());
        }
    }
    return m_families;
}
