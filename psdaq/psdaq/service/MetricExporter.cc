#include <iostream>
#include "MetricExporter.hh"


static
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

void Pds::MetricExporter::add(const std::string& name,
                              const std::map<std::string, std::string>& labels,
                              Pds::MetricType type, std::function<uint64_t()> value)
{
    prometheus::MetricType prometheusType;
    switch (type) {
        case Pds::MetricType::Gauge:
            prometheusType = prometheus::MetricType::Gauge;
            break;
        case Pds::MetricType::Counter:
            prometheusType = prometheus::MetricType::Counter;
            break;
        case Pds::MetricType::Rate:
            prometheusType = prometheus::MetricType::Gauge;
            break;
        default:
          std::cout<<"Bad type: "<<int(type)<<"\n";
    }
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_values.push_back(value);
    m_histos.push_back(nullptr);        // Placeholder; not used
    m_type.push_back(type);
    Pds::Previous previous;
    if (type == Pds::MetricType::Rate) {
        previous.value = value();
        previous.time = std::chrono::steady_clock::now();
    }
    m_previous.push_back(previous);
}

std::vector<prometheus::MetricFamily> Pds::MetricExporter::Collect() const
{
    // std::cout<<"Collect()\n";
    for (size_t i=0; i<m_type.size(); i++) {
        //std::cout<<"Collector  "<<m_families[i].name<<'\n';
        if (m_type[i] == Pds::MetricType::Rate) {
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
        else if (m_type[i] == Pds::MetricType::Histogram) {
            m_histos[i]->collect(m_families[i]);
        }
        else {
            setValue(m_families[i], m_values[i]());
        }
    }
    return m_families;
}


Pds::PromHistogram::PromHistogram(unsigned numBins, double binWidth, double binMin) :
  _boundaries(numBins),
  _counts    (numBins + 1),
  _sum       (0.0)
{
    //if (numBins)
    //    printf("PromHisto: numBins %u, binWidth %f, binMin %f\n", numBins, binWidth, binMin);
    for (unsigned i = 0; i < _boundaries.size(); ++i) {
        _boundaries[i] = binMin + binWidth * static_cast<double>(i);
    }
}

std::shared_ptr<Pds::PromHistogram>
    Pds::MetricExporter::add(const std::string& name,
                             const std::map<std::string, std::string>& labels,
                             unsigned numBins, double binWidth, double binMin)
{
    Pds::MetricType type = Pds::MetricType::Histogram;
    prometheus::MetricType prometheusType = prometheus::MetricType::Histogram;
    m_families.emplace_back(makeMetric(name, labels, prometheusType));
    m_families.back().metric[0].histogram.bucket.resize(numBins + 1);
    m_values.push_back(nullptr);        // Placeholder; won't be called
    m_histos.push_back(std::make_shared<PromHistogram>(numBins, binWidth, binMin));
    m_type.push_back(type);

    //for (unsigned i = 0; i < m_type.size(); ++i)
    //  printf("type[%u] %d\n", i, int(m_type[i]));
    return m_histos.back();
}

void Pds::PromHistogram::collect(prometheus::MetricFamily& family)
{
    auto& metric = family.metric[0];

    auto cumulative_count = 0ULL;
    for (std::size_t i = 0; i < _counts.size(); i++) {
        cumulative_count += _counts[i];
        auto& bucket = metric.histogram.bucket[i];
        bucket.cumulative_count = cumulative_count;
        bucket.upper_bound = (i == _boundaries.size()
                           ? std::numeric_limits<double>::infinity()
                           : _boundaries[i]);
        //printf("collect: i %zd, cnt %lu, ub %f\n", i, bucket.cumulative_count, bucket.upper_bound);
    }
    metric.histogram.sample_count = cumulative_count;
    metric.histogram.sample_sum = _sum;

    //printf("collect: count %llu, sum %f\n", cumulative_count, _sum);
    //for (unsigned i = 0; i < metric.histogram.bucket.size(); ++i)
    //    printf("collect: bucket[%u] cnt %lu, ub %f\n",
    //           i, metric.histogram.bucket[i].cumulative_count, metric.histogram.bucket[i].upper_bound);
}

void Pds::PromHistogram::observe(double sample)
{
    auto bin = static_cast<std::size_t>(std::distance(
        _boundaries.begin(),
        std::find_if(_boundaries.begin(), _boundaries.end(),
                     [sample](double boundary) { return boundary >= sample; })));
    if (bin >= 0 && bin < _counts.size()) {
        _counts[bin]++;
        _sum += sample;
    }
    else {
      fprintf(stderr, "%s:\n  Bin number %zd is out of range [0, %zd) for sample %f ([%f - %f))\n",
              __PRETTY_FUNCTION__, bin, _counts.size(), sample, *_boundaries.begin(), *_boundaries.end());
    }
    //printf("observe: sample %f, bin %zd, counts %lu, sum %f\n",
    //       sample, bin, _counts[bin], _sum);
}

void Pds::PromHistogram::clear()
{
    for (unsigned i = 0; i < _counts.size(); ++i)
        _counts[i] = 0;
    _sum = 0.0;
}
